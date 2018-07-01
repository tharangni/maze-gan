from argparse import Namespace

from helpers.initialization import weights_init_xavier
from helpers.checkpoint import Checkpoint
from torch.autograd import Variable
from helpers.logger import Logger
from helpers import data_loader, st_gumbel_softmax, maze_utils
from datetime import datetime
import torch.nn as nn
import torch
import os
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import math

ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
CWD = os.path.dirname(os.path.abspath(__file__))
RUN = datetime.today().strftime('%Y-%m-%d/%H-%M-%S')

CUDA = True if torch.cuda.is_available() else False
TENSOR = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

LOGGER = None


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm1d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def run(args: Namespace):
    global LOGGER
    global RUN

    # noinspection PyMethodMayBeStatic
    class Generator(nn.Module):

        def __init__(self):
            super(Generator, self).__init__()

            self.init_size = args.maze_size ** 2 // 4
            self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128 * self.init_size))
            self.model = nn.Sequential(
                nn.BatchNorm1d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm1d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(64, 1, 3, stride=1, padding=1),
            )

            self.out = nn.LogSigmoid()

        def forward(self, z):
            map1 = self.l1(z)
            map1 = map1.view(map1.size(0), 128, self.init_size)
            conv = self.model(map1).view(args.batch_size, args.img_size ** 2, 1)

            white_prob = self.out(conv).view(args.batch_size, args.img_size ** 2, 1)
            black_prob = self.out(-conv).view(args.batch_size, args.img_size ** 2, 1)

            probs = torch.cat([black_prob, white_prob], dim=-1)
            img = st_gumbel_softmax.straight_through(probs, args.temp, True)

            return img.view(args.batch_size, 1, args.img_size ** 2)

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            def discriminator_block(in_filters, out_filters, bn=True):
                block = [nn.Conv1d(in_filters, out_filters, 3, 2, 1),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Dropout(0.25)]
                if bn:
                    block.append(nn.BatchNorm1d(out_filters, 0.8))
                return block

            self.model = nn.Sequential(
                *discriminator_block(1, 16, bn=False),
                *discriminator_block(16, 32, bn=False),
                *discriminator_block(32, 64, bn=False),
                *discriminator_block(64, 128, bn=False),
            )

            # The height and width of down sampled image
            ds_size = math.ceil((args.img_size ** 2) / 4 ** 2)
            self.adv_layer = nn.Linear(128 * ds_size, 1)

        def forward(self, maze):
            out = self.model(maze)
            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)

            return validity

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # Initialize optimizers for generator and discriminator
    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=args.g_lr)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=args.d_lr)

    # Map to CUDA if necessary
    if CUDA:
        generator.cuda()
        discriminator.cuda()

    # Initialize weights
    generator.apply(weights_init_xavier)
    discriminator.apply(weights_init_xavier)

    # Create checkpoint handler and load state if required
    current_epoch = 0
    checkpoint_g = Checkpoint(CWD, generator, optimizer_g)
    checkpoint_d = Checkpoint(CWD, discriminator, optimizer_d)
    if args.resume:
        RUN, current_epoch = checkpoint_g.load()
        _, _ = checkpoint_d.load()
        LOGGER = Logger(CWD, RUN, args)
        print('Loaded models from disk. Starting at epoch {}.'.format(current_epoch + 1))
    else:
        LOGGER = Logger(CWD, RUN, args)

    # Configure data loader
    opts = {
        'binary': True,
    }
    batched_data = data_loader.load(args, opts)

    for epoch in range(current_epoch, args.n_epochs):
        for i, mazes in enumerate(batched_data):
            batches_done = epoch * len(batched_data) + i + 1

            mazes = mazes.reshape(args.batch_size, 1, -1).type(TENSOR)

            # Configure input
            real_images = Variable(mazes)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_d.zero_grad()

            z = Variable(torch.randn(real_images.size(0), args.latent_dim).type(TENSOR))
            fake_images = generator(z).detach()
            # Adversarial loss
            loss_d = -torch.mean(discriminator(real_images)) + torch.mean(discriminator(fake_images))

            loss_d.backward()
            optimizer_d.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

            # Train the generator every n_critic iterations
            if batches_done % args.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_g.zero_grad()

                # Generate a batch of images
                fake_images = generator(z)
                # Adversarial loss
                loss_g = -torch.mean(discriminator(fake_images))

                loss_g.backward()
                optimizer_g.step()

                if batches_done % args.sample_interval == 0:
                    fake_mazes = fake_images.reshape(fake_images.size(0), args.img_size, args.img_size)
                    fake_mazes[fake_mazes < 0.5] = 0
                    fake_mazes[fake_mazes > 0.5] = 1
                    real_mazes = real_images.reshape(real_images.size(0), args.img_size, args.img_size)

                    LOGGER.log_generated_sample(fake_mazes, batches_done)

                    LOGGER.log_batch_statistics(epoch, args.n_epochs, i + 1, len(batched_data), loss_d, loss_g)

                    LOGGER.log_tensorboard_basic_data(loss_g, loss_d, step=batches_done)

                    if args.log_details:
                        if batches_done == args.sample_interval:
                            LOGGER.save_image_grid(real_mazes, fake_mazes, batches_done)
                        else:
                            LOGGER.save_image_grid(None, fake_images, batches_done)

        # -- Save model checkpoints after each epoch -- #
        checkpoint_g.save(RUN, epoch)
        checkpoint_d.save(RUN, epoch)
    LOGGER.close_writers()
