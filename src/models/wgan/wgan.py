import argparse
import os
from datetime import datetime

import numpy as np

from torch.autograd import Variable

import torch.nn as nn
import torch

from helpers.checkpoint import Checkpoint
from helpers.initialization import weights_init_xavier
from helpers.logger import Logger
from helpers import st_gumbel_softmax, data_loader

ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
CWD = os.path.dirname(os.path.abspath(__file__))
RUN = datetime.today().strftime('%Y-%m-%d/%H-%M-%S')

CUDA = True if torch.cuda.is_available() else False
TENSOR = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

LOGGER = None

os.makedirs('images', exist_ok=True)


def run(args: argparse.Namespace):
    global LOGGER
    global RUN

    img_shape = (1, args.img_size, args.img_size)

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))

                return layers

            self.model = nn.Sequential(
                *block(args.latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
            )

            self.out = nn.LogSigmoid()

        def forward(self, z_batch):
            linear = self.model(z_batch)

            white_prob = self.out(linear).view(args.batch_size, args.img_size ** 2, 1)
            black_prob = self.out(-linear).view(args.batch_size, args.img_size ** 2, 1)
            probs = torch.cat([black_prob, white_prob], dim=-1)
            img = st_gumbel_softmax.straight_through(probs, args.temp, True)

            return img.view(img.shape[0], *img_shape)

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1)
            )

        def forward(self, img):
            img_flat = img.view(img.shape[0], -1)
            validity = self.model(img_flat)

            return validity

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # Optimizers
    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=args.g_lr)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=args.d_lr)

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
    mnist_loader = data_loader.load(args, opts)

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(current_epoch, args.n_epochs):
        for i, imgs in enumerate(mnist_loader):
            # Configure input
            real_imgs = Variable(imgs.type(TENSOR))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_d.zero_grad()

            # Sample noise as generator input
            z = Variable(torch.randn(imgs.shape[0], args.latent_dim).type(TENSOR))

            # Generate a batch of images
            fake_images = generator(z).detach()
            # Adversarial loss
            loss_d = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_images))

            loss_d.backward()
            optimizer_d.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

            # Train the generator every n_critic iterations
            if i % args.n_critic == 0:
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
                    LOGGER.log_generated_sample(fake_images, batches_done)

                    LOGGER.log_batch_statistics(epoch, args.n_epochs, i, len(mnist_loader), loss_d, loss_g)

                    LOGGER.log_tensorboard_basic_data(loss_g, loss_d, step=batches_done)

                    if args.log_details:
                        if batches_done == args.sample_interval:
                            LOGGER.save_image_grid(real_imgs, fake_images, batches_done)
                        else:
                            LOGGER.save_image_grid(None, fake_images, batches_done)
            batches_done += 1
        # -- Save model checkpoints after each epoch -- #
        checkpoint_g.save(RUN, epoch)
        checkpoint_d.save(RUN, epoch)
