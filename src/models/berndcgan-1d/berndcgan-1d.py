
from helpers.initialization import weights_init_xavier
from helpers.checkpoint import Checkpoint
from torch.autograd import Variable
from helpers.logger import Logger
from helpers import data_loader
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


def run(opt):
    global LOGGER
    global RUN

    # noinspection PyMethodMayBeStatic
    class Generator(nn.Module):

        def __init__(self):
            super(Generator, self).__init__()

            self.init_size = opt.maze_size**2 // 4
            self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size))
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
                nn.Sigmoid()
            )
    
        def forward(self, z):    
            map1 = self.l1(z)
            map1 = map1.view(map1.shape[0], 128, self.init_size)
            out = self.model(map1)

            img = RelaxedBernoulli(torch.tensor([opt.temp]).type(TENSOR), probs=out).rsample()
            
            return img

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
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )

            # The height and width of downsampled image
            ds_size = math.ceil((opt.maze_size **2) / 4**2)
            self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**1, 1),
                                        nn.Sigmoid())

        def forward(self, maze):
            print(maze.size())
            out = self.model(maze)
            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)

            return validity

    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # Initialize optimizers for generator and discriminator
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.g_lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr)

    # Map to CUDA if necessary
    if CUDA:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_xavier)
    discriminator.apply(weights_init_xavier)

    # Create checkpoint handler and load state if required
    current_epoch = 0
    checkpoint_g = Checkpoint(CWD, generator, optimizer_g)
    checkpoint_d = Checkpoint(CWD, discriminator, optimizer_d)
    if opt.resume:
        RUN, current_epoch = checkpoint_g.load()
        _, _ = checkpoint_d.load()
        LOGGER = Logger(CWD, RUN, opt)
        print('Loaded models from disk. Starting at epoch {}.'.format(current_epoch + 1))
    else:
        LOGGER = Logger(CWD, RUN, opt)

    # Configure data loader
    maze_loader = data_loader.mazes(opt)

    for epoch in range(current_epoch, opt.n_epochs):
        for i, mazes in enumerate(maze_loader):
            mazes= mazes.reshape(opt.batch_size, -1).type(TENSOR).float()

            # Adversarial ground truths
#            valid = Variable(torch.ones(mazes.shape[0], 1).type(TENSOR), requires_grad=False)
#            fake = Variable(torch.zeros(mazes.shape[0], 1).type(TENSOR), requires_grad=False)
            # Adversarial ground truths with noise
            valid = 0.8 + torch.rand(mazes.shape[0], 1).type(TENSOR) * 0.3
            valid = Variable(valid, requires_grad=False)
            fake = torch.rand(mazes.shape[0], 1).type(TENSOR) * 0.3
            fake = Variable(fake, requires_grad=False)

            # Configure input
            real_mazes = Variable(mazes)
            
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_g.zero_grad()

            # Sample noise as generator input
            z = Variable(torch.randn(mazes.shape[0], opt.latent_dim).type(TENSOR))

            # Generate a batch of images
            fake_mazes = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(fake_mazes), valid)

            g_loss.backward()
            optimizer_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_d.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_mazes = real_mazes.unsqueeze_(1)
            real_scores = discriminator(real_mazes)
            real_loss = adversarial_loss(real_scores, valid)
            fake_scores = discriminator(fake_mazes.detach())
            fake_loss = adversarial_loss(fake_scores, fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_d.step()

            batches_done = epoch * len(maze_loader) + i + 1
            if batches_done % opt.sample_interval == 0:
                fake_mazes = fake_mazes.reshape(fake_mazes.size(0), opt.maze_size, opt.maze_size)
                fake_mazes[fake_mazes < 0.5] = 0
                fake_mazes[fake_mazes > 0.5] = 1
                real_mazes = real_mazes.reshape(real_mazes.size(0), opt.maze_size, opt.maze_size)
                LOGGER.log_generated_sample(fake_mazes, batches_done)

                LOGGER.log_batch_statistics(epoch, opt.n_epochs, i + 1, len(maze_loader), d_loss, g_loss, real_scores,
                                            fake_scores)

                LOGGER.log_tensorboard_basic_data(g_loss, d_loss, real_scores, fake_scores, batches_done)

                if opt.log_details:
                    LOGGER.save_image_grid(real_mazes, fake_mazes, batches_done)
                    # LOGGER.log_tensorboard_parameter_data(discriminator, generator, batches_done)
        # -- Save model checkpoints after each epoch -- #
        checkpoint_g.save(RUN, epoch)
        checkpoint_d.save(RUN, epoch)
    LOGGER.close_writers()
