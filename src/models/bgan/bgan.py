from helpers.checkpoint import Checkpoint
from torch.autograd import Variable
from helpers.logger import Logger
from helpers.initialization import weights_init_xavier
from helpers import data_loader
from datetime import datetime
import torch.nn as nn
import numpy as np
import torch
import os

ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
CWD = os.path.dirname(os.path.abspath(__file__))
RUN = datetime.today().strftime('%Y-%m-%d/%H-%M-%S')

CUDA = True if torch.cuda.is_available() else False
TENSOR = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

LOGGER = None


def boundary_seeking_loss(y_pred):
    """Boundary seeking loss. Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/"""
    return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2)


def run(args):
    global LOGGER
    global RUN

    img_shape = (1, args.img_size, args.img_size)  # one channel only

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
                nn.Tanh()
            )

        def forward(self, z_batch):
            img = self.model(z_batch)
            img = img.view(img.size(0), *img_shape)
            return img

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        def forward(self, img):
            img_flat = img.view(img.shape[0], -1)
            validity = self.model(img_flat)
            return validity

    # Define losses
    discriminator_loss = torch.nn.BCELoss()
    generator_loss = boundary_seeking_loss

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # Initialize optimizers for generator and discriminator
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.g_lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr)

    # Map to CUDA if necessary
    if CUDA:
        generator.cuda()
        discriminator.cuda()
        discriminator_loss.cuda()

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
        'binary': False,
        'crop': 20
    }
    mnist_loader = data_loader.load(args, opts)

    for epoch in range(current_epoch, args.n_epochs):
        for i, imgs in enumerate(mnist_loader):

            # Adversarial ground truths with noise
            valid = 0.8 + torch.rand(imgs.shape[0], 1).type(TENSOR) * 0.3
            valid = Variable(valid, requires_grad=False)
            fake = torch.rand(imgs.shape[0], 1).type(TENSOR) * 0.3
            fake = Variable(fake, requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(TENSOR))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_g.zero_grad()

            # Sample noise as generator input
            z = Variable(TENSOR(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

            # Generate a batch of images
            fake_images = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = generator_loss(discriminator(fake_images))

            g_loss.backward()
            optimizer_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_d.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_scores = discriminator(real_imgs)
            real_loss = discriminator_loss(real_scores, valid)
            fake_scores = discriminator(fake_images.detach())
            fake_loss = discriminator_loss(fake_scores, fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_d.step()

            batches_done = epoch * len(mnist_loader) + i + 1
            if batches_done % args.sample_interval == 0:
                LOGGER.log_generated_sample(fake_images, batches_done)

                LOGGER.log_batch_statistics(epoch, args.n_epochs, i + 1, len(mnist_loader), d_loss, g_loss, real_scores,
                                            fake_scores)

                LOGGER.log_tensorboard_basic_data(g_loss, d_loss, real_scores, fake_scores, batches_done)

                if args.log_details:
                    if args.log_details:
                        if batches_done == args.sample_interval:
                            LOGGER.save_image_grid(real_imgs, fake_images, batches_done)
                        else:
                            LOGGER.save_image_grid(None, fake_images, batches_done)
        # -- Save model checkpoints after each epoch -- #
        checkpoint_g.save(RUN, epoch)
        checkpoint_d.save(RUN, epoch)
    LOGGER.close_writers()
