from helpers.checkpoint import Checkpoint
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from helpers import data_loader
from datetime import datetime
from helpers import logger
import torch.nn as nn
import numpy as np
import torch
import os

CWD = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
CUDA = True if torch.cuda.is_available() else False
NOW = datetime.today().strftime('%Y-%m-%d/%H-%M-%S')
WRITER = SummaryWriter(log_dir=os.path.join(CWD, 'runs', NOW))
TENSOR = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def boundary_seeking_loss(y_pred):
    """Boundary seeking loss. Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/"""
    return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2)


def run(opt):
    img_shape = (1, opt.img_size, opt.img_size)  # one channel only

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
                *block(100, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )

        def forward(self, z):
            img = self.model(z)
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
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.g_lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr)

    # Map to CUDA if necessary
    if CUDA:
        generator.cuda()
        discriminator.cuda()
        discriminator_loss.cuda()

    # Create checkpoint handler and load state if required
    checkpoint_g = Checkpoint(CWD, generator, optimizer_g)
    checkpoint_d = Checkpoint(CWD, discriminator, optimizer_d)

    current_epoch = 0
    print()
    if opt.resume:
        current_epoch_1 = checkpoint_g.load()
        current_epoch_2 = checkpoint_d.load()
        if current_epoch_1 != current_epoch_2:
            raise ValueError('Epochs don`t match. Generator and Discriminator out of sync.')
        current_epoch = current_epoch_1

    # Configure data loader
    mnist_loader = data_loader.mnist(opt)

    for epoch in range(current_epoch, opt.n_epochs):
        for i, imgs in enumerate(mnist_loader):

            # Adversarial ground truths
            valid = Variable(TENSOR(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(TENSOR(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(TENSOR))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_g.zero_grad()

            # Sample noise as generator input
            z = Variable(TENSOR(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = generator_loss(discriminator(fake_imgs))

            g_loss.backward()
            optimizer_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_d.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_scores = discriminator(real_imgs)
            real_loss = discriminator_loss(real_scores, valid)
            fake_scores = discriminator(fake_imgs.detach())
            fake_loss = discriminator_loss(fake_scores, fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_d.step()

            batches_done = epoch * len(mnist_loader) + i
            if batches_done % opt.sample_interval == 0:
                logger.log_batch_statistics(epoch, opt.n_epochs, i, len(mnist_loader),
                                            d_loss, g_loss, real_scores, fake_scores)

                logger.log_tensorboard_basic_data(WRITER, batches_done, g_loss, d_loss,
                                                  real_scores, fake_scores)

                if opt.log_details:
                    logger.save_image_grid(CWD, real_imgs, fake_imgs, batches_done)
                    logger.log_tensorboard_parameter_data(WRITER, batches_done, generator, discriminator)
