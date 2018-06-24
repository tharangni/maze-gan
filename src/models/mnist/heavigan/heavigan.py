from helpers.checkpoint import Checkpoint
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from helpers import data_loader
from helpers import st_heaviside
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


def run(opt):
    img_shape = (1, opt.img_size, opt.img_size)

    # noinspection PyMethodMayBeStatic
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
                *block(opt.latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )

        def forward(self, z):
            img = st_heaviside.straight_through(self.model(z))

            return img.view(img.size(0), *img_shape)

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

    # Create checkpoint handler and load state if required
    current_epoch = 0
    checkpoint_g = Checkpoint(CWD, generator, optimizer_g)
    checkpoint_d = Checkpoint(CWD, discriminator, optimizer_d)
    if opt.resume:
        global NOW
        global WRITER
        NOW, current_epoch = checkpoint_g.load()
        _, _ = checkpoint_d.load()
        WRITER = SummaryWriter(log_dir=os.path.join(CWD, 'runs', NOW))
        print('Loaded models from disk. Starting at epoch {}'.format(current_epoch + 1))

    # Configure data loader
    mnist_loader = data_loader.mnist(opt, True)

    for epoch in range(current_epoch, opt.n_epochs):
        for i, imgs in enumerate(mnist_loader):

            # Adversarial ground truths
            valid = Variable(torch.ones(imgs.shape[0], 1).type(TENSOR), requires_grad=False)
            fake = Variable(torch.zeros(imgs.shape[0], 1).type(TENSOR), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_g.zero_grad()

            # Sample noise as generator input
            z = Variable(torch.randn(imgs.shape[0], opt.latent_dim).type(TENSOR))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_d.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_scores = discriminator(real_imgs)
            real_loss = adversarial_loss(real_scores, valid)
            fake_scores = discriminator(gen_imgs.detach())
            fake_loss = adversarial_loss(fake_scores, fake)
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
                    logger.save_image_grid(CWD, NOW, real_imgs, gen_imgs, batches_done)
                    logger.log_tensorboard_parameter_data(WRITER, batches_done, generator, discriminator)
        # -- Save model checkpoints after each epoch -- #
        checkpoint_g.save(NOW, epoch)
        checkpoint_d.save(NOW, epoch)
    WRITER.close()
