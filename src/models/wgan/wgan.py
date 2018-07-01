import argparse
import os
from datetime import datetime

import numpy as np

import torchvision.transforms as transforms
from helpers import st_heaviside
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

from helpers.checkpoint import Checkpoint
from helpers.logger import Logger

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
                nn.Tanh()
            )

        def forward(self, z_batch):
            img = self.model(z_batch)
            img = img.view(img.shape[0], *img_shape)

            # return st_heaviside.straight_through(img)
            return img

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

    if CUDA:
        generator.cuda()
        discriminator.cuda()

    # Configure data loader
    os.makedirs('../../data/mnist', exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    # Optimizers
    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=args.g_lr)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=args.d_lr)

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

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(current_epoch, args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(TENSOR))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_d.zero_grad()

            # Sample noise as generator input
            z = Variable(TENSOR(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_d = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

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
                gen_imgs = generator(z)
                # Adversarial loss
                loss_g = -torch.mean(discriminator(gen_imgs))

                loss_g.backward()
                optimizer_g.step()

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.n_epochs,
                                                                                 batches_done % len(dataloader),
                                                                                 len(dataloader),
                                                                                 loss_d.item(), loss_g.item()))

            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], 'images/{0:0=6d}.png'.format(batches_done), nrow=5, normalize=True)
            batches_done += 1
