from helpers.checkpoint import Checkpoint
from helpers import st_gumbel_softmax
from torch.autograd import Variable
from helpers.logger import Logger
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


def run(opt):
    global LOGGER
    global RUN

    img_shape = (1, opt.img_size, opt.img_size)

    # noinspection PyMethodMayBeStatic
    class Generator(nn.Module):
        LSTM_OUT = 64

        def __init__(self):
            super(Generator, self).__init__()

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))

                return layers

            self.map1 = nn.Sequential(
                *block(opt.latent_dim, int(np.prod(img_shape)))
            )
            self.lstm = nn.LSTM(1, self.LSTM_OUT, bidirectional=True)
            self.model = nn.Sequential(
                *block(int(self.LSTM_OUT * 2), int(self.LSTM_OUT * 4), normalize=False),
                *block(int(self.LSTM_OUT * 4), 2, normalize=False)
            )
            self.out = nn.LogSoftmax(dim=-1)

            self.hidden_state = self.init_hidden_state()

        def init_hidden_state(self):
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            return (torch.zeros(2, opt.batch_size, self.LSTM_OUT).type(TENSOR),
                    torch.zeros(2, opt.batch_size, self.LSTM_OUT).type(TENSOR))

        def forward(self, z_batch):
            map1 = self.map1(z_batch)

            hidden_states, self.hidden_state = self.lstm(
                map1.view(int(np.prod(img_shape)), opt.batch_size, 1), self.hidden_state
            )

            intermediate = self.model(hidden_states.view(opt.batch_size, int(np.prod(img_shape)), self.LSTM_OUT * 2))
            img_logits = self.out(intermediate)
            img = st_gumbel_softmax.straight_through(img_logits, opt.temp, True)

            return img.view(img.size(0), *img_shape)

    # noinspection PyMethodMayBeStatic
    class Discriminator(nn.Module):
        LSTM_OUT = 64

        def __init__(self):
            super(Discriminator, self).__init__()

            self.lstm = nn.LSTM(1, self.LSTM_OUT, bidirectional=True)
            self.model = nn.Sequential(
                nn.Linear(int(self.LSTM_OUT * 2), self.LSTM_OUT / 4),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.out = nn.Sequential(
                nn.Linear(int(self.LSTM_OUT / 4 * np.prod(img_shape)), 1),
                nn.Sigmoid()
            )

            self.hidden_state = self.init_hidden_state()

        def init_hidden_state(self):
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            return (torch.zeros(2, opt.batch_size, self.LSTM_OUT).type(TENSOR),
                    torch.zeros(2, opt.batch_size, self.LSTM_OUT).type(TENSOR))

        def forward(self, img_batch):
            hidden_states, self.hidden_state = self.lstm(img_batch.view(int(np.prod(img_shape)), opt.batch_size, 1))
            intermediate = self.model(hidden_states.view(opt.batch_size, int(np.prod(img_shape)), self.LSTM_OUT * 2))
            validity = self.out(intermediate.view(opt.batch_size, -1))

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
        RUN, current_epoch = checkpoint_g.load()
        _, _ = checkpoint_d.load()
        LOGGER = Logger(CWD, RUN)
        print('Loaded models from disk. Starting at epoch {}.'.format(current_epoch + 1))
    else:
        LOGGER = Logger(CWD, RUN)

    # Configure data loader
    mnist_loader = data_loader.mnist(opt, binary=True)

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

            # Initialise generator hidden state
            generator.hidden_state = generator.init_hidden_state()

            # Sample noise as generator input
            z = Variable(torch.randn(imgs.size(0), opt.latent_dim).type(TENSOR))

            # Generate a batch of images
            fake_images = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(fake_images), valid)

            g_loss.backward()
            optimizer_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_d.zero_grad()

            # Initialise generator hidden state
            discriminator.hidden_state = discriminator.init_hidden_state()

            # Measure discriminator's ability to classify real from generated samples
            real_scores = discriminator(real_imgs)
            real_loss = adversarial_loss(real_scores, valid)
            fake_scores = discriminator(fake_images.detach())
            fake_loss = adversarial_loss(fake_scores, fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_d.step()

            batches_done = epoch * len(mnist_loader) + i + 1
            if batches_done % opt.sample_interval == 0:
                LOGGER.log_generated_sample(fake_images, batches_done)

                LOGGER.log_batch_statistics(epoch, opt.n_epochs, i + 1, len(mnist_loader), d_loss, g_loss, real_scores,
                                            fake_scores)

                LOGGER.log_tensorboard_basic_data(g_loss, d_loss, real_scores, fake_scores, batches_done)

                if opt.log_details:
                    LOGGER.save_image_grid(real_imgs, fake_images, batches_done)
                    LOGGER.log_tensorboard_parameter_data(discriminator, generator, batches_done)
        # -- Save model checkpoints after each epoch -- #
        checkpoint_g.save(RUN, epoch)
        checkpoint_d.save(RUN, epoch)
    LOGGER.close_writers()
