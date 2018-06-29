from helpers.checkpoint import Checkpoint
from helpers.initialization import weights_init_xavier
from helpers import st_gumbel_softmax
from torch.autograd import Variable
from helpers.logger import Logger
from helpers import data_loader
from datetime import datetime
import torch.nn as nn
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
        def __init__(self):
            super(Generator, self).__init__()

            self.init_size = opt.img_size // 4
            self.filters = 32
            self.map1 = nn.Linear(opt.latent_dim, self.filters * self.init_size ** 2)
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(self.filters),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.filters, self.filters, 3, stride=1, padding=1),
                nn.BatchNorm2d(self.filters, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.filters, self.filters // 2, 3, stride=1, padding=1),
                nn.BatchNorm2d(self.filters // 2, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.filters // 2, 1, 3, stride=1, padding=1),
            )
            self.out = nn.LogSigmoid()

        def forward(self, z_batch):
            map1 = self.map1(z_batch).view(opt.batch_size, self.filters, self.init_size, self.init_size)
            conv = self.conv_blocks(map1)

            white_prob = self.out(conv).view(opt.batch_size, opt.img_size ** 2, 1)
            black_prob = self.out(-conv).view(opt.batch_size, opt.img_size ** 2, 1)

            probs = torch.cat([black_prob, white_prob], dim=-1)
            img = st_gumbel_softmax.straight_through(probs, opt.temp, True)

            return img.view(img.size(0), *img_shape)

    class Discriminator(nn.Module):

        def __init__(self):
            super(Discriminator, self).__init__()

            self.filters = 8

            def discriminator_block(in_filters, out_filters, step, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, step, 1),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Dropout2d(0.25)]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.model = nn.Sequential(
                *discriminator_block(1, self.filters, 1, bn=False),
                *discriminator_block(self.filters, self.filters * 2, 2),
                *discriminator_block(self.filters * 2, self.filters * 4, 1),
                *discriminator_block(self.filters * 4, self.filters * 8, 2),
            )

            # The height and width of downsampled image
            ds_size = opt.img_size // 2 ** 2
            self.adv_layer = nn.Sequential(
                nn.Linear(self.filters * 8 * ds_size ** 2, 1),
                nn.Sigmoid()
            )

        def forward(self, img):
            out = self.model(img)
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
        LOGGER = Logger(CWD, RUN)
        print('Loaded models from disk. Starting at epoch {}.'.format(current_epoch + 1))
    else:
        LOGGER = Logger(CWD, RUN)

    # Configure data loader
    mnist_loader = data_loader.mnist(opt, binary=True, is_image=True, crop=20)

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
            fake_images = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(fake_images), valid)

            g_loss.backward()
            optimizer_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_d.zero_grad()

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

                LOGGER.log_batch_statistics(epoch, opt.n_epochs, i + 1, len(mnist_loader), d_loss, g_loss,
                                            real_scores,
                                            fake_scores)

                LOGGER.log_tensorboard_basic_data(g_loss, d_loss, real_scores, fake_scores, batches_done)

                if opt.log_details:
                    LOGGER.save_image_grid(real_imgs, fake_images, batches_done)
                    LOGGER.log_tensorboard_parameter_data(discriminator, generator, batches_done)
        # -- Save model checkpoints after each epoch -- #
        checkpoint_g.save(RUN, epoch)
        checkpoint_d.save(RUN, epoch)
    LOGGER.writer.close()