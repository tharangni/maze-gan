# Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/

from datetime import datetime
import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from helpers import MazeHelper
from helpers import Checkpoint
from helpers import Bunch

ROOT = os.path.join('..', '..', '..', '..')

os.makedirs('images', exist_ok=True)
os.makedirs('mazes', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--n_examples', type=int, default=60000, help='number of training examples to use')
parser.add_argument('--batch_size', type=int, default=60, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--maze_size', type=int, default=8, help='size of each maze dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image samples')
parser.add_argument('--detailed_log', type=bool, default=False, help='whether to log weight and gradient histograms')
parser.add_argument('--temp', type=float, default=0.2, help='gumbel softmax temperature')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.maze_size, opt.maze_size)

cuda = True if torch.cuda.is_available() else False


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
            *block(opt.latent_dim, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, int(np.prod(img_shape)) * 2),
        )
        self.LogSoftmax = nn.LogSoftmax(dim=-1)

    def sample_gumbel(self, shape, eps=1e-20):
        u = torch.rand(shape).type(Tensor)
        return -Variable(torch.log(-torch.log(u + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, tau):
        y = self.gumbel_softmax_sample(logits, tau)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1]).type(Tensor)
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

    def pass_through(self, logits, tau):
        # sample = F.gumbel_softmax(logits.view(-1, 2), tau=tau, hard=True)[:, 1]
        sample = self.gumbel_softmax(logits, tau)
        return sample[:, :, 1]

    def forward(self, z):
        img_probs = self.model(z)
        img_logits = self.LogSoftmax(img_probs.view(opt.batch_size, -1, 2))
        img = self.pass_through(img_logits, opt.temp)
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


adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

now = datetime.today().strftime('%Y-%m-%d/%H-%M-%S')
writer = SummaryWriter(log_dir=os.path.join('runs', now))

Tensor = torch.FloatTensor
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    Tensor = torch.cuda.FloatTensor

# Configure data loader
data_path = os.path.join(ROOT, 'data', 'mazes', '{}.{}x{}.data'.format(opt.n_examples, opt.maze_size, opt.maze_size))
maze_data = torch.load(data_path).reshape(-1, opt.batch_size, 1, opt.maze_size, opt.maze_size).type(Tensor)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

for epoch in range(opt.n_epochs):
    for i, real_mazes in enumerate(maze_data):

        # Adversarial ground truths
        valid = Variable(torch.ones(real_mazes.shape[0], 1).type(Tensor), requires_grad=False)
        fake = Variable(torch.zeros(real_mazes.shape[0], 1).type(Tensor), requires_grad=False)

        # Configure input
        real_mazes = Variable(real_mazes)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(torch.randn(real_mazes.shape[0], opt.latent_dim).type(Tensor))

        # Generate a batch of images
        fake_mazes = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(fake_mazes), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_scores = discriminator(real_mazes)
        real_loss = adversarial_loss(real_scores, valid)
        fake_scores = discriminator(fake_mazes.detach())
        fake_loss = adversarial_loss(fake_scores, fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(maze_data) + i
        if batches_done % opt.sample_interval == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [D(x): %.2f] [D(G(z)): %.2f]" %
                  (epoch + 1, opt.n_epochs, i + 1, len(maze_data), d_loss.item(), g_loss.item(),
                   real_scores.detach().mean().item(), fake_scores.detach().mean().item()))

            if batches_done == 0:
                # MazeHelper.save_maze_grid(real_mazes.detach().view(-1, opt.maze_size, opt.maze_size).data.cpu().numpy(),
                #                           'images/real_mazes.png')
                torch.save(real_mazes.detach().view(-1, opt.maze_size, opt.maze_size).cpu(),
                           'mazes/fake_%d.sample' % batches_done)
            # MazeHelper.save_maze_grid(fake_mazes.detach().view(-1, opt.maze_size, opt.maze_size).data.cpu().numpy(),
            #                           'images/fake_%d.png' % batches_done)
            torch.save(fake_mazes.detach().view(-1, opt.maze_size, opt.maze_size).cpu(),
                       'mazes/fake_%d.sample' % batches_done)

            writer.add_scalar('Generator/loss', g_loss.item(), batches_done)
            writer.add_scalar('Discriminator/loss', d_loss.item(), batches_done)
            writer.add_scalar('D(x)', real_scores.detach().mean().item(), batches_done)
            writer.add_scalar('D(G(z))', fake_scores.detach().mean().item(), batches_done)

            if opt.detailed_log:
                for name, param in generator.named_parameters():
                    name = name.replace('.', '/')
                    writer.add_histogram("Generator/" + name, param.data.cpu().numpy(),
                                         batches_done, bins='auto')
                    writer.add_histogram("Generator/" + name + '/grad', param.grad.data.cpu().numpy(),
                                         batches_done, bins='auto')

                for name, param in discriminator.named_parameters():
                    name = name.replace('.', '/')
                    writer.add_histogram("Discriminator/" + name, param.data.cpu().numpy(),
                                         batches_done, bins='auto')
                    writer.add_histogram("Discriminator/" + name + '/grad', param.grad.data.cpu().numpy(),
                                         batches_done, bins='auto')
    # -- Save models after each epoch -- #
    generator_checkpoint = Bunch(
        epoch=epoch + 1, state_dict=generator.state_dict(),
        optimizer=optimizer_G.state_dict()
    )
    Checkpoint(generator_checkpoint, 'generator').save()
    discriminator_checkpoint = Bunch(
        epoch=epoch + 1, state_dict=discriminator.state_dict(),
        optimizer=optimizer_D.state_dict()
    )
    Checkpoint(discriminator_checkpoint, 'discriminator').save()
writer.close()
