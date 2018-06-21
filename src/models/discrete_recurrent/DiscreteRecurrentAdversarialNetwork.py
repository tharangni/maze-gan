import os
import pickle
import torch.nn as nn
import torch.utils.data
import torch.random

from helpers.Checkpoint import Checkpoint
from helpers.MazeGenerator import gen_maze_data
from models.discrete_recurrent.Generator import Generator
from models.discrete_recurrent.Discriminator import Discriminator
from helpers.Bunch import Bunch


class DiscreteRecurrentAdversarialNetwork:

    def __init__(self, opts):
        self.name = 'RecurrentHeavisideMazes'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator_opts = Bunch(
            device=self.device, latent_size=opts.latent_size, hidden_size=opts.hidden_size, maze_size=opts.mx * opts.my,
            num_epochs=opts.num_epochs, batch_size=opts.batch_size, learning_rate=opts.g_lr,
            resume=opts.resume, temp=opts.temp, name=self.name
        )
        self.discriminator_opts = Bunch(
            device=self.device, hidden_size=opts.hidden_size, maze_size=opts.mx * opts.my,
            num_epochs=opts.num_epochs, batch_size=opts.batch_size, learning_rate=opts.d_lr,
            resume=opts.resume, name=self.name
        )
        self.latent_size = opts.latent_size
        self.hidden_size = opts.hidden_size
        self.num_epochs = opts.num_epochs
        self.batch_size = opts.batch_size
        self.mx = opts.mx
        self.my = opts.my
        self.N = opts.N
        self.results_path = os.path.join('results', self.name)
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        self.G = Generator(self.generator_opts)
        self.D = Discriminator(self.discriminator_opts)

        self.loss = nn.BCELoss().to(device=self.device)

    def train(self):
        # --- Maze Data --- #
        print('Generating {} {}x{} maze examples.'.format(self.N, self.mx, self.my))
        maze_data = gen_maze_data(self.N, self.mx, self.my)
        dump_file(os.path.join(self.results_path, 'real_mazes.pickle'), maze_data)

        data_loader = maze_data.reshape(-1, self.batch_size, self.mx, self.my)
        # -- Number of batches -- #
        num_batches = len(data_loader)

        # -- Generate labels -- #
        real = torch.ones([self.batch_size, 1], dtype=torch.float).to(self.device)
        fake = torch.zeros([self.batch_size, 1], dtype=torch.float).to(self.device)

        # --- Start training --- #
        g_fake_mazes = None
        for epoch in range(self.num_epochs):
            all_fake_mazes = None
            for batch_idx, real_mazes in enumerate(data_loader):
                real_mazes = real_mazes.reshape(self.batch_size, -1).to(self.device, dtype=torch.float32)

                # -- Reset gradients -- #
                self.D.optimizer.zero_grad()
                self.G.optimizer.zero_grad()

                # -- Train Discriminator -- #
                z = torch.rand(self.batch_size, self.latent_size).to(self.device)
                d_fake_mazes = self.G.forward(z)
                d_forward_args = Bunch(
                    real_mazes=real_mazes,
                    fake_mazes=d_fake_mazes.detach()
                )
                d_real_scores, d_fake_scores = self.D.forward(d_forward_args)
                d_backward_args = Bunch(
                    criterion=self.loss,
                    real_scores=d_real_scores,
                    fake_scores=d_fake_scores,
                    real=real,
                    fake=fake
                )
                d_loss = self.D.backward(d_backward_args)

                # -- Train Generator -- #
                z = torch.rand(self.batch_size, self.latent_size).to(self.device)
                g_fake_mazes = self.G.forward(z)
                g_fake_scores = self.D.model(g_fake_mazes)
                g_backward_args = Bunch(
                    criterion=self.loss,
                    scores=g_fake_scores,
                    real=real
                )
                g_loss = self.G.backward(g_backward_args)

                # -- Log events -- #
                if (batch_idx + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                          .format(epoch + 1, self.num_epochs, batch_idx + 1, num_batches, d_loss.item(), g_loss.item(),
                                  d_real_scores.mean().item(), g_fake_scores.mean().item()))

                # -- Save generated mazes -- #
                g_fake_mazes = g_fake_mazes.reshape(g_fake_mazes.size(0), self.mx, self.my)
                if all_fake_mazes is not None:
                    all_fake_mazes = torch.cat([all_fake_mazes, g_fake_mazes], dim=0)
                else:
                    all_fake_mazes = g_fake_mazes
                dump_file(os.path.join(self.results_path, 'fake_mazes_{}.pickle'.format(epoch + 1)), all_fake_mazes)

            # -- Save models -- #
            generator_checkpoint = Bunch(
                epoch=epoch + 1, state_dict=self.G.state_dict(),
                optimizer=self.G.optimizer.state_dict()
            )
            Checkpoint(generator_checkpoint, self.name, 'generator').save()
            discriminator_checkpoint = Bunch(
                epoch=epoch + 1, state_dict=self.D.state_dict(),
                optimizer=self.D.optimizer.state_dict()
            )
            Checkpoint(discriminator_checkpoint, self.name, 'discriminator').save()


def dump_file(loc, data):
    output = open(loc, 'wb')
    pickle.dump(data, output)
    output.close()


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
