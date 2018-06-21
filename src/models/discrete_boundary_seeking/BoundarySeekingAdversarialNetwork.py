import os
import pickle
import torch.nn as nn
import torch.utils.data
import torch.random
from torchvision.utils import save_image

from helpers.Checkpoint import Checkpoint
from helpers.MazeGenerator import gen_maze_data
from models.discrete_boundary_seeking.Generator import Generator
from models.discrete_boundary_seeking.Discriminator import Discriminator
from helpers.Bunch import Bunch


class BoundarySeekingAdversarialNetwork:

    def __init__(self, opts):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator_opts = Bunch(
            device=self.device, latent_size=opts.latent_size, hidden_size=opts.hidden_size, maze_size=opts.mx * opts.my,
            num_epochs=opts.num_epochs, batch_size=opts.batch_size, learning_rate=opts.g_lr,
            resume=opts.resume, temp=opts.temp
        )
        self.discriminator_opts = Bunch(
            device=self.device, hidden_size=opts.hidden_size, maze_size=opts.mx * opts.my,
            num_epochs=opts.num_epochs, batch_size=opts.batch_size, learning_rate=opts.d_lr, resume=opts.resume
        )
        self.latent_size = opts.latent_size
        self.hidden_size = opts.hidden_size
        self.num_epochs = opts.num_epochs
        self.batch_size = opts.batch_size
        self.mx = opts.mx
        self.my = opts.my
        self.N = opts.N
        self.results_path = os.path.join('results', 'BoundarySeekingGumbelMazes')
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        self.G = Generator(self.generator_opts)
        self.D = Discriminator(self.discriminator_opts)

        self.discriminator_loss = nn.BCELoss().to(device=self.device)

    def boundary_seeking_loss(self, y_pred):
        """
        Boundary seeking loss.
        Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
        """
        return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2).to(device=self.device, dtype=torch.float32)

    def train(self):
        # --- Maze Data --- #
        print('Generating {} {}x{} maze examples.'.format(self.N, self.mx, self.my))
        maze_data = gen_maze_data(self.N, self.mx, self.my)
        data_loader = maze_data.reshape(-1, self.batch_size, self.mx, self.my)
        # -- Number of batches -- #
        num_batches = len(data_loader)

        # -- Generate labels -- #
        real = torch.ones([self.batch_size, 1], dtype=torch.float).to(self.device)
        fake = torch.zeros([self.batch_size, 1], dtype=torch.float).to(self.device)

        # --- Start training --- #
        for epoch in range(self.num_epochs):
            real_mazes = None
            fake_mazes = None
            for batch_idx, real_mazes in enumerate(data_loader):
                real_mazes = real_mazes.reshape(self.batch_size, -1).to(self.device, dtype=torch.float32)

                # -- Reset gradients -- #
                self.D.optimizer.zero_grad()
                self.G.optimizer.zero_grad()

                # -- Train Discriminator -- #
                z = torch.randn(self.batch_size, self.latent_size).to(self.device)
                fake_mazes = self.G.forward(z)
                d_forward_args = Bunch(
                    real_mazes=real_mazes,
                    fake_mazes=fake_mazes.detach()
                )
                real_scores, fake_scores = self.D.forward(d_forward_args)
                d_backward_args = Bunch(
                    criterion=self.discriminator_loss,
                    real_scores=real_scores,
                    fake_scores=fake_scores,
                    real=real,
                    fake=fake
                )
                d_loss = self.D.backward(d_backward_args)

                # -- Train Generator -- #
                z = torch.randn(self.batch_size, self.latent_size).to(self.device)
                fake_mazes = self.G.forward(z)
                fake_scores = self.D.model(fake_mazes)
                g_backward_args = Bunch(
                    criterion=self.boundary_seeking_loss,
                    scores=fake_scores
                )
                g_loss = self.G.backward(g_backward_args)

                # -- Log events -- #
                if (batch_idx + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                          .format(epoch + 1, self.num_epochs, batch_idx + 1, num_batches, d_loss.item(), g_loss.item(),
                                  real_scores.mean().item(), fake_scores.mean().item()))

            # -- Save real mazes -- #
            if (epoch + 1) == 1:
                real_mazes = real_mazes.reshape(real_mazes.size(0), self.mx, self.my)
                dump_file(os.path.join(self.results_path, 'real_mazes.pickle'), real_mazes)
                real_mazes = real_mazes.reshape(real_mazes.size(0), 1, self.mx, self.my)
                save_image(real_mazes,
                           os.path.join(self.results_path, 'real_mazes.png'),
                           nrow=5, padding=5, pad_value=1)

            # -- Save generated mazes -- #
            if batch_idx + 1 == num_batches:
                fake_mazes = fake_mazes.reshape(fake_mazes.size(0), self.mx, self.my)
                dump_file(os.path.join(self.results_path, 'fake_mazes_{}.pickle'.format(epoch + 1)), fake_mazes)
                fake_mazes = fake_mazes.reshape(fake_mazes.size(0), 1, self.mx, self.my)
                save_image(fake_mazes,
                           os.path.join(self.results_path, 'fake_mazes_{}.png'.format(epoch + 1)),
                           nrow=5, padding=5, pad_value=1)

            # -- Save models -- #
            generator_checkpoint = Bunch(
                epoch=epoch + 1, state_dict=self.G.model.state_dict(),
                optimizer=self.G.optimizer.state_dict()
            )
            Checkpoint(generator_checkpoint, 'BoundarySeekingGumbelMazes', 'generator').save()
            discriminator_checkpoint = Bunch(
                epoch=epoch + 1, state_dict=self.D.model.state_dict(),
                optimizer=self.D.optimizer.state_dict()
            )
            Checkpoint(discriminator_checkpoint, 'BoundarySeekingGumbelMazes', 'discriminator').save()


def dump_file(loc, data):
    output = open(loc, 'wb')
    pickle.dump(data, output)
    output.close()


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
