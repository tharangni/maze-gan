import os
import pickle
import torch.nn as nn
import torch.utils.data
import torch.random
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from network.Continuous.Generator import Generator
from network.Continuous.Discriminator import Discriminator
from helpers.Bunch import Bunch


class GeneralAdversarialNetwork:

    def __init__(self, opts):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator_opts = Bunch(
            device=self.device,
            latent_size=opts.latent_size,
            hidden_size=opts.hidden_size,
            maze_size=opts.mx * opts.my,
            num_epochs=opts.num_epochs,
            batch_size=opts.batch_size,
            learning_rate=opts.g_lr,
            resume=opts.resume
        )
        self.discriminator_opts = Bunch(
            device=self.device,
            hidden_size=opts.hidden_size,
            maze_size=opts.mx * opts.my,
            num_epochs=opts.num_epochs,
            batch_size=opts.batch_size,
            learning_rate=opts.d_lr,
            resume=opts.resume
        )
        self.latent_size = opts.latent_size
        self.hidden_size = opts.hidden_size
        self.num_epochs = opts.num_epochs
        self.batch_size = opts.batch_size
        self.mx = opts.mx
        self.my = opts.my
        self.N = opts.N
        self.maze_dir = opts.maze_dir
        self.model_dir = "models"

        self.G = Generator(self.generator_opts)
        self.D = Discriminator(self.discriminator_opts)

        self.loss = nn.BCELoss()


    def train(self):
        # --- Maze Data --- #
        # print('Generating {} {}x{} maze examples.'.format(self.N, self.mx, self.my))
        # maze_data = gen_maze_data(self.N, self.mx, self.my)
        # data_loader = torch.utils.data.DataLoader(dataset=maze_data, batch_size=self.batch_size, shuffle=True)

        # --- MNIST Data --- #
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                  std=(0.5, 0.5, 0.5))])
        mnist = torchvision.datasets.MNIST(root='../../data/', train=True, transform=transform, download=True)
        data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=self.batch_size, shuffle=True)

        # -- Number of batches -- #
        num_batches = len(data_loader)

        # -- Generate labels -- #
        real = torch.ones([self.batch_size, 1], dtype=torch.float).to(self.device)
        fake = torch.zeros([self.batch_size, 1], dtype=torch.float).to(self.device)

        # --- Start training --- #
        for epoch in range(self.num_epochs):
            real_mazes = None
            fake_mazes = None
            for batch_idx, (real_mazes, _) in enumerate(data_loader):
                real_mazes = real_mazes.reshape(self.batch_size, -1).to(self.device)

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
                    criterion=self.loss,
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
                    criterion=self.loss,
                    scores=fake_scores,
                    real=real
                )
                g_loss = self.G.backward(g_backward_args)

                # -- Log events -- #
                if (batch_idx + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                          .format(epoch + 1, self.num_epochs, batch_idx + 1, num_batches, d_loss.item(), g_loss.item(),
                                  real_scores.mean().item(), fake_scores.mean().item()))

            # Save real mazes
            # if (epoch + 1) == 1:
            #     real_mazes = real_mazes.reshape(real_mazes.size(0), self.mx, self.my)
            #     dump_file(os.path.join(self.maze_dir, 'real_mazes.pickle'), real_mazes)

            # Save real images
            if (epoch + 1) == 1:
                images = real_mazes.reshape(fake_mazes.size(0), 1, 28, 28)
                save_image(denorm(images), os.path.join(self.maze_dir, 'real_images.png'))

            # Save generated images
            fake_mazes = fake_mazes.reshape(fake_mazes.size(0), 1, 28, 28)
            save_image(denorm(fake_mazes),
                       os.path.join(self.maze_dir, 'fake_images-{}.png'.format(epoch + 1)))

            # Save generated mazes
            # if (epoch + 1) == self.num_epochs:
            #     fake_mazes = fake_mazes.reshape(fake_mazes.size(0), self.mx, self.my)
            #     dump_file(os.path.join(self.maze_dir, 'fake_mazes.pickle'), fake_mazes)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.G.model.state_dict(), self.model_dir + '/Generator.ckpt')
        torch.save(self.D.model.state_dict(), self.model_dir + '/Discriminator.ckpt')


def dump_file(loc, data):
    output = open(loc, 'wb')
    pickle.dump(data, output)
    output.close()


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
