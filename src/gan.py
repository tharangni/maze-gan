import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from generator import Generator
from discriminator import Discriminator
from maze_gen import maze_data

# Device configuration
class GAN:

    def __init__(self,
                 device,
                 latent_size,
                 hidden_size,
                 maze_size,
                 num_epochs,
                 batch_size,
                 mx,
                 my,
                 N,
                 maze_dir):
        self.device = device
        self.G = Generator(self.device, latent_size, hidden_size, maze_size, num_epochs, batch_size)
        self.D = Discriminator(self.device, latent_size, hidden_size, maze_size, num_epochs, batch_size)
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.maze_size = maze_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.mx = mx
        self.my = my
        self.N = N
        self.maze_dir = maze_dir


    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def reset_grad(self):
        self.D.optimizer.zero_grad()
        self.G.optimizer.zero_grad()

    def train(self):

        # Image processing
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5),  # 3 for RGB channels
        #                          std=(0.5, 0.5, 0.5))])

        self.transform = transforms.Compose([transforms.ToTensor()])
        #generate maze data
        maze = self.transform(maze_data(self.mx, self.my, self.N))

        #Make this dynamic
        # MNIST dataset
        # mnist = torchvision.datasets.MNIST(root='../../data/',
        #                                    train=True,
        #                                    transform=transform,
        #                                    download=True)

        # Data loader
        data_loader = torch.utils.data.DataLoader(dataset=maze,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)

        #Creates a criterion that measures the Binary Cross Entropy between the target and the output
        loss_criterion = nn.BCELoss()
        total_step = len(data_loader)

        #Start training
        for epoch in range(self.num_epochs):
            for i, mazes in enumerate(data_loader):
                print(mazes)
                # mazes = mazes.reshape(self.batch_size, -1).to(self.device)
                mazes = mazes.view(maze.numel()).to(torch.float)

                # Create the labels which are later used as input for the BCE loss
                real_labels = torch.ones([self.batch_size, 1], dtype = torch.float).to(self.device)
                fake_labels = torch.zeros([self.batch_size, 1], dtype = torch.float).to(self.device)

                #Train Discrimator
                d_loss_fake, d_loss_real, fake_score, real_score, fake_mazes = self.D.train(self.G.model, mazes, loss_criterion, real_labels, fake_labels)
                d_loss = self.D.backprop( d_loss_fake, d_loss_real, self.reset_grad)

                #Train Generator
                g_loss = self.G.train(self.D.model, loss_criterion, real_labels)
                self.G.backprop(g_loss, self.reset_grad)

                if (i + 1) % 200 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                          .format(epoch, self.num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                                  real_score.mean().item(), fake_score.mean().item()))

            # Save real mazes
            if (epoch + 1) == 1:
                mazes = mazes.reshape((mx, my))
                # mazes = mazes.reshape(mazes.size(0), 1, 28, 28)
                save_image(self.denorm(mazes), os.path.join(self.maze_dir, 'real_mazes.png'))

            # Save sampled mazes
            fake_mazes = fake_mazes.reshape(fake_mazes.size(0), 1, 28, 28)
            save_image(self.denorm(fake_mazes), os.path.join(self.maze_dir, 'fake_mazes-{}.png'.format(epoch + 1)))
        torch.save(self.G.state_dict(), 'G.ckpt')
        torch.save(self.D.state_dict(), 'D.ckpt')
