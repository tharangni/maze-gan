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

        self.transform = transforms.Compose([transforms.ToTensor()])
        #generate maze data
        maze = self.transform(maze_data(self.N, self.mx, self.my))

        # Data loader
        print("-------------")
        print("batch size ", self.batch_size)
        data_loader = torch.utils.data.DataLoader(dataset=maze,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)

        #Creates a criterion that measures the Binary Cross Entropy between the target and the output
        loss_criterion = nn.BCELoss()
        total_step = len(data_loader)

        #Start training
        for epoch in range(self.num_epochs):
            print("enumerate ", enumerate(data_loader))
            for i , maze_set in enumerate(data_loader):
                #check here why it doesn't use batch
                print("maze_set ", len(maze_set))
                #for maze in maze_set:
                maze_set= maze_set.reshape(self.batch_size, -1).to(self.device).float()
                #print("i ", i)P
                #print("mazes ", maze)
                print("------")
                #maze = maze.view(maze.numel()).to(torch.float)

                print("Maze size ", maze_set.size())

                # Create the labels which are later used as input for the BCE loss
                real_labels = torch.ones([self.batch_size,1], dtype = torch.float).to(self.device)
                fake_labels = torch.zeros([self.batch_size,1], dtype = torch.float).to(self.device)
                print("Real_labels ", real_labels.size())
                print("Fake_labels ", fake_labels.size())

                print("Type of maze_set ", maze_set.dtype)
                print("Type of real labels ", real_labels.dtype)

                #Train Discrimator
                d_loss_fake, d_loss_real, fake_score, real_score, fake_mazes = self.D.train(self.G.model, maze_set, loss_criterion, real_labels, fake_labels)
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
                #reconstruct mazes here
                maze_set = maze_set.reshape((self.mx, self.my))
                # mazes = mazes.reshape(mazes.size(0), 1, 28, 28)
                save_image(self.denorm(maze_set), os.path.join(self.maze_dir, 'real_mazes.txt'))

            # Save sampled mazes
            fake_mazes = fake_mazes.reshape(fake_mazes.size(0), 1, 28, 28)
            save_image(self.denorm(fake_mazes), os.path.join(self.maze_dir, 'fake_mazes-{}.png'.format(epoch + 1)))
        torch.save(self.G.state_dict(), 'G.ckpt')
        torch.save(self.D.state_dict(), 'D.ckpt')
