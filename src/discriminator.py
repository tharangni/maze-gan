import torch
import torch.nn as nn
from gumbell import gumbel_softmax
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


class Discriminator:
    def __init__(self,
                 device,
                 hidden_size,
                 maze_size,
                 num_epochs,
                 batch_size):
        self.device = device
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = nn.Sequential(
            nn.Linear(maze_size, hidden_size),
            nn.LeakyReLU(0.5),
            #nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.5),
            #nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)

    def train(self,
              G,
              input_size,
              mazes,
              loss_criterion,
              real_labels,
              fake_labels,
              reset_grad):

        #Loss starts (x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))

        reset_grad()
        #Real Data BCE_Loss
        outputs = self.model(gumbel_softmax(mazes) + torch.randn(mazes.size()))
        d_loss_real = loss_criterion(outputs, real_labels)
        d_loss_real.backward()
        real_score = outputs

        ##Fake Data BCE_Loss
        z = torch.randn(self.batch_size, input_size).to(self.device)
        fake_mazes = G(z)
        outputs = self.model(gumbel_softmax(fake_mazes))

        #Fake data loss
        fake_score = outputs
        d_loss_fake = loss_criterion(outputs, fake_labels)
        d_loss_fake.backward()

        self.optimizer.step()

        return d_loss_fake + d_loss_real, fake_score, real_score, fake_mazes


    def backprop(self, d_loss_fake, d_loss_real, reset_grad):
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        self.optimizer.step()
        return d_loss
