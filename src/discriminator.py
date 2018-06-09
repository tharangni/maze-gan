import torch
import torch.nn as nn

class Discriminator:
    def __init__(self,
                 device,
                 latent_size,
                 hidden_size,
                 maze_size,
                 num_epochs,
                 batch_size):
        self.device = device
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.maze_size = maze_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = nn.Sequential(
            nn.Linear(maze_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())
        self.model = self.model # this pointless
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)

    def train(self,
              G,
              mazes,
              loss_criterion,
              real_labels,
              fake_labels):

        #Loss starts (x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        #Real BCE_Loss
        outputs = self.model(mazes)
        d_loss_real = loss_criterion(outputs, real_labels)
        real_score = outputs

        #Fake BCE_Loss
        z = torch.randn(self.batch_size, self.latent_size).to(self.device)
        fake_mazes = G(z)
        outputs = self.model(fake_mazes)
        fake_score = outputs
        d_loss_fake = loss_criterion(outputs, fake_labels)

        return d_loss_fake, d_loss_real, fake_score, real_score, fake_mazes


    def backprop(self, d_loss_fake, d_loss_real, reset_grad):
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        self.optimizer.step()
        return d_loss
