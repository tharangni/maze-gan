import torch
import numpy as np
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


class Generator():

    def __init__(self,
                 device,
                 input_size,
                 hidden_size,
                 maze_size,
                 num_epochs,
                 batch_size,
                 writer):
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.maze_size = maze_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, maze_size),
            # nn.Tanh())
            nn.Sigmoid())
        # set device
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)
        self.writer = writer

    def train(self, D, loss_criterion, real_labels, reset_grad):
        reset_grad()

        ## Compute loss with fake mazes
        # forward pass with the discrete variable
        # z_discrete = np.random.choice([0, 1], size=(self.batch_size, self.input_size), p=[4./10, 6./10])
        # z_tensor = torch.from_numpy(z_discrete).to(self.device)
        # z_tensor = z_tensor.float()
        z = torch.randn(self.batch_size, self.input_size).to(self.device)
        fake_mazes = self.model(z)

        # gumbel-softmax?

        outputs = D(fake_mazes)

        g_loss = loss_criterion(outputs, real_labels)

        g_loss.backward()

        self.optimizer.step()
        # maximize log(D(G(z))
        return g_loss

    def backprop(self, g_loss, reset_grad):
        reset_grad()
        g_loss.backward()
        self.optimizer.step()
