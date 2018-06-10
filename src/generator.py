import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

class Generator():

    def __init__(self,
                 device,
                 input_size,
                 hidden_size,
                 maze_size,
                 num_epochs,
                 batch_size):
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
            #nn.Tanh())
            nn.Sigmoid())
        #set device
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)

    def train(self, D, loss_criterion, real_labels):
        # Compute loss with fake mazes
        z = torch.randn(self.batch_size, self.input_size).to(self.device)
        fake_mazes = self.model(z)
        # gumbel-softmax?
        m = RelaxedBernoulli(torch.tensor([0.75]), probs=fake_mazes)
        fake_mazes = m.sample()
        outputs = D(fake_mazes)

        #maximize log(D(G(z))
        return loss_criterion(outputs, real_labels)

    def backprop(self, g_loss, reset_grad):
        reset_grad()
        g_loss.backward()
        self.optimizer.step()
