import torch
import torch.nn as nn

from helpers.Checkpoint import Checkpoint


class Generator(nn.Module):

    def __init__(self, opts):
        super(Generator, self).__init__()
        self.name = opts.name

        self.device = opts.device
        self.latent_size = opts.latent_size
        self.hidden_size = opts.hidden_size
        self.maze_size = opts.maze_size
        self.num_epochs = opts.num_epochs
        self.batch_size = opts.batch_size
        self.learning_rate = opts.learning_rate

        self.model = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.maze_size),
            nn.Tanh())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if opts.resume:
            Checkpoint(None, self.name, 'generator').load(self)
        self.model = self.model.to(device=self.device)

    def forward(self, z):
        return self.model(z)

    def backward(self, args):
        loss = args.criterion(args.scores, args.real)

        loss.backward()
        self.optimizer.step()

        return loss
