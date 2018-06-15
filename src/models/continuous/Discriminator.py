import torch
import torch.nn as nn

from helpers.Checkpoint import Checkpoint


class Discriminator(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.device = opts.device
        self.maze_size = opts.maze_size
        self.hidden_size = opts.hidden_size
        self.num_epochs = opts.num_epochs
        self.batch_size = opts.batch_size
        self.learning_rate = opts.learning_rate

        self.model = nn.Sequential(
            nn.Linear(self.maze_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if opts.resume:
            Checkpoint(None, 'MNIST', 'discriminator').load(self)
        self.model = self.model.to(self.device)

    def forward(self, args):
        real_scores = self.model(args.real_mazes)
        fake_scores = self.model(args.fake_mazes)

        return real_scores, fake_scores

    def backward(self, args):
        real_loss = args.criterion(args.real_scores, args.real)
        fake_loss = args.criterion(args.fake_scores, args.fake)
        loss = real_loss + fake_loss

        loss.backward()
        self.optimizer.step()

        return loss