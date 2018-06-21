import torch
import torch.nn as nn
from torch.autograd import Variable

from helpers.Checkpoint import Checkpoint
from helpers import GumbelSoftmaxTrick
from helpers import HeavisideTrick


class Generator(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.device = opts.device
        self.latent_size = opts.latent_size
        self.hidden_size = opts.hidden_size
        self.maze_size = opts.maze_size
        self.num_epochs = opts.num_epochs
        self.batch_size = opts.batch_size
        self.learning_rate = opts.learning_rate
        self.temperature = opts.temp

        self.model = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.maze_size),
            nn.Sigmoid()
            # nn.Tanh()
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if opts.resume:
            Checkpoint(None, 'BoundarySeekingGumbelMazes', 'generator').load(self)
        self.model = self.model.to(device=self.device)

    def forward(self, z):
        mazes = self.model(z)

        # -- GumbelSoftmaxTrick -- #
        maze_class_prob = Variable(torch.stack((1 - mazes, mazes), dim=2))
        maze_class_logits = Variable(torch.log(maze_class_prob))
        return GumbelSoftmaxTrick.pass_through(maze_class_logits, self.temperature)

        # -- HeavisidePassThroughTrick -- #
        # return HeavisideTrick.pass_through(mazes)

    def backward(self, args):
        loss = args.criterion(args.scores)
        loss.backward()
        self.optimizer.step()

        return loss
