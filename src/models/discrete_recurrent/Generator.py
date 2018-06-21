import torch
import torch.nn as nn

from helpers.Bunch import Bunch
from helpers.Checkpoint import Checkpoint
from helpers import GumbelSoftmaxTrick
from helpers import HeavisideTrick


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
        self.temperature = opts.temp

        self.lstm = nn.LSTM(1, self.hidden_size, bidirectional=True)
        self.h2o = nn.Linear(self.hidden_size * 2, 1)
        self.o2tanh = nn.Tanh()

        self.hidden_state = self.init_hidden_state()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if opts.resume:
            Checkpoint(None, self.name, 'generator').load(self)

    def init_hidden_state(self):
        # Before we've done anything, we don't have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, self.batch_size, self.hidden_size),
                torch.zeros(2, self.batch_size, self.hidden_size))

    def model(self, input_batch):
        self.hidden_state = self.init_hidden_state()

        hidden_states, self.hidden_state = self.lstm(
            input_batch.view(self.maze_size, self.batch_size, 1),
            self.hidden_state
        )

        output = self.h2o(hidden_states.view(self.batch_size, self.maze_size, -1))

        return self.o2tanh(output).view(self.batch_size, self.maze_size)

    def forward(self, z):
        mazes = self.model(z)

        # -- GumbelSoftmaxTrick -- #
        # maze_class_prob = torch.stack((1 - mazes, mazes), dim=2)
        # maze_class_logits = torch.log(maze_class_prob)
        # return GumbelSoftmaxTrick.pass_through(maze_class_logits, self.temperature)

        # -- HeavisidePassThroughTrick -- #
        return HeavisideTrick.pass_through(mazes)

    def backward(self, args):
        loss = args.criterion(args.scores, args.real)
        loss.backward()
        self.optimizer.step()

        return loss
