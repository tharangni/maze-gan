import torch
import torch.nn as nn

from helpers.Checkpoint import Checkpoint


class Discriminator(nn.Module):
    def __init__(self, opts):
        super(Discriminator, self).__init__()
        self.name = opts.name
        self.writer = opts.writer

        self.device = opts.device
        self.maze_size = opts.maze_size
        self.hidden_size = opts.hidden_size
        self.num_epochs = opts.num_epochs
        self.batch_size = opts.batch_size
        self.learning_rate = opts.learning_rate

        self.lstm = nn.LSTM(1, self.hidden_size, bidirectional=True).to(device=self.device, dtype=torch.float32)
        self.h2i = nn.Linear(self.hidden_size * 2, 1).to(device=self.device, dtype=torch.float32)
        self.i2o = nn.Linear(self.maze_size, 1).to(device=self.device, dtype=torch.float32)
        self.o2sig = nn.Sigmoid().to(device=self.device, dtype=torch.float32)

        self.hidden_state = self.init_hidden_state()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if opts.resume:
            Checkpoint(None, self.name, 'discriminator').load(self)

    def init_hidden_state(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, self.batch_size, self.hidden_size).to(device=self.device, dtype=torch.float32),
                torch.zeros(2, self.batch_size, self.hidden_size).to(device=self.device, dtype=torch.float32))

    def model(self, input_batch):
        self.hidden_state = self.init_hidden_state()

        hidden_states, self.hidden_state = self.lstm(
            input_batch.view(self.maze_size, self.batch_size, 1),
            self.hidden_state
        )
        inter = self.h2i(hidden_states.view(self.batch_size, self.maze_size, -1))
        output = self.i2o(inter.view(self.batch_size, self.maze_size))

        return self.o2sig(output)

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
