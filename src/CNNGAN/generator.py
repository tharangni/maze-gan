import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


class Generator():

    def __init__(self,
                 device,
                 input_size,
                 hidden_size,
                 maze_size,
                 num_epochs,
                 batch_size,
                 writer,
                 output_size):
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.maze_size = maze_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = G(10, output_size, 244)#self.input_size)
        # set device
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)
        self.writer = writer
        print("G ", self.model)

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
        test_tensor = torch.tensor([0.75]).to(self.device)
        m = RelaxedBernoulli(test_tensor, probs=fake_mazes)
        fake_mazes = m.sample()
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

from utils import initialize_weights
class G(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(G, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x