import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import torch.nn.functional as F

class Generator():

    def __init__(self,
                 device,
                 input_size,
                 hidden_size,
                 maze_size,
                 num_epochs,
                 batch_size,
                 writer,
                 output_dim):
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.maze_size = maze_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        #input dim
        self.model = G(batch_size, input_size)#input_size, output_dim, input_size)
        # set device
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        self.writer = writer
        #print("G ", self.model)

    def train(self, D, loss_criterion, real_labels, reset_grad):
        reset_grad()

        ## Compute loss with fake mazes
        # forward pass with the discrete variable 
        # z_discrete = np.random.choice([0, 1], size=(self.batch_size, self.input_size), p=[4./10, 6./10])
        # z_tensor = torch.from_numpy(z_discrete).to(self.device)
        # z_tensor = z_tensor.float()
        z = torch.randn((self.batch_size, 100)).view(-1, 100, 1, 1)
        #z = torch.randn(self.batch_size, self.input_size).to(self.device)
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

class G(nn.Module):
    # initializers
    def __init__(self, d=128, input_size=10):
        super(G, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        #self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        #print("G time")
        #print(input.shape)
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        #print(x.shape)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        #print(x.shape)
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        #print(x.shape)
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        #print(x.shape)
        #x = F.tanh(self.deconv5(x))
        x = F.sigmoid(self.deconv5(x))
        #print(x.shape)
        #print("======================")
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()