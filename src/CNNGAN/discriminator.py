import torch
import torch.nn as nn
from torchvision import models
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self,
                 device,
                 hidden_size,
                 maze_size,
                 num_epochs,
                 batch_size,
                 writer):
        super(Discriminator, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = D(batch_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00004)
        self.writer = writer
        self.model = self.model.to(self.device)

    def train(self,
              G,
              input_size,
              mazes,
              loss_criterion,
              real_labels,
              fake_labels,
              reset_grad,
              previous_d_G_z,
              threshold):
        # Loss starts (x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))

        reset_grad()
        outputs = self.model(mazes)

        if previous_d_G_z > threshold:
            d_loss_real = loss_criterion(outputs, real_labels)
            d_loss_real.backward()
        else:
            d_loss_real = 1

        real_score = outputs

        ##Fake Data BCE_Loss
        # Generate fake data first
        z = torch.randn((self.batch_size, 100)).view(-1, 100, 1, 1)
        fake_mazes = G(z)
        outputs = self.model(fake_mazes)

        # Fake data loss
        fake_score = outputs
        if previous_d_G_z > threshold:
            d_loss_fake = loss_criterion(outputs, fake_labels)
            d_loss_fake.backward()
        else:
            d_loss_fake = 1

        self.optimizer.step()

        return d_loss_fake + d_loss_real, fake_score, real_score, fake_mazes


class D(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d, 4, 1, 0)
        self.fc = nn.Linear(d, 1)
        self.d = d

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        # x = F.max_pool2d(x,kernel_size=3, stride=2)
        x = F.leaky_relu(self.conv5(x))
        x = x.view(x.size(0), self.d)  # make this dynamic
        x = F.sigmoid(self.fc(x))
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
