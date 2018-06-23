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
        self.model = Dnet(batch_size) #too big??
        #self.model.classifier[6] = nn.Linear(4096, 1)
        #modules = list(self.model.classifier.children())
        #modules.append(nn.Sigmoid())
        #self.model.classifier = nn.Sequential(*modules)
        #print("D ", self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00002)
        self.writer = writer
        #input = (torch.rand([batch_size - 1, maze_size]))
        #writer.add_graph(model=self.model, input_to_model=(input,), verbose=True)
        self.model = self.model.to(self.device)

    """def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out"""

    def train(self,
              G,
              input_size,
              mazes,
              loss_criterion,
              real_labels,
              fake_labels,
              reset_grad):
        # Loss starts (x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))

        reset_grad()
        #print("D inputs ", mazes.size())
        outputs = self.model(mazes)
        #print("D ouptuts ", outputs.size())
        d_loss_real = loss_criterion(outputs, real_labels)
        d_loss_real.backward()
        real_score = outputs

        ##Fake Data BCE_Loss
        # Generate fake data first
        z = torch.randn((self.batch_size, 100)).view(-1, 100, 1, 1)
        #z = torch.randn(self.batch_size, input_size).to(self.device)#input_size).to(self.device)
        #z = torch.randn(mazes[0].size()).to(self.device)
        #print("G input ", z.size())
        fake_mazes = G(z)
        #print("Fake mazes")
        #print("G outputs ", fake_mazes.size())
        outputs = self.model(fake_mazes)
        # Fake data loss
        fake_score = outputs
        d_loss_fake = loss_criterion(outputs, fake_labels)
        d_loss_fake.backward()

        self.optimizer.step()

        return d_loss_fake + d_loss_real, fake_score, real_score, fake_mazes

class Dnet(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Dnet, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d * 8, d, 4, 1, 0)
        self.fc = nn.Linear(d, 1)
        self.d = d
        #self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        #print("!!!!!!!!!!!")
        #print(input.shape)
        x = F.leaky_relu(self.conv1(input), 0.2)
        #print(x.shape)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        #print(x.shape)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        #print(x.shape)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        #print(x.shape)
        #x = F.max_pool2d(x,kernel_size=3, stride=2)
        #print("max ", x.shape)
        x = F.leaky_relu(self.conv5(x))
        #print("final cov ", x.shape)
        x = x.view(x.size(0), self.d)#make this dynamic
        #print(x.shape)
        x = F.sigmoid(self.fc(x))
        #print(x.shape)
        #print(x)
        #x = x.view(x.size(0), 256 * 6 * 6)
        #x = self.classifier(x)
        #print("------------------------")
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()