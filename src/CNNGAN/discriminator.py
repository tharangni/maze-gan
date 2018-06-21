import torch
import torch.nn as nn
from torchvision import models
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


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
        self.model = models.alexnet()
        self.model.classifier[6] = nn.Linear(4096, 1)
        modules = list(self.model.classifier.children())
        modules.append(nn.Sigmoid())
        self.model.classifier = nn.Sequential(*modules)
        #print("D ", self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)
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

        z = torch.randn(self.batch_size, input_size).to(self.device)#input_size).to(self.device)
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

    def backprop(self, d_loss_fake, d_loss_real, reset_grad):
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        self.optimizer.step()
        return d_loss
