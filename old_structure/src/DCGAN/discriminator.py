import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

class Discriminator(nn.Module):
    def __init__(self,
                 device,
                 hidden_size,
                 mx,
                 my,
                 num_epochs,
                 batch_size):
        super(Discriminator, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.mx = mx
        self.my = my
#        self.model = nn.Sequential(
#            nn.Linear(maze_size, hidden_size),
#            nn.LeakyReLU(0.2),
#            nn.Dropout(0.3),
#            nn.Linear(hidden_size, hidden_size),
#            nn.LeakyReLU(0.2),
#            nn.Dropout(0.3),
#            nn.Linear(hidden_size, 1),
#            nn.Sigmoid())
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = mx // 2**4
        self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, 1),
                                        nn.Sigmoid())

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002)

    def train(self,
              G,
              input_size,
              mazes,
              loss_criterion,
              real_labels,
              fake_labels,
              reset_grad):

        #Loss starts (x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))

        reset_grad()
        squeeze_mazes = mazes.unsqueeze_(1)
        out = self.model(squeeze_mazes)
        out = out.view(out.shape[0], -1)
        outputs = self.adv_layer(out)
        
        #Real Data BCE_Loss

        d_loss_real = loss_criterion(outputs, real_labels)
        d_loss_real.backward()
        real_score = outputs

        ##Fake Data BCE_Loss
        # Generate fake data first
    
        z = torch.randn(self.batch_size, input_size).to(self.device)
        z_out = G.l1(z)
        init_size = self.mx // 4
        z_out = z_out.view(z_out.shape[0], 128, init_size, init_size)
        fake_mazes = G.model(z_out)
        test_tensor = torch.tensor([0.75]).to(self.device)
        m = RelaxedBernoulli(test_tensor, probs=fake_mazes)
        fake_mazes = m.sample()
        
        out = self.model(fake_mazes)
        out = out.view(out.shape[0], -1)
        outputs = self.adv_layer(out)
        #Fake data loss
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
