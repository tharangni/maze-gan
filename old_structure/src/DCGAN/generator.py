import torch
import numpy as np
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.autograd import Variable


class Generator(nn.Module):

    def __init__(self,
                 device,
                 input_size,
                 hidden_size,
                 mx,
                 my,
                 num_epochs,
                 batch_size):
        super(Generator, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.maze_size = maze_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.mx = mx
        self.my = my
#        self.model = nn.Sequential(
#            nn.Linear(input_size, hidden_size),
#            nn.ReLU(),
#            nn.Linear(hidden_size, hidden_size),
#            nn.ReLU(),
#            nn.Linear(hidden_size, maze_size),
#            # nn.Tanh())
#            nn.Sigmoid())
        self.init_size = mx // 4
        self.l1 = nn.Sequential(nn.Linear(input_size, 128*self.init_size**2))
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        #set device
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002)

        
    

    def train(self, D, loss_criterion, real_labels, reset_grad):
        reset_grad()

        ## Compute loss with fake mazes
        # forward pass with the discrete variable 
        # z_discrete = np.random.choice([0, 1], size=(self.batch_size, self.input_size), p=[4./10, 6./10])
        # z_tensor = torch.from_numpy(z_discrete).to(self.device)
        # z_tensor = z_tensor.float()
        z = torch.randn(self.batch_size, self.input_size).to(self.device)
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        fake_mazes = self.model(out)
        # gumbel-softmax?
        test_tensor = Variable(torch.tensor([0.75]).to(self.device))
        m = RelaxedBernoulli(test_tensor, probs=fake_mazes)
        fake_mazes = m.sample()
        out = D.model(fake_mazes)
        out = out.view(out.shape[0], -1)
        outputs = D.adv_layer(out)
        

        g_loss = loss_criterion(outputs, real_labels)

        g_loss.backward()
        
     
#        for param in self.parameters():
#            print(param.grad)
#            print(param.grad.data.sum())

        self.optimizer.step()
        
            
        #maximize log(D(G(z))
        return g_loss

    def backprop(self, g_loss, reset_grad):
        reset_grad()
        g_loss.backward()
        self.optimizer.step()
