import torch
import torch.nn as nn

class Generator():

    def __init__(self,
                 device,
                 latent_size,
                 hidden_size,
                 image_size,
                 num_epochs,
                 batch_size):
        self.device = device
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh())
        #set device
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)

    def train(self, D, loss_criterion, real_labels):
        # Compute loss with fake images
        z = torch.randn(self.batch_size, self.latent_size).to(self.device)
        fake_images = self.model(z)
        outputs = D(fake_images)

        #maximize log(D(G(z))
        return loss_criterion(outputs, real_labels)

    def backprop(self, g_loss, reset_grad):
        reset_grad()
        g_loss.backward()
        self.optimizer.step()