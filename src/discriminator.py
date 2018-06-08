import torch
import torch.nn as nn

class Discriminator:
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
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)

    def train(self,
              G,
              images,
              loss_criterion,
              real_labels,
              fake_labels):

        #Loss starts (x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        #Real BCE_Loss
        outputs = self.model(images)
        d_loss_real = loss_criterion(outputs, real_labels)
        real_score = outputs

        #Fake BCE_Loss
        z = torch.randn(self.batch_size, self.latent_size).to(self.device)
        fake_images = G(z)
        outputs = self.model(fake_images)
        fake_score = outputs
        d_loss_fake = loss_criterion(outputs, fake_labels)

        return d_loss_fake, d_loss_real, fake_score, real_score, fake_images


    def backprop(self, d_loss_fake, d_loss_real, reset_grad):
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        self.optimizer.step()
        return d_loss