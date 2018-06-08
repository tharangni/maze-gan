import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from generator import Generator
from discriminator import Discriminator

# Device configuration
class GAN:

    def __init__(self,
                 device,
                 latent_size,
                 hidden_size,
                 image_size,
                 num_epochs,
                 batch_size,
                 img_dir):
        self.device = device
        self.G = Generator(self.device, latent_size, hidden_size, image_size, num_epochs, batch_size)
        self.D = Discriminator(self.device, latent_size, hidden_size, image_size, num_epochs, batch_size)
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.img_dir = img_dir


    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def reset_grad(self):
        self.D.optimizer.zero_grad()
        self.G.optimizer.zero_grad()

    def train(self):

        # Image processing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),  # 3 for RGB channels
                                 std=(0.5, 0.5, 0.5))])

        #Make this dynamic
        # MNIST dataset
        mnist = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transform,
                                           download=True)

        # Data loader
        data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)

        #Creates a criterion that measures the Binary Cross Entropy between the target and the output
        loss_criterion = nn.BCELoss()
        total_step = len(data_loader)

        #Start training
        for epoch in range(self.num_epochs):
            for i, (images, _) in enumerate(data_loader):
                images = images.reshape(self.batch_size, -1).to(self.device)

                # Create the labels which are later used as input for the BCE loss
                real_labels = torch.ones(self.batch_size, 1).to(self.device)
                fake_labels = torch.zeros(self.batch_size, 1).to(self.device)

                #Train Discrimator
                d_loss_fake, d_loss_real, fake_score, real_score, fake_images = self.D.train(self.G.model, images, loss_criterion, real_labels, fake_labels)
                d_loss = self.D.backprop( d_loss_fake, d_loss_real, self.reset_grad)

                #Train Generator
                g_loss = self.G.train(self.D.model, loss_criterion, real_labels)
                self.G.backprop(g_loss, self.reset_grad)

                if (i + 1) % 200 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                          .format(epoch, self.num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                                  real_score.mean().item(), fake_score.mean().item()))

            # Save real images
            if (epoch + 1) == 1:
                images = images.reshape(images.size(0), 1, 28, 28)
                save_image(self.denorm(images), os.path.join(self.img_dir, 'real_images.png'))

            # Save sampled images
            fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
            save_image(self.denorm(fake_images), os.path.join(self.img_dir, 'fake_images-{}.png'.format(epoch + 1)))
        torch.save(self.G.state_dict(), 'G.ckpt')
        torch.save(self.D.state_dict(), 'D.ckpt')