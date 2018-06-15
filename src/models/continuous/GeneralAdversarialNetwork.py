import os
import pickle
import torch.nn as nn
import torch.utils.data
import torch.random
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt

from helpers.Checkpoint import Checkpoint
from models.continuous.Generator import Generator
from models.continuous.Discriminator import Discriminator
from helpers.Bunch import Bunch


class GeneralAdversarialNetwork:

    def __init__(self, opts):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator_opts = Bunch(
            device=self.device, latent_size=opts.latent_size, hidden_size=opts.hidden_size, maze_size=opts.mx * opts.my,
            num_epochs=opts.num_epochs, batch_size=opts.batch_size, learning_rate=opts.g_lr, resume=opts.resume
        )
        self.discriminator_opts = Bunch(
            device=self.device, hidden_size=opts.hidden_size, maze_size=opts.mx * opts.my, num_epochs=opts.num_epochs,
            batch_size=opts.batch_size, learning_rate=opts.d_lr, resume=opts.resume
        )
        self.latent_size = opts.latent_size
        self.hidden_size = opts.hidden_size
        self.num_epochs = opts.num_epochs
        self.batch_size = opts.batch_size
        self.mx = opts.mx
        self.my = opts.my
        self.N = opts.N
        self.results_path = os.path.join('results', 'MNIST')
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        self.G = Generator(self.generator_opts)
        self.D = Discriminator(self.discriminator_opts)

        self.loss = nn.BCELoss().to(device=self.device)

    def train(self):

        # --- MNIST Data --- #
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                  std=(0.5, 0.5, 0.5))])
        mnist = torchvision.datasets.MNIST(root='../../data/', train=True, transform=transform, download=True)
        # data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=self.batch_size,
        #                                           shuffle=False, num_workers=16, pin_memory=True)
        data_loader = torch.zeros_like(mnist.train_data).to(device=self.device, dtype=torch.float32)
        for idx, (image, _) in enumerate(mnist):
            data_loader[idx] = image
        data_loader = data_loader.reshape(-1, self.batch_size, 28, 28)

        # -- Number of batches -- #
        num_batches = len(data_loader)

        # -- Generate labels -- #
        real = torch.ones([self.batch_size, 1]).to(device=self.device)
        fake = torch.zeros([self.batch_size, 1]).to(device=self.device)

        # --- Start training --- #
        for epoch in range(self.num_epochs):
            real_images = None
            fake_images = None
            # for batch_idx, (real_images, _) in enumerate(data_loader):
            for batch_idx, real_images in enumerate(data_loader):
                real_images = real_images.reshape(self.batch_size, -1).to(device=self.device, dtype=torch.float32)

                # -- Reset gradients -- #
                self.D.optimizer.zero_grad()
                self.G.optimizer.zero_grad()

                # -- Train Discriminator -- #
                z = torch.randn(self.batch_size, self.latent_size).to(device=self.device)
                fake_images = self.G.forward(z)
                d_forward_args = Bunch(
                    real_mazes=real_images,
                    fake_mazes=fake_images.detach()
                )
                real_scores, fake_scores = self.D.forward(d_forward_args)
                d_backward_args = Bunch(
                    criterion=self.loss,
                    real_scores=real_scores,
                    fake_scores=fake_scores,
                    real=real,
                    fake=fake
                )
                d_loss = self.D.backward(d_backward_args)

                # -- Train Generator -- #
                z = torch.randn(self.batch_size, self.latent_size).to(device=self.device)
                fake_images = self.G.forward(z)
                fake_scores = self.D.model(fake_images)
                g_backward_args = Bunch(
                    criterion=self.loss,
                    scores=fake_scores,
                    real=real
                )
                g_loss = self.G.backward(g_backward_args)

                # -- Log events -- #
                if (batch_idx + 1) % 100 == 0:
                    plt.figure()
                    plt.imshow(denorm(real_images[0].view(28, 28)), cmap='gray')
                    plt.show()
                    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                          .format(epoch + 1, self.num_epochs, batch_idx + 1, num_batches, d_loss.item(), g_loss.item(),
                                  real_scores.mean().item(), fake_scores.mean().item()))

            # -- Save real images -- #
            if (epoch + 1) == 1:
                images = real_images.reshape(fake_images.size(0), 1, 28, 28)
                save_image(denorm(images), os.path.join(self.results_path, 'real_images.png'))

            # -- Save generated images -- #
            fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
            save_image(denorm(fake_images),
                       os.path.join(self.results_path, 'fake_images-{}.png'.format(epoch + 1)))

            # -- Save models -- #
            generator_checkpoint = Bunch(
                epoch=epoch + 1, state_dict=self.G.model.state_dict(),
                optimizer=self.G.optimizer.state_dict()
            )
            Checkpoint(generator_checkpoint, 'MNIST', 'generator').save()
            discriminator_checkpoint = Bunch(
                epoch=epoch + 1, state_dict=self.D.model.state_dict(),
                optimizer=self.D.optimizer.state_dict()
            )
            Checkpoint(discriminator_checkpoint, 'MNIST', 'discriminator').save()


def dump_file(loc, data):
    output = open(loc, 'wb')
    pickle.dump(data, output)
    output.close()


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
