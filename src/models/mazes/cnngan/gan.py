import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter
from PIL import Image
from torchvision import datasets
from CNNGAN.generator import Generator
from CNNGAN.discriminator import Discriminator
from maze_gen import gen_maze_data
from utils import dump_file


# Device configuration
class GAN:
    def __init__(self,
                 device,
                 args):
        self.device = device
        self.writer = SummaryWriter()
        self.maze_dir = args.maze_dir
        self.model_dir = "models"
        self.path = "CNNGAN/"
        self.training_dir = args.td
        self.hidden_size = args.hidden_size
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.mx = args.mx
        self.my = args.my
        self.N = args.N
        self.img_size = 28
        #self.set_up_data()
        # self.data_loader = self.load_data()
        self.G = Generator(self.device, args.input_size, args.hidden_size, args.mx * args.my, args.num_epochs,
                           args.batch_size, self.writer,
                           5)
        self.D = Discriminator(self.device, args.hidden_size, args.mx * args.my, args.num_epochs, args.batch_size,
                               self.writer)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def reset_grad(self):
        self.D.optimizer.zero_grad()
        self.G.optimizer.zero_grad()

    def set_up_data(self):
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)

        if len([name for name in os.listdir(self.training_dir) if os.path.isfile(name)]) < 1:
            self.create_training_data()

    def train(self):

        data_loader = self.load_data()
        self.G.model.weight_init(mean=0, std=0.5)
        self.D.model.weight_init(mean=0, std=0.5)

        # Creates a criterion that measures the Binary Cross Entropy between the target and the output
        loss_criterion = nn.BCELoss()
        total_step = len(data_loader)

        # Start training
        # epochs_file = csv.writer(open(os.path.join(self.path, self.model_dir, "epoch.csv"), 'w', newline=''), delimiter=',')
        # epochs_file.writerow(['epoch_no', 'batch_no', 'd_loss', 'g_loss', 'D(x)', 'D(G(X))'])

        previous_d_G_z = 1000000000000000000
        previous_d_z = -1000000000000000000
        threshold = 0.3  # hyperparmeter # old treshold
        threshold2 = 0.7
        epoch_range = 0

        for epoch in range(self.num_epochs):
            for local_batch, maze_set in enumerate(data_loader):
                maze_set = torch.from_numpy(np.array(maze_set))
                #print("maze set ", maze_set.shape)
                real_labels = torch.ones(self.batch_size, 1)
                fake_labels = torch.zeros(self.batch_size, 1)

                # Train Discrimator
                d_loss, fake_score, real_score, fake_mazes = self.D.train(self.G.model,
                                                                          self.G.input_size,
                                                                          maze_set,
                                                                          loss_criterion,
                                                                          real_labels,
                                                                          fake_labels,
                                                                          self.reset_grad,
                                                                          previous_d_G_z,
                                                                          previous_d_z,
                                                                          threshold,
                                                                          threshold2,
                                                                          epoch_range > epoch)

                # Train Generator
                g_loss = self.G.train(self.D.model, loss_criterion, real_labels, self.reset_grad)

                self.writer.add_scalars('VGAN/epoch', {'g_loss': g_loss,
                                                       'd_loss': d_loss,
                                                       'D(x)': real_score.mean().item(),
                                                       'D(G(z))': fake_score.mean().item(),
                                                       }, epoch + 1)

                # Write to results for plotting
                # epochs_file.writerow(
                #    [epoch + 1, local_batch + 1, d_loss.item(), g_loss.item(), real_score.mean().item(),
                #     fake_score.mean().item()])

                if (local_batch + 1) % 100 == 0 or (epoch + 1) % 100 == 0:
                    for name, param in self.G.model.named_parameters():
                        self.writer.add_histogram("CNNGAN/Generator/" + name, param.clone().cpu().data.numpy(),
                                                  epoch + 1)

                    for name, param in self.D.model.named_parameters():
                        self.writer.add_histogram("CNNGAN/Discriminator/" + name, param.clone().cpu().data.numpy(),
                                                  epoch + 1)

                if (previous_d_G_z > threshold and previous_d_z < threshold2) or epoch_range > epoch:
                    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                          .format(epoch + 1, self.num_epochs, local_batch + 1, total_step, d_loss.item(), g_loss.item(),
                                  real_score.mean().item(), fake_score.mean().item()))
                else:
                    print(
                        'Epoch [{}/{}], Step [{}/{}], Stalled d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                            .format(epoch + 1, self.num_epochs, local_batch + 1, total_step, d_loss, g_loss.item(),
                                    real_score.mean().item(), fake_score.mean().item()))
                #previous_d_G_z = fake_score.mean().item()
                #previous_d_z =  real_score.mean().item()

                # Save real mazes
            if (epoch + 1) == 1:
                maze_set = maze_set.reshape(self.batch_size, 64, 64)
                dump_file(os.path.join(self.maze_dir, 'real_mazes.pickle'), maze_set)

            # Save sampled mazes
            fake_mazes = fake_mazes.reshape(self.batch_size, 64, 64)
            dump_file(os.path.join(self.maze_dir, 'fake_mazes-{}.pickle'.format(epoch + 1)), fake_mazes)

        if not os.path.exists(self.path + self.model_dir):
            os.makedirs(self.path + self.model_dir)
        torch.save(self.G.model.state_dict(), self.path + self.model_dir + '/G.ckpt')
        torch.save(self.D.model.state_dict(), self.path + self.model_dir + '/D.ckpt')

        self.writer.export_scalars_to_json("./tensorboard_data.json")  # use this istead of pickle??
        self.writer.close()

    def create_training_data(self):
        print("Creating data here ", self.training_dir)
        gen_maze_data(self.N, self.mx, self.my, save_to_file=True, dir=self.training_dir, randomise=True)

    def load_data(self):
        transform = transforms.Compose([
            transforms.Scale(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        data = datasets.MNIST("data", train=True, download=True,
                              transform=transform)
        mnist_loader = torch.zeros_like(data.train_data).type(torch.FloatTensor)
        for idx in range(len(data)):
            mnist_loader[idx], _ = data[idx]

        return mnist_loader.reshape(-1, self.batch_size, 1, self.img_size, self.img_size).type(torch.FloatTensor)

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transform,),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        # train_loader = torch.utils.data.DataLoader(
        #     ImageFilelist(
        #         root=self.training_dir,
        #         flist=os.listdir(self.training_dir),
        #         transform=transform),
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=4,
        #     pin_memory=True)
        return train_loader


# For data loading
class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform, target_transform=None):
        self.root = root
        self.imlist = flist  # flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = cv2.imread(os.path.join(self.root, impath), 0)  # self.loader(os.path.join(self.root, impath))
        img = img[:, :, None]
        img = transforms.ToPILImage()(img)
        img = self.transform(img)
        return img + torch.autograd.Variable(torch.randn(img.size()) * 0.5)  # add noise

    def __len__(self):
        return len(self.imlist)
