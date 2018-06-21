import os
import pandas as pd
import torch
import csv
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
from PIL import Image
from CNNGAN.generator import Generator
from CNNGAN.discriminator import Discriminator
from maze_gen import gen_maze_data
from utils import dump_file


# Device configuration
class GAN:

    def __init__(self,
                 device,
                 input_size,
                 hidden_size,
                 num_epochs,
                 batch_size,
                 mx,
                 my,
                 N,
                 maze_dir):
        self.device = device
        self.writer = SummaryWriter()
        # self.set_up_data()
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.mx = mx
        self.my = my
        self.N = N
        self.maze_dir = maze_dir
        self.model_dir = "models"
        self.path = "CNNGAN/"
        self.training_dir = "training_data/"
        self.data_loader = self.load_data()
        out_dim = self.data_loader.__iter__().__next__()[0]
        #print("=======================")
        #print(out_dim.size())
        #print(out_dim.shape)
        #print(out_dim.shape[1])
        self.G = Generator(self.device, input_size, hidden_size, mx * my, num_epochs, batch_size, self.writer, out_dim.shape[0])
        self.D = Discriminator(self.device, hidden_size, mx * my, num_epochs, batch_size, self.writer)

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

        data_loader = self.data_loader

        # Creates a criterion that measures the Binary Cross Entropy between the target and the output
        loss_criterion = nn.BCELoss()
        total_step = len(data_loader)

        # Start training
        # epochs_file = csv.writer(open(os.path.join(self.path, self.model_dir, "epoch.csv"), 'w', newline=''), delimiter=',')
        # epochs_file.writerow(['epoch_no', 'batch_no', 'd_loss', 'g_loss', 'D(x)', 'D(G(X))'])

        for epoch in range(self.num_epochs):
            for local_batch, maze_set in enumerate(data_loader):
                #print("Mazxe", maze_set.shape)
                # maze_set = maze_set.reshape(self.batch_size, -1).to(self.device).float()
                # l + torch.randn(1, 10)*(r-l) - USING SOFT LABELS INSTEAD OF HARD
                # Real: 0.0 - 0.1
                # Fake: 0.9 - 1.0
                # adding 10% noise to training (i.e. add 10% fake labels to real and vice versa)
                real_labels = 0 + torch.rand([self.batch_size, 1], dtype=torch.float).to(self.device) * (0.1 - 0.0)
                fake_labels = 0.9 + torch.rand([self.batch_size, 1], dtype=torch.float).to(self.device) * (1.0 - 0.9)

                noise_samples = 20

                if (epoch % noise_samples == 0):
                    real_labels = 0.9 + torch.randn([self.batch_size, 1], dtype=torch.float).to(self.device) * (
                            1.0 - 0.9)
                    fake_labels = 0 + torch.randn([self.batch_size, 1], dtype=torch.float).to(self.device) * (0.1 - 0.0)

                # Train Discrimator
                d_loss, fake_score, real_score, fake_mazes = self.D.train(self.G.model,
                                                                          self.G.input_size,
                                                                          maze_set,
                                                                          loss_criterion,
                                                                          real_labels,
                                                                          fake_labels,
                                                                          self.reset_grad)

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
                        self.writer.add_histogram("CNNGAN/Generator/" + name, param.clone().cpu().data.numpy(), epoch + 1)

                    for name, param in self.D.model.named_parameters():
                        self.writer.add_histogram("CNNGAN/Discriminator/" + name, param.clone().cpu().data.numpy(),
                                                  epoch + 1)

                    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                          .format(epoch + 1, self.num_epochs, local_batch + 1, total_step, d_loss.item(), g_loss.item(),
                                  real_score.mean().item(), fake_score.mean().item()))

            # Save real mazes
            if (epoch + 1) == 1:
                #print(maze_set.shape)
                #maze_set = maze_set.reshape(maze_set.size(0), self.mx, self.my)
                #maze_set = maze_set.reshape(maze_set.size(0), 3, 244, 244)
                dump_file(os.path.join(self.path + self.maze_dir, 'real_mazes.pickle'), maze_set)

            # Save sampled mazes
            #fake_mazes = fake_mazes.reshape(fake_mazes.size(0), self.mx, self.my)
            #print(fake_mazes.shape)
            fake_mazes = fake_mazes.reshape(fake_mazes.size(0), 3, 244, 244)
            dump_file(os.path.join(self.path + self.maze_dir, 'fake_mazes-{}.pickle'.format(epoch + 1)), fake_mazes)

        if not os.path.exists(self.path + self.model_dir):
            os.makedirs(self.path + self.model_dir)
        torch.save(self.G.model.state_dict(), self.path + self.model_dir + '/G.ckpt')
        torch.save(self.D.model.state_dict(), self.path + self.model_dir + '/D.ckpt')

        self.writer.export_scalars_to_json("./tensorboard_data.json")  # use this istead of pickle??
        self.writer.close()

    def create_training_data(self):
        gen_maze_data(self.N, self.mx, self.my, save_to_file=True, dir=self.path + self.training_dir)

    def load_data(self):
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                root=self.path + self.training_dir,
                flist=os.listdir(self.path + self.training_dir),
                transform=transforms.Compose([transforms.RandomSizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              ])),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)
        return train_loader


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist

#For data loading
class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist  # flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #    target = self.target_transform(target)
        return img

    def __len__(self):
        return len(self.imlist)
