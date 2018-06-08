from gan import GAN
import torch
import os


def start():
    #look for cmd arguments here

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mx = 4
    my = 4
    N = 200
    maze_size = 16
    latent_size = 4
    hidden_size = 8
    num_epochs = 200
    batch_size = 100
    maze_dir = 'maze_results'

    if not os.path.exists(maze_dir):
        os.makedirs(maze_dir)

    gan = GAN(device,
              latent_size,
              hidden_size,
              maze_size,
              num_epochs,
              batch_size,
              mx,
              my,
              N,
              maze_dir)

    #train
    gan.train()

    #save gan
if __name__ == '__main__':
    start()
