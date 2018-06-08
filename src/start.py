from gan import GAN
import torch
import os


def start():
    #look for cmd arguments here

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_size = 64
    hidden_size = 256
    image_size = 784
    num_epochs = 200
    batch_size = 100
    img_dir = 'image_results'

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    gan = GAN(device,
              latent_size,
              hidden_size,
              image_size,
              num_epochs,
              batch_size,
              img_dir)

    #train
    gan.train()

    #save gan
if __name__ == '__main__':
    start()