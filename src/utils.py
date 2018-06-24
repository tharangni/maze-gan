import pickle
import sys
import numpy as np
import torch.nn as nn
import torch
import cv2
from matplotlib import pyplot as plt
import importlib


def dump_file(loc, data):
    output = open(loc, 'wb')
    pickle.dump(data, output)
    output.close()


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


# Visualising image in batch
def visualise_image_pickle(name):
    print("Visualise arguments ", name)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    mazes = pickle.load(open(name, 'rb'))
    for maze in mazes[:10]:
        # is it a valid maze?
        if torch.cuda.is_available():
            maze = maze.cpu()
        maze = maze.detach().numpy()
        plt.imshow(maze, cmap='gray', aspect='auto')
        plt.pause(1.5)
        plt.close()

    plt.close()


if __name__ == '__main__':
    visualise_image_pickle(sys.argv)
