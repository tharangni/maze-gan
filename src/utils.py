import pickle
import sys
import numpy as np
import torch.nn as nn
import torch
import cv2
from matplotlib import pyplot as plt

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

def visualise_image_pickle(name):
    print("Visualise arguments ", name)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    # path = os.path.join(dir, 'real_mazes.pickle')
    # visualise sample from final results
    mazes = pickle.load(open(name, 'rb'))
    import cv2
    # Load an color image in grayscale
    #img = cv2.imread('../training_maze0.png', 0)#0 for greyscale #
    #print("Image shape ", img.shape)
    #cv2.imshow('image', img)
    #a = plt.imshow(img, cmap='gray', aspect='auto')
    #print(a.get_array().shape)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #image = plt.imread("../training_maze0.png")
    #plt.pause(5)
    # takes sample and plot
    print("ogggg ",mazes.shape)
    for maze in mazes[:10]:
        maze[maze < 0.5] = 0
        maze[maze > 0.5] = 1
        # is it a valid maze?
        if torch.cuda.is_available():
            maze = maze.cpu()
        maze = maze.detach().numpy()
        print(mazes.shape)
        #print(maze.shape)
        print("og, ", maze.shape)
        print("new ", maze.shape)
        x = plt.imshow(maze, cmap='gray', aspect='auto')  # makes image square but turn blocks into rectangles
        print("x ", x)
        plt.pause(5)
        #plt.close()

if __name__ == '__main__':
    visualise_image_pickle(sys.argv)