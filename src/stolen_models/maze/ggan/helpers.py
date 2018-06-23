import glob
import os

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import label


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


class Checkpoint:

    def __init__(self, state, model_name):
        self.state = state
        self.path = os.path.join('checkpoints', model_name + '.checkpoint.pth.tar')
        os.makedirs('checkpoints', exist_ok=True)

    def save(self):
        if self.state is None:
            raise ValueError('No state provided.')
        torch.save(self.state, self.path)

    def load(self, model):
        checkpoint = torch.load(self.path)
        start_epoch = checkpoint.epoch
        model.model.load_state_dict(checkpoint.model)
        model.optimizer.load_state_dict(checkpoint.optimizer)

        return start_epoch


class MazeHelper:

    @staticmethod
    def save_maze_grid(mazes, path):
        fig = plt.figure(1, dpi=160)

        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(5, 5),  # creates 5x5 grid of axes
                         axes_pad=0.04,  # pad between axes in inch.
                         )
        for i in range(25):
            grid[i].imshow(mazes[i, :, :], cmap='gray')
            grid[i].axes.get_xaxis().set_visible(False)
            grid[i].axes.get_yaxis().set_visible(False)

        plt.savefig(path, dpi=160, bbox_inches='tight')

    @staticmethod
    def check_maze(maze):
        # single connected-component
        labeled_array, num_features = label(maze)
        np_maze = np.array(maze)
        mx, my = np_maze.shape
        if num_features > 1:
            return False
        # no loops
        maze = 1 - maze
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        labeled_array, num_features = label(maze, structure=s)
        for feat in range(1, num_features + 1):
            indexes = np.array(np.where(labeled_array == feat))

            if np.all(indexes[:, :] != 0) and np.all(indexes[0, :] != mx - 1) and np.all(indexes[1, :] != my - 1):
                return False

        return True

    @staticmethod
    def draw_maze(maze, time=1.5):
        plt.autoscale(tight=True)
        plt.imshow(maze, cmap='gray')
        plt.pause(time)
        plt.close()

    @staticmethod
    def test_all_results_ind(directory):
        files = glob.glob(directory + '/fake_*.sample')
        files.sort()
        for idx, file in enumerate(files):
            correct = 0
            total = 0
            mazes = torch.load(file, map_location='cpu').detach()
            total += mazes.size(0)
            for each_maze in mazes:
                if MazeHelper.check_maze(each_maze):
                    correct += 1
            print(file, ':', correct, "/", total)


if __name__ == '__main__':
    MazeHelper.test_all_results_ind('mazes')

    data = torch.load('mazes/fake_800.sample')
    print(data.size())
    for d in data[:25, :, :]:
        MazeHelper.draw_maze(d, 0.5)
