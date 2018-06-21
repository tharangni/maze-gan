import torch
from torch import Tensor
from torch.autograd import Variable

from helpers.MazeGenerator import draw_maze

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def heaviside(x):
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    x_hard = torch.where(x >= 0, ones, zeros)

    return x_hard


def pass_through(mazes):
    """
    input: [*] continuous Tensor
    return: [*] discrete_vanilla Tensor with gradient
    """
    mazes_hard = heaviside(mazes)

    return ((mazes_hard - mazes).detach() + mazes).to(device=DEVICE, dtype=torch.float32)


if __name__ == '__main__':
    mazes = torch.randn(4, 4, 4)
    mazes = Variable(Tensor(mazes))
    mazes = mazes.reshape(4, -1)

    maze_w_gradient = pass_through(mazes)
    for maze in maze_w_gradient:
        draw_maze(maze.reshape(4, 4))
    print(maze_w_gradient)
