import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.MazeGenerator import gen_maze_data, draw_maze


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


if __name__ == '__main__':
    mazes = gen_maze_data(100, 4, 4)
    mazes = Variable(Tensor(mazes))
    mazes = mazes.reshape(100, -1)
    maze_mean = torch.mean(mazes, dim=0)
    draw_maze(maze_mean.view(4, 4))
    soft_maze = gumbel_softmax(maze_mean, 0.8)
    draw_maze(soft_maze.view(4, 4))
