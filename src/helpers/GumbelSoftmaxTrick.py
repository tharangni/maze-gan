import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.MazeGenerator import draw_maze

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).type(Tensor)
    return -Variable(torch.log(-torch.log(U + eps) + eps).type(Tensor))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def pass_through(outputs, temperature):
    probs = torch.stack([1.0 - outputs, outputs], dim=-1).type(Tensor)
    print(probs)
    logits = torch.log(probs)

    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).type(Tensor).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    return ((y_hard - y).detach() + y)[:, :, 1]


if __name__ == '__main__':
    mazes = torch.rand(4, 1, 28, 28)
    mazes = Variable(Tensor(mazes))
    mazes = mazes.reshape(4, -1)

    probs = torch.stack([1.0 - mazes, mazes], dim=-1)
    logits = torch.log(probs)

    # maze_w_gradient = pass_through(mazes, 0.8)
    maze_w_gradient = F.gumbel_softmax(logits.view(-1, 2), tau=0.8, hard=True)[:, 1]
    for maze in maze_w_gradient.view(4, 28, 28):
        draw_maze(maze)
    print(maze_w_gradient.size(), maze_w_gradient)
