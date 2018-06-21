import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.MazeGenerator import draw_maze

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def pass_through(logits, temperature):
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

    return torch.argmax((y_hard - y).detach() + y, dim=2).to(device=DEVICE, dtype=torch.float32)


if __name__ == '__main__':
    mazes = torch.rand(4, 28, 28)
    mazes = Variable(Tensor(mazes))
    mazes = mazes.reshape(4, -1)
    print(mazes)
    maze_probs = torch.stack((1 - mazes, mazes), dim=2)
    maze_logits = torch.log(maze_probs)

    print(maze_logits.size(), maze_logits)
    maze_w_gradient = pass_through(maze_logits, 0.8)
    for maze in maze_w_gradient:
        draw_maze(maze.reshape(28, 28))
    print(maze_w_gradient)
