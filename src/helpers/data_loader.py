import os
from argparse import Namespace

import torch
from torchvision import datasets
from torchvision.transforms import transforms

ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
CUDA = True if torch.cuda.is_available() else False
TENSOR = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def mnist(opt: Namespace, binary: bool, is_image: bool = False) -> torch.Tensor:
    os.makedirs(os.path.join(ROOT, 'data', 'mnist'), exist_ok=True)
    transform = []

    if not is_image:
        transform.append(transforms.ToPILImage())
    transform.append(transforms.Resize(opt.img_size))
    transform.append(transforms.ToTensor())
    if binary:
        transform.append(transforms.Lambda(lambda x: torch.round(x)))
    else:
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    data = datasets.MNIST(os.path.join(ROOT, 'data', 'mnist'), train=True, download=True,
                          transform=transforms.Compose(transform))
    mnist_loader = torch.zeros(data.train_data.size(0), opt.img_size, opt.img_size).type(TENSOR)

    for idx in range(len(data)):
        mnist_loader[idx], _ = data[idx]

    return mnist_loader.reshape(-1, opt.batch_size, 1, opt.img_size, opt.img_size).type(TENSOR)


def mazes(opt: Namespace) -> torch.Tensor:
    data_path = os.path.join(ROOT, 'data', 'mazes',
                             '{}.{}x{}.data'.format(opt.n_examples, opt.maze_size, opt.maze_size))

    return torch.load(data_path).type(TENSOR).reshape(-1, opt.batch_size, 1, opt.maze_size, opt.maze_size)
