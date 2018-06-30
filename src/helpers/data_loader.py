import os
from argparse import Namespace
from typing import Union, Dict, Any

import torch
from torchvision import datasets
from torchvision.transforms import transforms

ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
CUDA = True if torch.cuda.is_available() else False
TENSOR = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def load(args: Namespace, opts: Dict[str, Any]):
    """Load either MNIST or MAZE data.

    Args:
        args: The CLI arguments.
        opts: Additional model-specific options. Useful only for MNIST to specify, resize, binarization, etc.

    Returns:
        The dataset as a Tensor fully loaded into memory, shaped according to batch size and maze size..
    """
    if args.dataset == 'mnist':
        return _mnist(args, **opts)
    elif args.dataset == 'mazes':
        return _mazes(args)
    else:
        raise ValueError('Unknown dataset {}'.format(args.dataset))


def _mnist(args: Namespace, binary: bool, crop: Union[None, int] = None) -> torch.Tensor:
    """Load the MNIST dataset.

    Args:
        args: The CLI arguments.
        binary: Whether to binarize the tensors.
        crop: The size of the image after a center crop. If None, will not crop the image.

    Returns:
        The MNIST dataset as a Tensor fully loaded into memory, shaped according to batch size and maze size..
    """
    os.makedirs(os.path.join(ROOT, 'data', 'mnist'), exist_ok=True)
    transform = []

    if crop is not None:
        transform.append(transforms.CenterCrop(crop))

    transform.append(transforms.Resize(args.img_size))
    transform.append(transforms.ToTensor())

    if binary:
        transform.append(transforms.Lambda(lambda x: torch.round(x)))
    else:
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    data = datasets.MNIST(os.path.join(ROOT, 'data', 'mnist'), train=True, download=True,
                          transform=transforms.Compose(transform))
    mnist_loader = torch.zeros(data.train_data.size(0), args.img_size, args.img_size).type(TENSOR)

    for idx in range(len(data)):
        mnist_loader[idx], _ = data[idx]

    return mnist_loader.reshape(-1, args.batch_size, 1, args.img_size, args.img_size).type(TENSOR)


def _mazes(args: Namespace) -> torch.Tensor:
    """Load the MAZE dataset.

    Args:
        args: The CLI arguments.

    Returns:
        The MAZE dataset as a Tensor fully loaded into memory, shaped according to batch size and maze size.
    """
    data_path = os.path.join(ROOT, 'data', 'mazes',
                             '{}.{}x{}.data.tar'.format(args.n_examples, args.maze_size, args.maze_size))

    return torch.load(data_path).type(TENSOR).reshape(-1, args.batch_size, 1, args.maze_size, args.maze_size)
