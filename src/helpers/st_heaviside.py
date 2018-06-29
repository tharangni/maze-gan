import torch
from torch.autograd import Variable

TENSOR = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def heaviside(x: Variable):
    ones = torch.ones_like(x).type(TENSOR)
    zeros = torch.zeros_like(x).type(TENSOR)
    x_hard = torch.where(x >= 0, ones, zeros).type(TENSOR)

    return x_hard


def straight_through(x: Variable) -> Variable:
    """Discretize a continuous Variable using the Heaviside function. The discrete Variable will be approximated in the
    backward pass using the continuous representation, thus maintaining a gradient.

    Args:
        x: The variable to discretize, a Tensor of any shape.

    Returns:
        A discretized Variable that retains the gradient from the original representation.
    """

    x_hard = heaviside(x)

    return (x_hard - x).detach() + x
