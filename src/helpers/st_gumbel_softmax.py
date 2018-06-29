from torch.autograd import Variable
from torch.nn import functional
import torch

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def sample_gumbel(shape: tuple, eps: float = 1e-20) -> Variable:
    """Sample from the Gumbel distribution according to the input shape.

    Args:
        shape: The shape of the i.i.d samples.
        eps: small number for stability

    Returns:
        A sample from the Gumbel distribution.
    """
    u = torch.rand(shape).type(Tensor)
    return -Variable(torch.log(-torch.log(u + eps) + eps))


def gumbel_softmax_sample(logits: Variable, tau: float) -> Variable:
    """Sample from the Gumbel-Softmax distribution given some log-probabilities and a temperature parameter.

    Args:
        logits: log-odds of a categorical variable represented in one-hot-encoding
            (i.e. batch_size x num_features x num_categories).
        tau: a number used to smooth the distribution | the higher the temperature, the more closer you get
            to the original categorical distribution.

    Returns:
        A sample from the Gumbel-Softmax distribution in the same shape as the input logits. This is a continuous
            Variable that has a gradient.
    """
    y = logits + sample_gumbel(logits.size())
    return functional.softmax(y / tau, dim=-1)


def quantize(y: Variable) -> Variable:
    """Discretize a Variable of binary probabilities to a one-hot representation and output a binary Variable.

    Args:
        y:  A continuous Variable of probabilities of shape batch_size x num_features x num_categories, where the last
        dimension sums to 1 (i.e. for each pixel of each maze in the batch)

    Returns:
        The discretized Variable.
    """
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1]).type(Tensor)
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def straight_through(logits: Variable, tau: float, hard=True) -> Variable:
    """Sample from the the Gumbel-Softmax distribution and optionally discretize the output.  The output contains only
    the last element of the last dimension, which says whether a pixel in a maze should be white or black. The discrete
    output will be approximated in the backward pass using the continuous representation, thus maintaining a gradient.

    Args:
        logits: Log-odds of a categorical variable represented in one-hot-encoding as
            batch_size x num_features x num_categories.
        tau: A number used to smooth the distribution | the higher the temperature, the closer you get
            to the original categorical distribution.
        hard: Whether to return the discrete representation of the of Gumbel-Softmax sample.

    Returns:
        The (quantized) Variable representing the pixel values.
    """
    y = gumbel_softmax_sample(logits, tau)

    if hard:
        return quantize(y)[:, :, -1]
    else:
        return y[:, :, 1]
