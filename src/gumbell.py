import torch
from torch.autograd import Variable

temperature = 5

def sample_gumbel(shape, eps=1e-20):
  U = torch.rand(shape)
  return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits):
  sample = Variable(sample_gumbel(logits.size()[-1]))
  if logits.is_cuda:
    sample = sample.cuda()
  y = logits + sample
  return torch.nn.functional.softmax(y/temperature)


def gumbel_softmax(logits, hard=False):
  y = gumbel_softmax_sample(logits)
  return y