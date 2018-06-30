import torch
import torch.nn as nn


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)
