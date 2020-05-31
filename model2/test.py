import torch
from torch import nn

x = torch.randn(1, 3)
y = torch.randn(1, 3)
z = torch.cat((x, y), 1)
pass
