import torch
from torch import nn

x = torch.tensor([[[.0, .0, .0], [1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]])
print(x.size())
m = nn.AdaptiveAvgPool2d((1,4))
print(m(x))
pass
