import torch

x = torch.tensor([[[.1, .2, .4],
                   [5., 6., 7.],
                   [8., 9., 10.],
                   [11., 12., 13.]]])
print(x.size())
mask = [
    [1, 0, 1],
    [1, 0, 1],
    [1, 0, 1],
    [1, 1, 0],
]
torch.BoolTensor()
mask = torch.BoolTensor(mask)
z = torch.masked_select(x, mask).view(4, -1)
print(z)
pass
