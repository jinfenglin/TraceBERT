import torch


a = [1,2,3]
b = [4,5,6]
print(a *3 + b)

# a = torch.tensor([1, 2, 3])
# b = torch.tensor([0, 2, 6])
# print(torch.abs(a - b))

# def custom_avg_pooler(hidden_states, attention_mask):
#     pooled_tensors = []
#     for item, mask in zip(hidden_states, attention_mask):  # shape of (512,728)
#         mask = (mask > 0)
#         # token features after masking
#         masked_hidden = torch.masked_select(item, mask.view(-1, 1)).view(-1, 3)
#         pooled_hidden = torch.mean(masked_hidden, dim=0)
#         pooled_tensors.append(pooled_hidden)
#     return torch.stack(pooled_tensors)
#
#
# x = torch.tensor([[[.1, .2, .4],
#                    [5., 6., 7.],
#                    [8., 9., 10.],
#                    [11., 12., 13.]],
#
#                   [[.1, .2, .4],
#                    [5., 6., 7.],
#                    [8., 9., 10.],
#                    [11., 12., 13.]]
#
#                   ])
#
# mask = [[1, 0, 1, 0],
#         [1, 1, 1, 0]]
# mask = torch.tensor(mask)
#
# z = custom_avg_pooler(x, mask)
# print(z)
# pass
