import torch

from transformers import BertConfig

from model2.TBert import TBert

model = TBert(BertConfig())
model.eval()

t1_ids = torch.tensor(model.ntokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
t2_ids = torch.tensor(model.ntokenizer.encode("Hello, my cat is cute", add_special_tokens=True)).unsqueeze(0)

with torch.no_grad():
    score = model(t1_ids, None, t2_ids, None)
    print(score)

    score2 = model(t1_ids, None, t2_ids, None)
    print(score2)

    t1_embd = model.create_nl_embed(t1_ids, None)
    t2_embd = model.create_pl_embed(t2_ids, None)
    print(model.cls(t1_embd, t2_embd))
    print(model.cls(t2_embd, t1_embd))



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
