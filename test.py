import torch

x = torch.arange(4)
x = x.view(4, 1, 1)
# x_expanded = x.expand(4, 3, 3)

# x_expanded = x_expanded.reshape(1, 2, 3, 2, 3)
# x_expanded = x_expanded.permute(1, 3, 0, 2, 4)
# x_expanded = x_expanded.contiguous().view(1, 6, 6)
# print(x_expanded)

x_expanded = x.expand(4, 3, 3)

# 重新排列张量以获得预期的6x6形状
x_expanded = x_expanded.reshape(2, 2, 3, 3)
x_expanded = x_expanded.permute(2, 0, 3, 1)
x_expanded = x_expanded.contiguous().view(6, 6)
print(x_expanded)

