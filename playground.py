import torch
import torch.nn as nn
import torchvision.models


a = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
b = torch.tensor([[2, 2, 2], [3, 3, 3], [1, 1, 1]])

a_reshaped = a.reshape([a.shape[0], 1, a.shape[1]])
b_reshaped = b.reshape([b.shape[0], b.shape[1], 1])

bla = torch.matmul(a_reshaped, b_reshaped).squeeze(dim=1)

h = 5