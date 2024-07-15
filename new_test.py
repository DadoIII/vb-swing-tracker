import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

pos_weight = torch.tensor([2])
binary_cross_entropy = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

pred = torch.Tensor([[1,1,1],[1,1,1]])
target = torch.Tensor([[0,0,0],[0,0,0]])

loss_conf = binary_cross_entropy(pred, target)
print(loss_conf)