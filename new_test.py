import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

# Calculate sigmoid(0)
sigmoid_0 = torch.sigmoid(torch.tensor(-1))

# Calculate -10 * log(sigmoid(0))
result = -10 * 0.5 * torch.log(sigmoid_0) / 2
print(sigmoid_0)
print(result.item())