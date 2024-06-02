import torch
import numpy as np

# def cross_entropy_loss(logits, target):
#     # Compute softmax
#     exp_logits = torch.exp(logits)
#     softmax = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
    
#     # Negative log likelihood
#     neg_log_likelihood = -torch.log(softmax[range(targets.shape[0]), targets])
    
#     # Average loss across the batch
#     loss = torch.mean(neg_log_likelihood)
    
#     return loss

#     softmax = torch.exp(input) / torch.sum(torch.exp(input), dim=1, keepdim=True)
        
#     # Convert targets to one-hot encoded vectors
#     one_hot = torch.nn.functional.one_hot(target)

#     return torch.mean(-torch.log(softmax[range(target.shape[0]), target]))

# # Example usage:
# logits = torch.tensor([[2.0, 1.0, 0.1],  # Example logits for 3 classes
#                        [0.1, 2.0, 1.0],
#                        [0.1, 1.0, 2.0]])
# targets = torch.tensor([0, 1, 2])  # Example true labels

# loss = cross_entropy_loss(logits, targets)
# print(loss.item())

class Conv2d:

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size
    ):
        self.stride = 1
        self.padding = 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize kernel weights and bias
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.random.randn(out_channels)

    def __call__(self, input):
        padded_input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        batch_size, _, in_width, in_height = padded_input.shape

        out_width = (in_height - self.kernel_size) // self.stride + 1
        out_height = (in_width - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for bs in range(batch_size):
            for ch in range(self.out_channels):
                for i in range(0, in_height - self.kernel_size + 1, self.stride):
                    for j in range(0, in_width - self.kernel_size + 1, self.stride):
                        output[bs, ch, i // self.stride, j // self.stride] = np.sum(padded_input[bs, :, i:i+self.kernel_size, j:j+self.kernel_size] * self.weight[ch, :, :, :]) + self.bias[ch]

        return output

input = np.ones((2, 2, 5, 5))
c = Conv2d(1, 1, 5)
print(c.__call__(input))
