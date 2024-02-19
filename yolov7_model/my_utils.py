import torch
import torch.nn as nn
import torch.nn.functional as F

def distance_based_weights(distances, num_of_scales):
    '''
    Function that calculates the loss function weights for each scale depending on the label distances for each sample from the batch.

    Parameters:
        distances (torch.Tensor): Tensor of disatnces between labels for each sample of shape (batch_size, 1)
        num_of_scales (int): The number of scales the yolo model is detecting object at. (Usually 3)

    Returns:
        torch.Tensor: A tensor of weights for each sample of shape (batch_size, num_of_scales)
    '''

    batch_size = distances.shape[0]

    # Determine intervals for each scale
    range_values = torch.linspace(0.1, 0.5, num_of_scales).repeat(batch_size, 1)

    # Initialize weights tensor
    distances = distances.repeat(1, num_of_scales)

    # Create Gaussians centered around ranges_values
    std_dev = 0.22  # Changing this dictates how much weight should be put on the "correct" scale (higher = more equal, lower = higher peak)
    gaussians = torch.exp(-(distances - range_values)**2 / (2 * std_dev**2))

    # Normalize weights across scales for each sample
    weights = F.normalize(gaussians, p=1, dim=1)

    return weights
    

def weighted_loss(outputs, targets, distances):
    total_loss = 0
    for scale_outputs, scale_targets, distance in zip(outputs, targets, distances):
        # Calculate the loss for each scale (e.g., using MSE loss)
        scale_loss = F.mse_loss(scale_outputs, scale_targets, reduction='none')  # Disable reduction
        # Calculate distance-based weights for the current scale
        scale_weights = distance_based_weights(distances, 3)
        # Weight the loss for the current scale
        weighted_scale_loss = torch.sum(scale_loss * scale_weights.unsqueeze(1))  # Element-wise multiplication
        # Accumulate the weighted loss
        total_loss += weighted_scale_loss
    return total_loss


class MyDetectSingle(nn.Module):
    # My detection layers to output the position of single elbow and wrist

    def __init__(self, num_of_layers, ch=()):  # detection layer
        super(MyDetectSingle, self).__init__()
        self.nc = 2  # number of classes
        self.nl = num_of_layers  # number of detection layers
        #self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.m = nn.ModuleList()
        for x in ch:
            conv_layers = nn.Sequential(
                nn.Conv2d(x, x//2, kernel_size=5, stride=4, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(x, x//4, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.Conv2d(x//4, self.nc * 3, kernel_size=1)
            )
            self.m.append(conv_layers)

    def forward(self, x):
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, _, _ = x[i].shape
            x[i] = x[i].view(bs, self.nc * 3).contiguous()

            if not self.training:  # inference
                y = x[i].sigmoid()
                y[:, 1:3] *= 416  # xy elbow
                y[:, 4:6] *= 416  # xy wrist
                z.append(y)

        if self.training:
            out = x
        else:
            out = (torch.cat(z, 1), x)

        return out

class MyDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, num_of_layers, ch=()):  # detection layer
        super(MyDetect, self).__init__()
        self.nc = 2  # number of classes
        self.nl = num_of_layers  # number of detection layers
        #self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.m = nn.ModuleList(nn.Conv2d(x, self.nc * 3, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.nc * 3, ny, nx).permute(0, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

#weights = distance_based_weights(torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5]).reshape(-1, 1), 3)
#print(weights)
