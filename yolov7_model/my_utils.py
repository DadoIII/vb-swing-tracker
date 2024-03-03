import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math

from models.common import ImplicitA, ImplicitM, DWConv, Conv

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


class MyIKeypoint(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), nkpt=2, ch=(), inplace=True, dw_conv_kpt=False):  # detection layer
        super(MyIKeypoint, self).__init__()
        self.nc = nc  # number of classes
        self.nkpt = nkpt
        self.dw_conv_kpt = dw_conv_kpt
        self.no_det=(nc + 5)  # number of outputs per anchor for box and class
        self.no_kpt = 3*self.nkpt ## number of outputs per anchor for keypoints
        self.no = self.no_det+self.no_kpt
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.flip_test = False
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no_det * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)
        
        if self.nkpt is not None:
            if self.dw_conv_kpt: #keypoint head is slightly more complex
                self.m_kpt = nn.ModuleList(
                            nn.Sequential(DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1)) for x in ch)
            else: #keypoint head is a single convolution
                self.m_kpt = nn.ModuleList(nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch)

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            if self.nkpt is None or self.nkpt==0:
                x[i] = self.im[i](self.m[i](self.ia[i](x[i])))  # conv
            else :
                x[i] = torch.cat((self.im[i](self.m[i](self.ia[i](x[i]))), self.m_kpt[i](x[i])), axis=1)

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_det = x[i][..., :6]
            x_kpt = x[i][..., 6:]

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.nkpt == 0:
                    y = x[i].sigmoid()
                else:
                    y = x_det.sigmoid()

                if self.inplace:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2) # wh
                    if self.nkpt != 0:
                        x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        #x_kpt[..., 0::3] = (x_kpt[..., ::3] + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        #x_kpt[..., 1::3] = (x_kpt[..., 1::3] + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        #print('=============')
                        #print(self.anchor_grid[i].shape)
                        #print(self.anchor_grid[i][...,0].unsqueeze(4).shape)
                        #print(x_kpt[..., 0::3].shape)
                        #x_kpt[..., 0::3] = ((x_kpt[..., 0::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        #x_kpt[..., 1::3] = ((x_kpt[..., 1::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        #x_kpt[..., 0::3] = (((x_kpt[..., 0::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        #x_kpt[..., 1::3] = (((x_kpt[..., 1::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                    y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    if self.nkpt != 0:
                        y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[i].repeat((1,1,1,1,self.nkpt))) * self.stride[i]  # xy
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



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
    
def get_keypoint_distance(kpt1, kpt2):
    return math.sqrt((kpt1[0] - kpt2[0]) ** 2 + (kpt1[1] - kpt2[1]) ** 2)

def get_elbow_from_skeleton(kpts, steps, left_handed = False):
    if left_handed:
        return kpts[7*steps:8*steps]
    else:
        return kpts[8*steps:9*steps]
    
def get_wrist_from_skeleton(kpts, steps, left_handed):
    if left_handed:
        return kpts[9*steps:10*steps]
    else:
        return kpts[10*steps:11*steps]


def plot_elbow_wrist(im, kpts, steps, left_handed = False):
    # Assume the full original skeleton keypoints were passed in
    if len(kpts) > 2 * steps:
        kpts = torch.cat([get_elbow_from_skeleton(kpts, steps, left_handed), get_wrist_from_skeleton(kpts, steps, left_handed)], dim=0)

    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 0, 0], [0, 255, 0]])

    radius = 5
    num_kpts = len(kpts) // steps

    # Draw the circle
    for kid in range(num_kpts):
        r, g, b = palette[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)


def plot_elbow_wrist_history(im, history_kpts):
    palette = np.array([[255, 0, 0], [0, 255, 0]])

    # Draw the line for previous frames
    if history_kpts:
        for kpts in history_kpts:
            if len(kpts) > 8:
                for i in range(0, len(kpts) - 8, 4):
                    if kpts[i] != -1 and kpts[i+4] != -1:
                        cv2.line(im, (kpts[i], kpts[i+1]), (kpts[i+4], kpts[i+5]), (255, 0, 0), thickness=2)
                    if kpts[i+2] != -1 and kpts[i+6] != -1:
                        cv2.line(im, (kpts[i+2], kpts[i+3]), (kpts[i+6], kpts[i+7]), (0, 255, 0), thickness=2)

class KeypointHistory:
    def __init__(self):
        self.active_history = []
        self.old_history = []

    def append_keypoints(self, kpts: torch.tensor, steps = 3, left_handed = False):
        if self.active_history != []:
            print("new cycle")
            my_iter = iter(self.active_history)

            while True:
                try:
                    item = next(my_iter)
                    last_positions = self.get_latest_position(item)

                    chosen_idx = None
                    smallest_distance = 50
                    chosen_elbow_pos = None
                    chosen_wrist_pos = None

                    for idx in range(kpts.shape[0]):
                        new_positions = kpts[idx, :]
                        elbow_pos = get_elbow_from_skeleton(new_positions, steps, left_handed).tolist()
                        wrist_pos = get_wrist_from_skeleton(new_positions, steps, left_handed).tolist()
                        if last_positions[2:] == [-1, -1]:
                            distance = get_keypoint_distance(elbow_pos[:2], last_positions[:2])
                        elif last_positions[:2] == [-1, -1]:
                            distance = get_keypoint_distance(wrist_pos[:2], last_positions[2:])
                        else:
                            distance = get_keypoint_distance(elbow_pos[:2], last_positions[:2]) + get_keypoint_distance(wrist_pos[:2], last_positions[2:]) / 2
                        #print("distance", distance, elbow_pos[2], wrist_pos[2])
                        if distance < smallest_distance:
                            smallest_distance = distance
                            chosen_idx = idx
                            chosen_elbow_pos = elbow_pos
                            chosen_wrist_pos = wrist_pos
                
                    if chosen_idx != None:
                        if elbow_pos[2] > 0.5:
                            item += [int(x) for x in chosen_elbow_pos[:2]]
                        else:
                            item += item + [-1, -1]
                        if wrist_pos[2] > 0.5:
                            item += item + [int(x) for x in chosen_wrist_pos[:2]]
                        else:
                            item += [-1, -1]
                        print("here")
                        kpts = torch.cat([kpts[:chosen_idx, :], kpts[chosen_idx + 1:, :]], dim=0)


                except StopIteration:
                    break
        
        # If there appears to be new keypoints, add them to active history 
        if kpts.shape[0] > 0:
            for idx in range(kpts.shape[0]):
                self.active_history.append([int(x) for x in get_elbow_from_skeleton(kpts[idx, :], steps, left_handed).tolist()[:2]] + [int(x) for x in get_wrist_from_skeleton(kpts[idx, :], steps, left_handed).tolist()[:2]])

        self.remove_inactive()

    
    def remove_inactive(self):
        my_iter = iter(self.active_history)

        while True:
            try:
                history = next(my_iter)
                if history[-min(len(history), 20):] == [-1, -1, -1, -1] * min(len(history), 20):
                    self.active_history.remove(history)
                    self.old_history.append(history)

            except StopIteration:
                break


    def get_latest_position(self, bounding_box):
        for i in range(len(bounding_box) - 1, -1, -4):
            if bounding_box[max(i - 3, 0):i + 1] == [-1,-1,-1,-1]:
                continue
            else:
                return bounding_box[max(i - 3, 0):i + 1]
        return None

    def get_history(self):
        return self.active_history + self.old_history
    
    def stats(self):
        print("active history size:", len(self.active_history))
        print("old history size:", len(self.old_history))

# TODO: COntinue with last position and tracking image overlay
class LastPosition:
    def __init__(self):
        self.prev_positions = []

    def new_positions(self, new_positions):
        self.prev_positions = new_positions

# Function to overlay a transparent image onto another image
def overlay_transparent(background, overlay, position=(0, 0)):
    h, w = overlay.shape[:2]

    # Extract the alpha channel from the overlay image
    overlay_alpha = overlay[:, :, 3] / 255.0

    # Calculate the region of interest (ROI) for the overlay
    y1, y2 = position[1], position[1] + h
    x1, x2 = position[0], position[0] + w

    # Blend the images using alpha blending
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (1 - overlay_alpha) * background[y1:y2, x1:x2, c] + \
                                        overlay_alpha * overlay[:, :, c]

    return background
