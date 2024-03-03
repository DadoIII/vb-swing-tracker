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
    # Calucate the euclidian distance between two keypoints (x1, y1), (x2, y2)
    return math.sqrt((kpt1[0] - kpt2[0]) ** 2 + (kpt1[1] - kpt2[1]) ** 2)

def get_elbow_from_skeleton(kpts, check_confidence=False, confidence_threshold=0.5, left_handed=False):
    """
    Gets the x, y, confidence values of the elbow from the full yolov7-keypoints skeleton.

    Parameters:
        kpts (torch.tensor): All of the yolov7-keypoints.
        check_confidence (bool): Whether to automatically check the confidence of the prediction. When set to True, it returns the x and y only when the prediction confidence is higher than the confidence_threshold. It will not return the confidence anymore.
        confidence_threshold (double): Only used when check_confidence is True to set the confidence_threshold.
        left_handed (bool): Whether to track the left arm.

    Returns:
        torch.tensor or tuple or None: Depending on the conditions, it returns different types:
            : If check_confidence is False, returns a torch.tensor containing the x, y, and confidence values.
            : If check_confidence is True and the confidence of the prediction is higher than confidence_threshold, returns a tuple containing the x and y values only.
            : If check_confidence is True but the confidence of the prediction is below confidence_threshold, returns None.
    """
    if left_handed:
        pos = kpts[7*3:8*3]
    else:
        pos = kpts[8*3:9*3]

    if not check_confidence:
        return pos
    elif pos[-1] > confidence_threshold:
        return pos[:2]
    else:
        return None

    
def get_wrist_from_skeleton(kpts, check_confidence=False, confidence_threshold=0.5, left_handed=False):
    """
    Gets the x, y, confidence values of the wrist from the full yolov7-keypoints skeleton.

    Parameters:
        kpts (torch.tensor): All of the yolov7-keypoints.
        check_confidence (bool): Whether to automatically check the confidence of the prediction. When set to True, it returns the x and y only when the prediction confidence is higher than the confidence_threshold. It will not return the confidence anymore.
        confidence_threshold (double): Only used when check_confidence is True to set the confidence_threshold.
        left_handed (bool): Whether to track the left arm.

    Returns:
        torch.tensor or tuple or None: Depending on the conditions, it returns different types:
            : If check_confidence is False, returns a torch.tensor containing the x, y, and confidence values.
            : If check_confidence is True and the confidence of the prediction is higher than confidence_threshold, returns a tuple containing the x and y values only.
            : If check_confidence is True but the confidence of the prediction is below confidence_threshold, returns None.
    """
    if left_handed:
        pos = kpts[9*3:10*3]
    else:
        pos = kpts[10*3:11*3]

    if not check_confidence:
        return pos
    elif pos[-1] > confidence_threshold:
        return pos[:2]
    else:
        return None


def plot_elbow_wrist(im, kpts, left_handed=False):
    """
    Plots circles indicating a elbow and a wrist on an image.

    Paremeters:
        im (np.ndarray): Image to plot the keypoints on.
        kpts (torch.tensor): Tensor of keypoints. Can be a tensor of the full yolov7-keypoints skeleton. Or any array of length 6 with values: [elbow_x, elbow_y, elbow_confidence, wrist_x, wrist_y, wrist_confidence]
        left_handed (bool): Whether to track the left arm. (Only works for the full yolov7-keypoints skeleton)
    """
    # Assume the full original skeleton keypoints were passed in
    # So extract the elbow and wrist values
    if len(kpts) > 6:
        kpts = torch.cat([get_elbow_from_skeleton(kpts, left_handed), get_wrist_from_skeleton(kpts, left_handed)], dim=0)

    palette = np.array([[255, 0, 0], [0, 255, 0]])
    radius = 5
    num_kpts = len(kpts) // 3

    # Draw the circle
    for kid in range(num_kpts):
        r, g, b = palette[kid]
        x_coord, y_coord = kpts[3 * kid], kpts[3 * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            conf = kpts[3 * kid + 2]
            if conf < 0.5:
                continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

class LastPositions:
    def __init__(self):
        self.prev_positions = []

    def find_continuations(self, new_positions, continuation_distance = 50):
        """
        Loops over the list of new positions and tries to find continuations from previous positions.
        A continuation is when the new keypoint position is within a certain distance from the previous position (default is 50 pixels)

        Parameters:
            new_positions [(x,y)]: A list of tuples containing the x and y coordiantes of new positions
            continuation_distance (int): A number indicationg the maximum pixel distance that is considered as a continuation 

        Returns:
            List(Tuple(Tuple)): A list containing tuples. Each tuple has 2 tuples with x,y coordinates indicating the starting and ending position of the lines to be drawn.

        Example:
            The returned list has the following structure:
            [
                ((10, 10), (20, 20)),  # A line from x:10, y:10 to x:20, y:20
                ((40, 50), (90, 80)),  # A line from x:40, y:50 to x:90, y:80
            ]
        """
        return_lines = []
        for new_pos in new_positions:
            pos = None
            for prev_pos in self.prev_positions:
                if get_keypoint_distance(new_pos, prev_pos) < continuation_distance:
                    pos = prev_pos
            if pos != None:
                return_lines.append((pos, new_pos))
                self.prev_positions.remove(pos)

        self.prev_positions = new_positions

        return return_lines

    def get_positions(self):
        return self.prev_positions

def draw_lines(image, positions, colour=(0, 0, 255, 255), thickness=2):
    for (pos1, pos2) in positions:
        draw_line(image, pos1, pos2, colour, thickness)

def draw_line(image, point1, point2, colour=(0, 0, 0, 255), thickness=2):
    cv2.line(image, point1, point2, colour, thickness)


def overlay_transparent(background, overlay, position=(0, 0)):
    """
    Overlays a transparent image onto a background. The blending done per pixel and based on the transparency of the overlay pixels.
    
    Parameters:
        background (np.ndarray): Background image to use.
        overlay (np.ndarray): Overlay image to use. (Has to be RGBA)
        position Tuple(int): (x, y) offset of the overlay image.

    Returns:
        np.ndarray: The resulting image with the overlay on top

    Example:
        overlay pixel value: (255, 0, 0, 255) + background pixel value (0, 255, 255) -> (255, 0, 0)
        overlay pixel value: (200, 0, 0, 127) + background pixel value (0, 200, 0) -> (100, 100, 0)
        overlay pixel value: (255, 0, 0, 0) + background pixel value (0, 255, 0) -> (0, 255, 0)
    """
    h, w = overlay.shape[:2]
    y1, y2 = position[1], position[1] + h
    x1, x2 = position[0], position[0] + w

    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (alpha_overlay * overlay[:, :, c] +
                                       alpha_background * background[y1:y2, x1:x2, c])

    return background
