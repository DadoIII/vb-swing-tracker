import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from typing import List

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from models.yolo import MyIKeypoint

from my_utils import *

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_target(self, idx: int, x_num_cells: int, y_num_cells: int):
        """
        Gets targets far a dataset with a specific index and specific scale.

        Parameters:
            idx (int): The index of the datapoint to get the target for.
            x_num_cells (int): Number of grid cells in the x direction to preprocess the targets for.
            x_num_cells (int): Number of grid cells in the y direction to preprocess the targets for.

        Returns: 
            torch.Tensor: A tensor of shape (x_num_cells, y_num_cells, 6) representing the target for the supplied index.
        """
        targets = self.targets[idx]

        # Create empty x x y x 12 tensor
        processed_targets = np.zeros((x_num_cells, y_num_cells, 12))

        # Format and normalise labels
        for i, values in enumerate(targets.values()):
            for (x, y) in values:
                # Calculate the cell sizes
                x_cell_size = 1 / x_num_cells 
                y_cell_size = 1 / y_num_cells
                # Figure out which box the label belongs to
                box_x = int((x * 10) / (x_cell_size * 10))  # Multiplying by 10 because of limited floating-point precision
                box_y = int((y * 10) / (y_cell_size * 10))
                # Normalise the width and height within the box
                value_x = round(((x * 10) % (x_cell_size * 10)) / (x_cell_size * 10), 3)
                value_y = round(((y * 10) % (x_cell_size * 10)) / (x_cell_size * 10), 3)
                processed_targets[box_x,box_y, i*3: i*3+3] = [value_x, value_y, 1]

        return processed_targets
        

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Compute the custom loss between model predictions and ground truth targets.
        
        Args:
            predictions (list): List of tensors containing model predictions at each scale.
                                Shape: [(bs, x1, y1, 6), (bs, x2, y2, 6), ..., (bs, xn, yn, 6)]
            targets (list): List of tensors containing ground truth targets at each scale.
                            Shape: [(bs, x1, y1, 6), (bs, x2, y2, 6), ..., (bs, xn, yn, 6)]
        
        Returns:
            loss (tensor): Scalar tensor representing the computed loss.
        """
        total_loss = 0.0
        
        # Loop through predictions and targets at each scale
        for pred, target in zip(predictions, targets):
            # Compute the loss for each scale (you can use any suitable loss function)
            scale_loss = self.compute_scale_loss(pred, target)
            # Accumulate the losses across all scales
            total_loss += scale_loss
        
        # Average the losses across scales
        average_loss = total_loss / self.num_scales
        
        return average_loss

    def compute_scale_loss(self, pred, target):
        """
        Compute the loss for a single scale.
        
        Args:
            pred (tensor): Predictions tensor for a single scale.
                           Shape: (bs, x, y, 6)
            target (tensor): Ground truth targets tensor for a single scale.
                             Shape: (bs, x, y, 6)
        
        Returns:
            scale_loss (tensor): Scalar tensor representing the computed loss for the scale.
        """

        pred_x_elbow, pred_y_elbow, pred_conf_elbow, pred_x_wrist, pred_y_wrist, pred_conf_wrist = torch.split(pred, 1, dim=-1)
        target_x_elbow, target_y_elbow, target_conf_elbow, target_x_wrist, target_y_wrist, target_conf_wrist = torch.split(target, 1, dim=-1)
        
        # Compute the mask for reliable predictions based on ground truth confidence scores
        mask_elbow = (target_conf_elbow == 1).squeeze(-1)
        mask_wrist = (target_conf_wrist == 1).squeeze(-1)
        
        # Compute the positional loss for the elbow (considering only confident predictions)
        if mask_elbow.nonzero().numel() > 0:
            loss_x_elbow = F.mse_loss(pred_x_elbow[mask_elbow], target_x_elbow[mask_elbow])
            loss_y_elbow = F.mse_loss(pred_y_elbow[mask_elbow], target_y_elbow[mask_elbow])
        else:
            loss_x_elbow = loss_y_elbow = torch.tensor(0.0, device=pred.device)

        # Compute the positional loss for the wrist (considering only confident predictions)
        if mask_wrist.nonzero().numel() > 0:
            loss_x_wrist = F.mse_loss(pred_x_wrist[mask_wrist], target_x_wrist[mask_wrist])
            loss_y_wrist = F.mse_loss(pred_y_wrist[mask_wrist], target_y_wrist[mask_wrist])
        else:
            loss_x_wrist = loss_y_wrist = torch.tensor(0.0, device=pred.device)
        
        #print(F.binary_cross_entropy(torch.Tensor([0]), torch.Tensor([1])))
        # Compute the confidence loss for the elbow and wrist
        loss_conf_elbow = F.binary_cross_entropy(pred_conf_elbow.view(-1), target_conf_elbow.view(-1))
        loss_conf_wrist = F.binary_cross_entropy(pred_conf_wrist.view(-1), target_conf_wrist.view(-1))
        
        # Aggregate the losses for elbow and wrist
        # print(loss_x_elbow)
        # print(loss_y_elbow)
        # print(loss_x_wrist)
        # print(loss_y_wrist)
        # print(loss_conf_elbow)
        # print(loss_conf_wrist)
        scale_loss = loss_x_elbow + loss_y_elbow + loss_x_wrist + loss_y_wrist + loss_conf_elbow + loss_conf_wrist
        
        return scale_loss

def run_epoch(model, optimiser):
    pass

def main():
    num_epochs = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    

    layer = MyIKeypoint(ch=(256, 512, 768, 1024))
    layer.f = [114, 115, 116, 117]
    layer.i = 118
    model.model[-1] = layer

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Make parameters of the last layers trainable
    for param in model[-1].parameters():
        param.requires_grad = True

    if torch.cuda.is_available():
        model.half().to(device)

    optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    for epoch in range(num_epochs):
        run_epoch(model, optimiser)

if __name__ == "__main__":
    main()