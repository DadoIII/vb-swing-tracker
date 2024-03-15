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
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Load and preprocess sample here
        return sample


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
        # Implement your scale-specific loss computation here
        # For example, you can use functions from torch.nn.functional (F) to compute the loss
        
        # Example:
        pred_x_elbow, pred_y_elbow, pred_conf_elbow, pred_x_wrist, pred_y_wrist, pred_conf_wrist = torch.split(pred, 1, dim=-1)
        target_x_elbow, target_y_elbow, target_conf_elbow, target_x_wrist, target_y_wrist, target_conf_wrist = torch.split(target, 1, dim=-1)
        
        # Compute the mask for reliable predictions based on ground truth confidence scores
        mask_elbow = (target_conf_elbow == 1).squeeze(-1)
        mask_wrist = (target_conf_wrist == 1).squeeze(-1)
        
        # Compute the positional loss for the elbow (considering only confident predictions)
        loss_x_elbow = F.mse_loss(pred_x_elbow[mask_elbow], target_x_elbow[mask_elbow])
        loss_y_elbow = F.mse_loss(pred_y_elbow[mask_elbow], target_y_elbow[mask_elbow])
        
        # Compute the positional loss for the wrist (considering only confident predictions)
        loss_x_wrist = F.mse_loss(pred_x_wrist[mask_wrist], target_x_wrist[mask_wrist])
        loss_y_wrist = F.mse_loss(pred_y_wrist[mask_wrist], target_y_wrist[mask_wrist])
        
        # Compute the confidence loss for the elbow and wrist
        loss_conf_elbow = F.binary_cross_entropy_with_logits(pred_conf_elbow.squeeze(-1)[mask_elbow], target_conf_elbow.squeeze(-1)[mask_elbow])
        loss_conf_wrist = F.binary_cross_entropy_with_logits(pred_conf_wrist.squeeze(-1)[mask_wrist], target_conf_wrist.squeeze(-1)[mask_wrist])
        
        # Aggregate the losses for elbow and wrist
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