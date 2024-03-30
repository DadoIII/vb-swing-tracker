import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import csv
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
    def __init__(self, data, targets, image_folder, scales, device="cpu"):
        self.data = data
        self.targets = targets
        self.device = device
        self.image_folder = image_folder
        self.scales = scales

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_folder + self.data[idx])
        #image = letterbox(image, 960, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        if torch.cuda.is_available():
            image = image.half().to(self.device)

        targets = []
        for (x, y) in self.scales:
            targets.append(self.get_target(idx, x, y))

        return image, targets
    
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
        processed_targets = torch.zeros((x_num_cells, y_num_cells, 12)).half()

        # Format and normalise labels
        for i, values in enumerate(targets):
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
                processed_targets[box_x,box_y, i*3: i*3+3] = torch.tensor([value_x, value_y, 1]).half()

        return processed_targets.to(self.device)
        

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Compute the custom loss between model predictions and ground truth targets.
        
        Parameters:
            predictions (list): List of tensors containing model predictions at each scale.
                                Shape: [(bs, x1, y1, 12), (bs, x2, y2, 12), ..., (bs, xn, yn, 12)]
            targets (list): List of tensors containing ground truth targets at each scale.
                            Shape: [(bs, x1, y1, 12), (bs, x2, y2, 12), ..., (bs, xn, yn, 12)]
        
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
        average_loss = total_loss / len(predictions)
        
        return average_loss

    def compute_scale_loss(self, pred, target):
        """
        Compute the loss for a single scale.
        
        Parameters:
            pred (tensor): Predictions tensor for a single scale.
                           Shape: (bs, x, y, 12)
            target (tensor): Ground truth targets tensor for a single scale.
                             Shape: (bs, x, y, 12)
        
        Returns:
            scale_loss (tensor): Scalar tensor representing the computed loss for the scale.
        """

        # Define the number of keypoints
        num_keypoints = 4

        #print("---------- pred ------------")
        #print(pred.view(-1)[:50])
        #print("---------- target ------------")
        #print(target.view(-1)[:50])

        # Split predictions and targets into separate tensors for each keypoint
        pred = pred.split(3, dim=-1)  # Split along the last dimension
        target = target.split(3, dim=-1)
        scale_loss = 0.0

        confidence_criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # Compute loss for each keypoint
        for i in range(num_keypoints):
            pred_x, pred_y,  pred_conf = pred[i][:, :, :, 0], pred[i][:, :, :, 1], pred[i][:, :, :, 2]
            target_x, target_y,  target_conf = target[i][:, :, :, 0], target[i][:, :, :, 1], target[i][:, :, :, 2]

            # Compute the mask for reliable predictions based on ground truth confidence scores
            mask = (target_conf == 1).squeeze(-1)

            # Compute the positional loss for the current keypoint (considering only ground truth keypoints)
            if mask.nonzero().numel() > 0:
                loss_x = F.mse_loss(pred_x[mask], target_x[mask])
                loss_y = F.mse_loss(pred_y[mask], target_y[mask])
            else:
                loss_x = loss_y = torch.tensor(0.0, device=pred[i].device)

            # Compute the confidence loss for the current keypoint
            #loss_conf = F.binary_cross_entropy(pred_conf.view(-1), target_conf.view(-1))
            
            loss_conf = confidence_criterion(pred_conf, target_conf)

            # Accumulate the losses
            keypoint_loss = loss_x + loss_y + loss_conf
            #print("loss x:", loss_x)
            #print("loss y:", loss_y)
            #print("loss conf:", loss_conf)
            scale_loss += keypoint_loss
            
        return scale_loss


def read_csv_file(file_path):
    image_names = []
    targets = []

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            image_names.append(row[0])

            row_list = []  # List of all keypoints for each image
            for keypoints in row[1:]:
                temp_list = []  # List of one type of keypoints
                keypoints = keypoints.split(" ")
                if len(keypoints) > 1:
                    for i in range(0,len(keypoints), 2):
                        temp_list.append((float(keypoints[i]), float(keypoints[i+1])))
                row_list.append(temp_list)
            targets.append(row_list)

    return image_names, targets

def run_epoch(model, dataloader, optimiser, criterion):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (images, targets) in enumerate(dataloader):
        # Zero the gradients
        optimiser.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Print gradients for each parameter
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f'Parameter: {name}, Mean Gradient: {param.grad.mean()}, Max Gradient: {param.grad.abs().max()}')

        # Update weights
        optimiser.step()

        # Accumulate loss
        epoch_loss += loss.item()

    # Calculate average epoch loss
    epoch_loss /= len(dataloader)

    return epoch_loss

def main():
    num_epochs = 15

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    
    image_names, targets = read_csv_file("../images/labels/annotations_multi.csv")

    # Create dataset
    batch_size = 6
    labeled_image_folder = "../images/labeled_images/"
    scales = [(120, 120), (60, 60), (30, 30), (15, 15)]
    dataset = CustomDataset(image_names, targets, labeled_image_folder, scales, device)
    subset_dataset = dataset[:6]
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    #Initialise loss
    criterion = CustomLoss()

    layer = MyIKeypoint(ch=(256, 512, 768, 1024))
    layer.f = [114, 115, 116, 117]
    layer.i = 118
    model.model[-1] = layer

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Make parameters of the last layers trainable
    for param in model.model[-1].parameters():
        param.requires_grad = True

    if torch.cuda.is_available():
        model.half().to(device)

    optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    for epoch in range(num_epochs):
        epoch_loss = run_epoch(model, dataloader, optimiser, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        

if __name__ == "__main__":
    main()