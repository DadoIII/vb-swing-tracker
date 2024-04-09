import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import csv
import numpy as np
from torchvision import transforms
from typing import List
import time

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from models.yolo import MyIKeypoint

from my_utils import *

class CustomDataset(Dataset):
    def __init__(self, annotation_path, image_folder, scales, device="cpu"):
        self.data, self.targets = read_csv_file(annotation_path)
        self.image_folder = image_folder
        self.scales = scales
        self.device = device

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
            torch.Tensor: A tensor of shape (x_num_cells, y_num_cells, 12) representing the target for the supplied index.
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
    def __init__(self, image_width, image_height):
        super(CustomLoss, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.BCE = nn.BCEWithLogitsLoss(reduction='mean')

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

        # Get the number of grid cells
        x_cells, y_cells = pred.shape[1], pred.shape[2]

        # Split predictions and targets into separate tensors for each keypoint
        pred = pred.split(3, dim=-1)  # Split along the last dimension
        target = target.split(3, dim=-1)
        scale_loss = 0.0

        # Compute loss for each keypoint
        for i in range(num_keypoints):
            pred_x, pred_y,  pred_conf = F.sigmoid(pred[i][:, :, :, 0]), F.sigmoid(pred[i][:, :, :, 1]), pred[i][:, :, :, 2]
            target_x, target_y, target_conf = target[i][:, :, :, 0], target[i][:, :, :, 1], target[i][:, :, :, 2]

            # Scale by the grid cell size
            scaled_pred_x = pred_x * self.image_width / x_cells
            scaled_target_x = target_x * self.image_width / x_cells
            scaled_pred_y = pred_y * self.image_height / y_cells
            scaled_target_y = target_y * self.image_height / y_cells

            # Compute the mask for reliable predictions based on ground truth confidence scores
            mask = (target_conf == 1).squeeze(-1)

            # Compute the positional loss for the current keypoint (considering only ground truth keypoints)
            if mask.nonzero().numel() > 0:
                loss_x = F.mse_loss(scaled_pred_x[mask], scaled_target_x[mask])
                loss_y = F.mse_loss(scaled_pred_y[mask], scaled_target_y[mask])
            else:
                loss_x = loss_y = torch.tensor(0.0, device=pred[i].device, requires_grad=True)

            # Compute the confidence loss for the current keypoint
            #loss_conf = F.binary_cross_entropy(pred_conf.view(-1), target_conf.view(-1))
            
            loss_conf = self.BCE(pred_conf, target_conf)

            # Accumulate the losses
            keypoint_loss = loss_x + loss_y + (loss_conf * 10)
            scale_loss += keypoint_loss
            
        #return scale_loss
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

def run_epoch(model, dataloader, criterion, optimiser=None, scheduler=None, update_weights=True):
    model.train() if update_weights else model.eval()
    epoch_loss = 0.0

    for batch_idx, (images, targets) in enumerate(dataloader):
        if update_weights:
            optimiser.zero_grad()

        # Forward pass
        outputs = model(images)

        if not update_weights:
            outputs = outputs[1]

        #(outputs[0][0,0,0,:])

        # Compute loss
        loss = criterion(outputs, targets)

        if update_weights:
            # Backward pass
            loss.backward()

            with open("gradient_log.txt", "a") as gradient_log:
                # Print gradients for each parameter
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradient_log.write(f'Parameter: {name}, Mean Gradient: {param.grad.mean()}, Max Gradient: {param.grad.max()}, Min Gradient: {param.grad.min()}\n')

            # Update weights
            optimiser.step()

            if scheduler != None:
                scheduler.step()

        # Accumulate loss
        epoch_loss += loss.item()

    # Calculate average epoch loss
    epoch_loss /= len(dataloader)

    return epoch_loss

def main():
    #torch.manual_seed(1)

    image_width = image_height = 960
    num_epochs = 125
    lr = 1e-4
    momentum = 0.85
    weight_decay = 0.2
    lr_decay = True  # Learning rate decay
    model_name = f'{num_epochs}_epochs_{lr}_lr_{momentum}_m_{weight_decay}_wd_lr_decay={lr_decay}'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create dataset
    batch_size = 6
    labeled_image_folder = "../images/labeled_images/"
    scales = [(120, 120), (60, 60), (30, 30), (15, 15)]
    dataset = CustomDataset("../images/labels/annotations_multi.csv", labeled_image_folder, scales, device)
    train_set, val_set = torch.utils.data.random_split(dataset, [1200, 270])
    #train_set, val_set, temp = torch.utils.data.random_split(dataset, [240, 24, 1206])
    #train_set, val_set, temp = torch.utils.data.random_split(dataset, [1, 1, 1468])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    #Initialise loss
    criterion = CustomLoss(image_width, image_height)

    # Define model
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    # Adjust model
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

    #optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.01)
    optimiser = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Scheduler for learning rate decay
    if lr_decay:
        scheduler = lr_scheduler.StepLR(optimiser, step_size=25, gamma=0.9)
    else:
        scheduler = None

    # Empty the txt file
    with open("gradient_log.txt", "w") as gradient_log:
        pass

    with open('training_progress_' + model_name + '.txt', 'w') as file:
        file.write("Epoch,Training Loss,Validation Loss\n")

        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_loss = run_epoch(model, train_loader, criterion, optimiser, scheduler)
            val_loss = run_epoch(model, val_loader, criterion, update_weights=False)
            end_time = time.time()
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {end_time-start_time}s')

            # Write epoch results to file
            file.write(f"{epoch+1},{epoch_loss},{val_loss}\n")

    torch.save(model, model_name + '.pt')

if __name__ == "__main__":
    main()