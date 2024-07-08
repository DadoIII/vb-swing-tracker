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
        #self.BCE = nn.BCEWithLogitsLoss(reduction='mean')

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
        scale_outputs = {"losses": {}, "accs": {}, "FP": {}}

        # Loop through predictions and targets at each scale and compute the loss at each scale
        for pred, target in zip(predictions, targets):
            scale_loss, acc, false_positives = self.compute_scale_loss(pred, target)
            # Accumulate the losses across all scales
            total_loss += scale_loss
            x, y = pred.shape[1:3]
            scale_outputs["losses"][f"{x},{y}"] = float(scale_loss)
            scale_outputs["accs"][f"{x},{y}"] = float(acc)
            scale_outputs["FP"][f"{x},{y}"] = float(false_positives)
        
        # Average the losses across scales
        average_loss = total_loss / len(predictions)

        return average_loss, scale_outputs

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
        true_positives = 0
        false_positives = 0
        positive_targets = 0

        # Compute loss for each keypoint
        for i in range(num_keypoints):
            pred_x, pred_y, pred_conf = F.sigmoid(pred[i][:, :, :, 0]), F.sigmoid(pred[i][:, :, :, 1]), pred[i][:, :, :, 2]
            target_x, target_y, target_conf = target[i][:, :, :, 0], target[i][:, :, :, 1], target[i][:, :, :, 2]

            # Scale by the grid cell size
            scaled_pred_x = pred_x * self.image_width / x_cells
            scaled_target_x = target_x * self.image_width / x_cells
            scaled_pred_y = pred_y * self.image_height / y_cells
            scaled_target_y = target_y * self.image_height / y_cells

            # Compute the mask for reliable predictions based on ground truth confidence scores
            mask = (target_conf == 1).squeeze(-1)

            positive_keypoints = torch.sum(mask)
            pos_weight = (x_cells * y_cells - positive_keypoints) / positive_keypoints
            binary_cross_entropy = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

            # Compute the positional loss for the current keypoint (considering only ground truth keypoints)
            if mask.nonzero().numel() > 0:
                loss_x = F.mse_loss(scaled_pred_x[mask], scaled_target_x[mask])
                loss_y = F.mse_loss(scaled_pred_y[mask], scaled_target_y[mask])
            else:
                loss_x = loss_y = torch.tensor(0.0, device=pred[i].device, requires_grad=True)

            # Compute the confidence loss for the current keypoint
            #loss_conf = F.binary_cross_entropy(pred_conf.view(-1), target_conf.view(-1))
            
            #pred_conf_masked = pred_conf * mask

            loss_conf = binary_cross_entropy(pred_conf, target_conf)

            # Accumulate the losses
            #keypoint_loss = loss_x + loss_y + (loss_conf * 10)
            keypoint_loss = loss_conf
            scale_loss += keypoint_loss

             # Calculate true positives and false positives
            pred_labels = (pred_conf > 0.8).squeeze()  # Predicted labels based on confidence threshold
            positive_targets += torch.sum(mask)  # Count the number of positive targets
            true_positives += torch.sum(pred_labels & mask).item()  # Count true positives
            false_positives += torch.sum(pred_labels & ~mask).item()  # Count false positives

        positive_accuracy = true_positives / positive_targets

        #return scale_loss
        return scale_loss, positive_accuracy, false_positives


    def compute_benchmark(self, predictions, targets, confidence = 0.5):
        POSITIONS = ("elbow_right", "wrist_right", "elbow_left", "wrist_left")
        scale_outputs = {"accs": {}, "FP": {}}
        
        bs, x_count, y_count, _ = targets[0].shape
        x_size, y_size = self.image_width / x_count, self.image_height / y_count

        # Collect the coordinates of the targets
        total_TP = [0 for _ in range(bs)]
        total_FP = [0 for _ in range(bs)]
        true_positives = [{"elbow_right": {}, "wrist_right": {}, "elbow_left": {}, "wrist_left": {}} for _ in range(bs)]

        for batch in range(bs):
            for keypoint_index in range(0, 11, 3):
                mask = targets[0][batch, :, :, keypoint_index+2] > 0.5
                indices = torch.nonzero(mask)

                for idx in indices:
                    x, y = idx.tolist()
                    x_pos = int(x * x_size + x_size * targets[0][batch, x, y, keypoint_index])
                    y_pos = int(y * y_size + y_size * targets[0][batch, x, y, keypoint_index+1])
                    total_TP[batch] += 1
                    true_positives[batch][POSITIONS[keypoint_index//3]][(x_pos, y_pos)] = False

        # Collect positive predicitons from the original yolov7 model
        # And calculate the the number of FP and accuracy
        for batch, prediction in enumerate(predictions):
            prediction_keypoints = [[] for _ in range(4)]
            for skeleton in prediction:
                if (right_elbow := get_elbow_from_skeleton(skeleton, left_handed=False, check_confidence=True, confidence_threshold=confidence)) is not None:
                    prediction_keypoints[0].append(right_elbow) 
                if (right_wrist := get_wrist_from_skeleton(skeleton, left_handed=False, check_confidence=True, confidence_threshold=confidence)) is not None:
                    prediction_keypoints[1].append(right_wrist)
                if (left_elbow := get_elbow_from_skeleton(skeleton, left_handed=True, check_confidence=True, confidence_threshold=confidence)) is not None:
                    prediction_keypoints[2].append(left_elbow)
                if (left_wrist := get_wrist_from_skeleton(skeleton, left_handed=True, check_confidence=True, confidence_threshold=confidence)) is not None:
                    prediction_keypoints[3].append(left_wrist)

            for pos in range(4):
                true_keypoints = true_positives[batch][POSITIONS[pos]]                
                for keypoint in prediction_keypoints[pos]: 
                    false_positive = True 
                    for true_keypoint in true_keypoints.keys():
                        if true_keypoints[true_keypoint] == False and get_keypoint_distance(keypoint, true_keypoint) <= 0.04 * (self.image_width + self.image_height) / 2:
                            true_keypoints[true_keypoint] = True
                            false_positive = False
                            break
                    if false_positive:
                        total_FP[batch] += 1

        acc_sum = 0
        for i, batch_TP in enumerate(true_positives):
            temp_sum = 0
            for keypoint_type in list(batch_TP.values()):
                temp_sum += sum(list(keypoint_type.values()))
            acc_sum += temp_sum / total_TP[i]
        
        return acc_sum / bs, sum(total_FP) / bs


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
    model.train()
    epoch_loss = 0.0

    scale_outputs_batch = {"losses": {}, "accs": {}, "FP": {}}

    if update_weights:
        optimiser.zero_grad()

    for batch_idx, (images, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(images)

        # Compute loss
        loss, scale_outputs = criterion(outputs, targets)

        if update_weights:
            # Backward pass
            loss.backward()

            with open("gradient_log.txt", "a") as gradient_log:
                # Print gradients for each parameter
                for name, param in model.named_parameters():
                    if param.grad is not None and name.startswith('model.118.m_kpt.3.11'):
                        gradient_log.write(f'Parameter: {name}, Mean Gradient: {param.grad.mean()}, Max Gradient: {param.grad.max()}, Min Gradient: {param.grad.min()}\n')

        # Accumulate loss
        epoch_loss += loss.item()
        for output in scale_outputs.keys():
            for scale, value in scale_outputs[output].items():
                if scale in scale_outputs_batch[output].keys():
                    scale_outputs_batch[output][scale] += value
                else:
                    scale_outputs_batch[output][scale] = value

    # Update weights
    if update_weights:
        optimiser.step()

    if scheduler is not None:
        scheduler.step()
        #print(f"Learning Rate: {scheduler.get_lr()}")

    # Calculate average epoch loss
    epoch_loss /= len(dataloader)

    for output in scale_outputs_batch.keys():
        for scale, value in scale_outputs_batch[output].items():
            if output == 'accs':
                scale_outputs_batch[output][scale] = round(scale_outputs_batch[output][scale] / len(dataloader) * 100, 1)
            elif output == 'losses':
                scale_outputs_batch[output][scale] = round(scale_outputs_batch[output][scale] / len(dataloader), 2)
            elif output == 'FP':
                scale_outputs_batch[output][scale] = round(scale_outputs_batch[output][scale] / len(dataloader))

    return epoch_loss, scale_outputs_batch

def run_benchmark(model, dataloader, criterion):
    model.eval()

    scale_outputs_batch = {"accs": {}, "FP": {}}

    for batch_idx, (images, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(images)

        # Compute loss
        scale_outputs = criterion.compute_benchmark(outputs, targets)

       
        # Accumulate loss
        for output in scale_outputs.keys():
            for scale, value in scale_outputs[output].items():
                if scale in scale_outputs_batch[output].keys():
                    scale_outputs_batch[output][scale] += value
                else:
                    scale_outputs_batch[output][scale] = value


    for output in scale_outputs_batch.keys():
        for scale, value in scale_outputs_batch[output].items():
            if output == 'accs':
                scale_outputs_batch[output][scale] = round(scale_outputs_batch[output][scale] / len(dataloader) * 100, 1)
            elif output == 'FP':
                scale_outputs_batch[output][scale] = round(scale_outputs_batch[output][scale] / len(dataloader))

    return scale_outputs_batch


def main():
    #torch.manual_seed(1)

    image_width = image_height = 960
    num_epochs = 50
    lr = 5e-4
    momentum = 0.9
    weight_decay = 0.15
    lr_decay = True  # Learning rate decay
    model_name = f'{num_epochs}_epochs_{lr}_lr_{momentum}_m_{weight_decay}_wd_lr_decay={lr_decay}'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create dataset
    batch_size = 6
    labeled_image_folder = "../images/labeled_images/"
    #scales = [(120, 120), (60, 60), (30, 30), (15, 15)]
    scales = [(30, 30), (15, 15)]
    dataset = CustomDataset("../images/labels/annotations_multi.csv", labeled_image_folder, scales, device)
    #train_set, val_set, temp = torch.utils.data.random_split(dataset, [2626, 500, 4])
    train_set, val_set = torch.utils.data.random_split(dataset, [2200, 250])
    #train_set, val_set, temp = torch.utils.data.random_split(dataset, [500, 300, 2330])
    #train_set, val_set, temp = torch.utils.data.random_split(dataset, [1000, 300, 1830])
    #train_set, val_set, temp = torch.utils.data.random_split(dataset, [1, 1, 1468])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    #Initialise loss
    criterion = CustomLoss(image_width, image_height)

    # Define model
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    _ = model.eval()

#     target_train_loss, target_train_scale_outputs = run_epoch(model, train_loader, criterion, update_weights=False)
#     target_val_loss, target_val_scale_outputs = run_epoch(model, val_loader, criterion, update_weights=False)
#     target_stats = f'''===== Target stats =====
# Training Loss: {target_train_loss:.4f}, Validation Loss: {target_val_loss:.4f}
# Training accuracy: {target_train_scale_outputs['accs']}%
# Validation accuracy: {target_val_scale_outputs['accs']}%
# Training false positives: {target_train_scale_outputs['FP']}
# Validation false positives: {target_val_scale_outputs['FP']}
# Train losses: {target_train_scale_outputs['losses']}
# Val losses: {target_val_scale_outputs['losses']}'''


    # Adjust model
    #layer = MyIKeypoint(ch=(256, 512, 768, 1024))
    #layer.f = [114, 115, 116, 117]
    layer = MyIKeypoint(ch=(768, 1024))
    layer.f = [116, 117]
    layer.i = 118

    new_m_kpt = nn.ModuleList()
    for i, module_list in enumerate(layer.m_kpt):
        new_module_list = nn.Sequential()
        for j, module in enumerate(module_list):
            # Skip the last layer for each head
            if j < len(module_list) - 1:
                # Copy parameters from the original model to the new module
                state_dict = model.model[-1].m_kpt[i+2][j].state_dict()
                module.load_state_dict(state_dict)
            new_module_list.add_module(str(j), module)
        new_m_kpt.append(new_module_list)

    # Assign the new m_kpt sequence to the new layer
    layer.m_kpt = new_m_kpt
    model.model[-1] = layer

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

   # Make parameters of the last layers trainable
    for head in model.model[-1].m_kpt:
        for param in head.parameters():
            param.requires_grad = True
        # for i, layer in enumerate(head):
        #     if i >= len(head) - 1:  # Check if the layer index is one of the last layer
        #         for param in layer.parameters():
        #             param.requires_grad = True

    if torch.cuda.is_available():
        model.half().to(device)

    #optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    optimiser = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    # Scheduler for learning rate decay
    if lr_decay:
        scheduler = lr_scheduler.StepLR(optimiser, step_size=50, gamma=0.5)
    else:
        scheduler = None

    # Empty the txt file
    with open("gradient_log.txt", "w") as gradient_log:
        pass

    with open('training_progress_' + model_name + '.txt', 'w') as file:
        file.write("Epoch,Training Loss,Validation Loss\n")
        
        # Print losses before the first epoch
        epoch_loss, train_scale_outputs = run_epoch(model, train_loader, criterion, update_weights=False)
        val_loss, val_scale_outputs = run_epoch(model, val_loader, criterion, update_weights=False)
        print(f'==== Epoch [0/{num_epochs}] ====')
        print(f'Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print(f'Training accuracy: {train_scale_outputs['accs']}%')
        print(f'Validation accuracy: {val_scale_outputs['accs']}%')
        print(f'Training false positives: {train_scale_outputs['FP']}')
        print(f'Validation false positives: {val_scale_outputs['FP']}')
        print(f'Train losses: {train_scale_outputs['losses']}')
        print(f'Val losses: {val_scale_outputs['losses']}')
        file.write(f"0,{epoch_loss},{val_loss}\n")

        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_loss, train_scale_outputs = run_epoch(model, train_loader, criterion, optimiser, scheduler)
            val_loss, val_scale_outputs = run_epoch(model, val_loader, criterion, update_weights=False)
            end_time = time.time()
            print(f'==== Epoch [{epoch+1}/{num_epochs}] ====  Time: {end_time-start_time}s')
            print(f'Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
            print(f'Training accuracy: {train_scale_outputs['accs']}%')
            print(f'Validation accuracy: {val_scale_outputs['accs']}%')
            print(f'Training false positives: {train_scale_outputs['FP']}')
            print(f'Validation false positives: {val_scale_outputs['FP']}')
            print(f'Train losses: {train_scale_outputs['losses']}')
            print(f'Val losses: {val_scale_outputs['losses']}')

            # Write epoch results to file
            file.write(f"{epoch+1},{epoch_loss},{val_loss}\n")

    torch.save(model, model_name + '.pt')

if __name__ == "__main__":
    main()