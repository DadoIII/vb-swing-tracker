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
        num_of_scales (int): The number of scales the yolo model is detecting object at. (Usually 4)

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

def elbow_wrist_nms(predictions, conf_thres=0.5, overlap_distance=0.05):
    """
    Compiles a list of detected keypoints above a certain confidence threshold and removes ones of the same type that are too close.

    Parameters:
        predictions (List(torch.Tensor)): A list of tensors of shape (bs, x_grids, y_grids, 12)
        conf_thres (float): A threshold for confidence of a singular elbow or wrist prediction
        overlap_distance (float): Distance, which is considered as overlapping keypoints. It should be a ratio betwenn 0 and 1.

    Returns:
            keypoints: A 3d list containing tuples (x, y, conf) keypoint detections, where x and y are represent a ratio of the width/height.
                The three dimensions are for the different images in the batch, the different keypoint types and each individual keypoint within each type.
    """

    bs = predictions[0].shape[0]
    keypoints = [[[],[],[],[]] for _ in range(bs)]

    # for scale in prediction:
    #     x_size = 1 / scale.shape[1]
    #     y_size = 1 / scale.shape[2]
    #     elbow_mask = scale[:, :, :, 2] > conf_thres
    #     wrist_mask = scale[:, :, :, -1] > conf_thres
    #     elbow_indices = torch.nonzero(elbow_mask)
    #     wrist_indices = torch.nonzero(wrist_mask)
    #     for idx in elbow_indices:
    #         bs, x, y = idx.tolist()
    #         x_pos = float(x * x_size + x_size * scale[bs, x, y, 0])
    #         y_pos = float(y * y_size + y_size * scale[bs, x, y, 1])
    #         conf = float(scale[bs, x, y, 2])
    #         overlapping = False
    #         for x, y, _ in elbows:
    #             if get_keypoint_distance((x,y), (x_pos, y_pos)) < overlap_distance:
    #                 overlapping = True
    #         if not overlapping:
    #             elbows.append((x_pos, y_pos, conf))

    #     for idx in wrist_indices:
    #         bs, x, y = idx.tolist()
    #         x_pos = float(x * x_size + x_size * scale[bs, x, y, 3])
    #         y_pos = float(y * y_size + y_size * scale[bs, x, y, 4])
    #         conf = float(scale[bs, x, y, 5])
    #         overlapping = False
    #         for x, y, _ in wrists:
    #             if get_keypoint_distance((x,y), (x_pos, y_pos)) < overlap_distance:
    #                 overlapping = True
    #         if not overlapping:
    #             wrists.append((x_pos, y_pos, conf))

    for scale in predictions:
        x_size = 1 / scale.shape[1]
        y_size = 1 / scale.shape[2]

        for batch in range(bs):
            for keypoint_index in range(0, 11, 3):
                mask = scale[batch, :, :, keypoint_index+2] > conf_thres
                indices = torch.nonzero(mask)

                for idx in indices:
                    x, y = idx.tolist()
                    x_pos = float(x * x_size + x_size * scale[batch, x, y, keypoint_index])
                    y_pos = float(y * y_size + y_size * scale[batch, x, y, keypoint_index+1])
                    conf = float(scale[batch, x, y, keypoint_index+2])
                    overlapping = False
                    
                    for existing_keypoint in keypoints[batch][keypoint_index // 3]:
                        if get_keypoint_distance(existing_keypoint[:2], (x_pos, y_pos)) < overlap_distance:
                            overlapping = True
                            break
                    
                    if not overlapping:
                        keypoints[batch][keypoint_index // 3].append((x_pos, y_pos, conf))

    return keypoints
    
def get_keypoint_distance(kpt1, kpt2):
    """Calucate the euclidian distance between two keypoints (x1, y1), (x2, y2)"""
    return math.sqrt((kpt1[0] - kpt2[0]) ** 2 + (kpt1[1] - kpt2[1]) ** 2)

def get_elbow_from_skeleton(kpts, left_handed=False, check_confidence=False, confidence_threshold=0.5):
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

    
def get_wrist_from_skeleton(kpts, left_handed=False, check_confidence=False, confidence_threshold=0.5):
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


def plot_keypoints(im, keypoints):
    """
    Plots circles indicating elbows and wrists on an image.

    Paremeters:
        im (np.ndarray): Image to plot the keypoints on.
        keypoints (list[list[(float)]]): A list containing lists of tuples containing the x, y, conf of differnet keypoints
    """
    height, width = im.shape[:2]
    colours = [(0, 0, 255), (0, 255, 0), (180, 105, 255), (255, 0 , 0)]

    for i in range(len(keypoints)):

        for x, y, _ in keypoints[i]:
            cv2.circle(im, (int(x * width), int(y * height)), 5, colours[i], -1)


def plot_elbow_wrist(im, kpts, left_handed=False):
    """
    Plots circles indicating a elbow and a wrist on an image.

    Paremeters:
        im (np.ndarray): Image to plot the keypoints on.
        kpts (torch.tensor): Tensor of keypoints. Can be a tensor of the full yolov7-keypoints skeleton. Or any array of length 6 with values: [elbow_x, elbow_y, elbow_confidence, wrist_x, wrist_y, wrist_confidence]
        left_handed (bool): Whether to track the left arm. (Only works for the full yolov7-keypoints skeleton)
    """
    # Assume the full original skeleton keypoints were passed in
    # And extract the elbow and wrist values
    if len(kpts) > 6:
        kpts = torch.cat([get_elbow_from_skeleton(kpts, left_handed, check_confidence=True), get_wrist_from_skeleton(kpts, left_handed, check_confidence=True)], dim=0)

    palette = np.array([[255, 0, 0], [0, 255, 0]])
    radius = 5
    num_kpts = len(kpts) // 2

    # Draw the circle
    for kid in range(num_kpts):
        r, g, b = palette[kid]
        x_coord, y_coord = kpts[2 * kid], kpts[2 * kid + 1]
        #if not (x_coord % 640 == 0 or y_coord % 640 == 0):
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