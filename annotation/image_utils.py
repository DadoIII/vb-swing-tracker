from typing import List, Tuple, Dict
import numpy as np
import cv2

def compile_to_dict(image: np.ndarray, elbow_pos: List[Tuple[int, int]], wrist_pos: List[Tuple[int, int]]) -> Dict:
    """
    Compiles dictionary including the image with its elbow and wrist position labels and checks if they are out of bounds.

    Parameters:
        image (np.ndarray): The image to include in the dictionary.
        elbow_pos [(int, int)]: List of elbow positions to include in the dictionary.
        wrist_pos [(int, int)]: List of wrist positions to include in the dictionary.

    Returns:
        Dict: The dictionary including the image with its labels.

    Example:
        The returned dictionary has the following structure:
        {
            'image': np.ndarray,  # The input image
            'elbow_pos': [],      # List to store elbow positions
            'wrist_pos': []       # List to store wrist positions
        }
    """

    dic = {}
    dic['elbow_pos'] = []
    dic['wrist_pos'] = []
    
    # Save image 
    dic['image'] = image
    
    height, width = image.shape[:2]

    # Save eblow positions
    for x, y in elbow_pos:
        if x >= 0 and x < width and y >= 0 and y < height:  # Check for elbow out of bounds after crop 
            dic['elbow_pos'].append((x,y))

    # Save wrist positions
    for x, y in wrist_pos:
        if x >= 0 and x < width and y >= 0 and y < height:  # Check for elbow out of bounds after crop 
            dic['wrist_pos'].append((x,y))

    return dic


def ten_crop(image: np.ndarray, elbow_pos: List[Tuple[int, int]], wrist_pos: List[Tuple[int, int]]) -> List[Dict]:
    """
    Performs the ten crop algorithm. Takes crops from all the corners + the center crop. It also horizontally flips them.

    Parameters:
        image (np.ndarray): The image to crop.
        elbow_pos [(x,y)]: List of all the elbow positions in the image.
        wrist_pos [(x,y)]: List of all the wrist positions in the image.

    Returns:
        List[Dict]: Returns a list containing the 10 crops.
            The crops are in a form of dictionaries that include
            the cropped image and the revelant elbow/wrist positions.
    """

    labeled_images = []

    # (x, y) start of crops
    crop_starts = [
        (0, 0),  # Top-left 
        (0, 100),  # Bottom-left
        (100, 0),  # Top-right
        (100, 100),  # Bottom-right
        (50, 50)  # Center
    ]

    # Do the 5 crops
    for (x_start, y_start) in crop_starts:
        cropped_image = crop_image(image, x_start, y_start, 416, 416)
        cropped_elbow_pos = [(x - x_start, y - y_start) for (x, y) in elbow_pos]  # Get the new elbow positions after the crop 
        cropped_wrist_pos = [(x - x_start, y - y_start) for (x, y) in wrist_pos]  # Get the new wrist positions after the crop
        dic = compile_to_dict(cropped_image, cropped_elbow_pos, cropped_wrist_pos)
        labeled_images.append(dic)

        # Flip each image + elbow and wrist positions horizontally
        flipped_image = cv2.flip(cropped_image, 1)  # Flip image horizontally
        fliped_elbow_pos = [(415 - x, y) for (x, y) in cropped_elbow_pos]  # Get the new elbow positions after the flip
        fliped_wrist_pos = [(415 - x, y) for (x, y) in cropped_wrist_pos]  # Get the new wrist positions after the flip
        dic = compile_to_dict(flipped_image, fliped_elbow_pos, fliped_wrist_pos)
        labeled_images.append(dic)

    return labeled_images


def crop_image(image: np.ndarray, start_x , start_y, width, height) -> np.ndarray:
    """
    Crop a region from the input image.

    Parameters:
        image (np.ndarray): The input image to be cropped.
        start_x (int): The x-coordinate of the top-left corner of the crop region.
        start_y (int): The y-coordinate of the top-left corner of the crop region.
        width (int): The width of the crop region.
        height (int): The height of the crop region.

    Returns:
        np.ndarray: The cropped region of the input image.

    Raises:
        Exception: If the crop region is out of bounds for the input image.
    """

    # Make sure the crop is in bounds
    if start_x < 0 or start_y < 0:
        raise Exception(f"Crop is out of bounds for the image. Cannot start at pixel: ({start_x}, {start_y}).")
    if start_x + width > image.shape[1]:
        raise Exception(f"Crop is out of bounds for the image. Cannot take pixel: ({start_x+width},y) from an image with width: {image.width()}.")
    if start_y + height > image.shape[0]:
        raise Exception(f"Crop is out of bounds for the image. Cannot take pixel: (x,{start_y+height}) from an image with height: {image.height()}.")
    
    return image[start_y:start_y+height, start_x:start_x+width]


def normalise_single_labels(elbow_pos: List[Tuple[int, int]], wrist_pos: List[Tuple[int, int]]) -> List[int]:
    """
    From a list of elbow and wrist labels with the x and y positions normalised.

    Noramlised x and y are in range (0,1) and represent the percentage of width/height of the image rounded to 3 decimal places.

    Parameters:
        elbow_pos [(x, y)]: List of elbow (x, y) positions in pixel coordinates.
        wrist_pos [(x, y)]: List of wrist (x, y) positions in pixel coordinates.
    
    Returns:
        List[int]: List containing: 
            : Whether an elbow is present (1 for True, 0 for False)
            : Normalized x and y coordinates of the elbow if present 
            : Whether a wrist is present (1 for True, 0 for False)
            : Normalized x and y coordinates of the wrist if present
        
    Example:
        >>> elbow_pos = [(100, 50)]
        >>> wrist_pos = []
        >>> normalise_single_labels(elbow_pos, wrist_pos)
        >>> [1, 0.240, 0.120, 0, 0, 0]
    """

    if elbow_pos == []:
        data = [0, 0, 0]
    else:
        normalised_elbow_x = round(elbow_pos[0][0] / 416, 3)
        normalised_elbow_y = round(elbow_pos[0][1] / 416, 3)
        data = [1, normalised_elbow_x, normalised_elbow_y]

    if wrist_pos == []:
        data += [0, 0, 0]
    else:
        normalised_wrist_x = round(wrist_pos[0][0] / 416, 3)
        normalised_wrist_y = round(wrist_pos[0][1] / 416, 3)
        data += [1, normalised_wrist_x, normalised_wrist_y]

    return data


def normalise_multiple_labels(elbow_pos: List[Tuple[int, int]], wrist_pos: List[Tuple[int, int]]) -> np.ndarray:
    """
    Generates normalized label data for multiple elbow and wrist positions rounded to 3 decimal places.
    
    This function assumes the size of the image is 416x416 and uses 13x13 grid cells.
    Each grid cell contains 6 values representing the label information:
        - The first three values represent the elbow position:
            - 1 if an elbow is present, 0 otherwise
            - Normalized x-coordinate of the elbow within the cell
            - Normalized y-coordinate of the elbow within the cell
        - The last three values represent the wrist position:
            - 1 if a wrist is present, 0 otherwise
            - Normalized x-coordinate of the wrist within the cell
            - Normalized y-coordinate of the wrist within the cell

    Parameters:
        elbow_pos [(x, y)]: List of elbow (x, y) positions in pixel coordinates.
        wrist_pos [(x, y)]: List of wrist (x, y) positions in pixel coordinates.

    Returns:
        np.ndarray: Numpy array of shape (13, 13, 6) containing the normalized label data.
    """

    # Create empty 13 x 13 x 6 labels (13 x 13 yolo object detection cells and 6 labels each)
    labels = np.zeros((13, 13, 6))

    # Format and normalise labels
    for x, y in elbow_pos:
        # Figure out which box the label belongs to
        box_x = x // 32  
        box_y = y // 32
        # Normalise the width and height within the box
        value_x = round((x % 32) / 32, 3)
        value_y = round((y % 32) / 32, 3)
        labels[box_x,box_y,:3] = [1, value_x, value_y]
    
    for x, y in wrist_pos:
        # Figure out which box the label belongs to
        box_x = x // 32  
        box_y = y // 32
        # Normalise the width and height within the box
        value_x = round((x % 32) / 32, 3)
        value_y = round((y % 32) / 32, 3)
        labels[box_x,box_y,3:] = [1, value_x, value_y]

    return labels