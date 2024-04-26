from typing import List, Tuple, Dict
import numpy as np
import cv2

def compile_to_dict(image: np.ndarray, positions: Dict[str, List[Tuple[int, int]]]) -> Dict:
    """
    Compiles dictionary including the image with its elbow and wrist position labels and checks if they are out of bounds.

    Parameters:
        image (np.ndarray): The image to include in the dictionary.
        positions ({str: [(x,y)]}): A dictionary containing lists with the keypoint positions.

    Returns:
        Dict: The dictionary including the image with its labels.

    Example:
        The returned dictionary has the following structure:
        {
            'image': np.ndarray,  # The input image
            'elbow_right': [],    # List to store right arm elbow positions
            'wrist_right': []     # List to store right arm wrist positions
            'elbow_left': [],     # List to store left arm wrist positions
            'wrist_left': []      # List to store left arm wrist positions
        }
    """

    dic = {'elbow_right': [],
           'elbow_left': [],
           'wrist_right': [],
           'wrist_left': [],
    }
    
    # Save image 
    dic['image'] = image
    
    height, width = image.shape[:2]

    for name, values in positions.items():
        for x,y in values:
            if x >= 0 and x < width and y >= 0 and y < height:  # Check for label out of bounds after crop 
                dic[name].append((x,y))

    return dic


def ten_crop(image: np.ndarray, positions: Dict[str, List[Tuple[int, int]]], crop_size: int = 960, crop_amount: int = 100) -> List[Dict]:
    """
    Performs the ten crop algorithm. Takes crops from all the corners + the center crop. It also horizontally flips them.

    Parameters:
        image (np.ndarray): The image to crop.
        positions ({str: [(x,y)]}): A dictionary containing lists with the keypoint positions.

    Returns:
        List[Dict]: Returns a list containing the 10 crops.
            The crops are in a form of dictionaries that include
            the cropped image and the revelant elbow/wrist positions.
    """

    labeled_images = []
    
    c = crop_amount
    # (x, y) start of crops
    crop_starts = [
        (0, 0),  # Top-left 
        (0, c),  # Bottom-left
        (c, 0),  # Top-right
        (c, c),  # Bottom-right
        (c//2, c//2)  # Center
    ]

    # Check if padding will be needed and if so, ask user if they want to proceed
    padding = False
    x_pad = y_pad = 0
    image_height, image_width = image.shape[:2]
    if image_height < crop_size + crop_amount or image_width < crop_size + crop_amount:
        print(f"The height or width of the crop is out of bounds for the image.")
        user_input = input("Do you want to pad the image and continue? (Y/N): ")
        if user_input.lower() == 'y' or user_input.lower() == 'yes':
            padding = True
            x_pad = (crop_size + crop_amount - image_width) // 2
            y_pad = (crop_size + crop_amount - image_height) // 2
        else:
            return "User cancel"

    #print(f"Pad: {x_pad},  {y_pad}")

    # Do the 5 crops
    for (x_start, y_start) in crop_starts:
        # Crop the image
        cropped_image = image[y_start:image_height-(crop_amount-y_start), x_start:image_width-(crop_amount-x_start)]
        # Pad the image
        if padding:
            cropped_image = pad_image(cropped_image, 960)

        # for x, y in positions["elbow_left"]:
        #     print(x, y)
        #     print([x - x_start > x_pad, x - x_start < x_pad + image_width, y - y_start > y_pad, y - y_start < y_pad + image_height])

        # Get the new keypoint positions after the crop
        cropped_positions = {key: [(x - x_start, y - y_start) for (x, y) in value  # Shift the keypoint based on the crop
                                   if x - x_start > x_pad and x - x_start < x_pad + image_width - crop_amount # Make sure the keypoint is in bounds after crop
                                   and y - y_start > y_pad and y - y_start < y_pad + image_height - crop_amount]
                            for key, value in positions.items()}  # For each keypoint type

        dic = compile_to_dict(cropped_image, cropped_positions)
        labeled_images.append(dic)

        # Flip each image + elbow and wrist positions horizontally
        flipped_image = cv2.flip(cropped_image, 1)  # Flip image horizontally
        # Get the new elbow positions after the flip
        left_elbows = [(crop_size - 1 - x, y) for (x, y) in cropped_positions["elbow_right"]]
        left_wrists = [(crop_size - 1 - x, y) for (x, y) in cropped_positions["wrist_right"]]
        right_elbows = [(crop_size - 1 - x, y) for (x, y) in cropped_positions["elbow_left"]]
        right_wrists = [(crop_size - 1 - x, y) for (x, y) in cropped_positions["wrist_left"]]
        flipped_positions = {"elbow_right": right_elbows,
                             "wrist_right": right_wrists,
                             "elbow_left": left_elbows,
                             "wrist_left": left_wrists} 
        dic = compile_to_dict(flipped_image, flipped_positions)
        labeled_images.append(dic)

    return labeled_images


def normalise_single_labels(elbow_pos: List[Tuple[int, int]], wrist_pos: List[Tuple[int, int]], image_width: int = 960, image_height: int = 960) -> List[int]:
    """
    Takes the first elbow and wrist position and normalises it.

    Noramlised x and y are in range (0,1) and represent the percentage of width/height of the image rounded to 3 decimal places.

    Parameters:
        elbow_pos [(x, y)]: List of elbow (x, y) positions in pixel coordinates.
        wrist_pos [(x, y)]: List of wrist (x, y) positions in pixel coordinates.
        image_width (int): The pixel width of the image.
        image_height (int): The pixel height of the image.
    
    Returns:
        List[int]: List containing: 
            : Normalized x and y coordinates of the elbow if present 
            : Whether an elbow is present (1 for True, 0 for False)
            : Normalized x and y coordinates of the wrist if present
            : Whether a wrist is present (1 for True, 0 for False)
        
    Example:
        >>> elbow_pos = [(100, 250), (200, 500)]
        >>> wrist_pos = []
        >>> normalise_single_labels(elbow_pos, wrist_pos)
        >>> [0.104, 0.260, 1, 0, 0, 0]
    """

    if elbow_pos == []:
        data = [0, 0, 0]
    else:
        normalised_elbow_x = round(elbow_pos[0][0] / image_width, 3)
        normalised_elbow_y = round(elbow_pos[0][1] / image_height, 3)
        data = [normalised_elbow_x, normalised_elbow_y, 1]

    if wrist_pos == []:
        data += [0, 0, 0]
    else:
        normalised_wrist_x = round(wrist_pos[0][0] / image_width, 3)
        normalised_wrist_y = round(wrist_pos[0][1] / image_height, 3)
        data += [normalised_wrist_x, normalised_wrist_y, 1]

    return data


def normalise_multiple_labels(positions: Dict[str, List[Tuple[int, int]]], image_width: int = 960, image_height: int = 960):
    """
    Generates normalized label data for multiple elbow and wrist positions rounded to 3 decimal places.

    Parameters:
        positions ({str: [(x,y)]}): A dictionary containing lists with the keypoint positions.
        image_width (int): The pixel width of the image.
        image_height (int): The pixel height of the image.

    Returns:
        list: A list of space separated keypoint coordinates. Each list entry is for different keypoint.
    """

    return_list = []

    for values in positions.values():
        coords = []
        for (x, y) in values:
            coords.append(str(round(x / image_width, 3)))
            coords.append(str(round(y / image_height, 3)))
        return_list.append(' '.join(coords))

    return return_list

def pad_image(image, size=960, colour=(114,114,114)):
    """
    Pads image to a square of a speficied size if the height and or with are less than the size.

    Parameters:
        image (np.ndarray): Image to pad.
        size (int): The size of the padded image.
        colour (int, int, int): RGB of the padding.

    Returns:
        np.ndarray: The padded image.
    """
    height, width = image.shape[:2]  # current shape [height, width]

    dh = (max(0, size - height)) / 2
    dw = (max(0, size - width)) / 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=colour)

    return padded_image