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


def ten_crop(image: np.ndarray, positions: Dict[str, List[Tuple[int, int]]], crop_width: int = 960, crop_height: int = 960) -> List[Dict]:
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
        cropped_image = crop_image(image, x_start, y_start, crop_width, crop_height)
        # Get the new keypoint positions after the flip
        cropped_positions = {key: [(x - x_start, y - y_start) for (x, y) in value] for key, value in positions.items()}
        dic = compile_to_dict(cropped_image, cropped_positions)
        labeled_images.append(dic)

        # Flip each image + elbow and wrist positions horizontally
        flipped_image = cv2.flip(cropped_image, 1)  # Flip image horizontally
        # Get the new elbow positions after the flip
        left_elbows = [(crop_width - 1 - x, y) for (x, y) in cropped_positions["elbow_right"]]
        left_wrists = [(crop_width - 1 - x, y) for (x, y) in cropped_positions["wrist_right"]]
        right_elbows = [(crop_width - 1 - x, y) for (x, y) in cropped_positions["elbow_left"]]
        right_wrists = [(crop_width - 1 - x, y) for (x, y) in cropped_positions["wrist_left"]]
        flipped_positions = {"elbow_right": right_elbows,
                             "wrist_right": right_wrists,
                             "elbow_left": left_elbows,
                             "wrist_left": left_wrists} 
        dic = compile_to_dict(flipped_image, flipped_positions)
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
        OR
        str: String saying that the user canceled the cropping process.

    Raises:
        Exception: If the padding of the image is not performed correctly.
    """
    # Check if the crop is in bounds
    oob = False
    if start_x < 0 or start_x + width > image.shape[1]:
        print(f"The width of the crop is out of bounds for the image.")
        oob = True
    if start_y < 0 or start_y + height > image.shape[0]:
        print(f"The height of the crop is out of bounds for the image.")
        oob = True

    if oob: # Ask to pad if the crop is out of bounds
        user_input = input("Do you want to pad the image and continue? (Y/N): ")
        if user_input.lower() == 'y' or user_input.lower() == 'yes':
            padded_image = pad_image(image[max(0, start_y):min(start_y + height, image.shape[0]), max(0, start_x):min(start_x + width, image.shape[1])])
            if padded_image.shape[:2] == (height, width):
                return padded_image
            else:
                raise Exception(f'Padded image had a size of {padded_image.shape[:2]} instead of {(height, width)}. Most likely a bug in the code.')
        else:
            return "User cancel"
    
    return image[start_y:start_y+height, start_x:start_x+width]


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

def pad_image(image, size=1060, colour=(114,114,114)):
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

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=colour)

    return image