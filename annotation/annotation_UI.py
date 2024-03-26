from typing import List
import tkinter as tk
import numpy as np
import cv2
import os
import csv
import image_utils

# Basic paths
UNLABELED_IMAGES = '../images/unlabeled_images/'  # Images to label, these will be deleted from this folder as they get labeled so keep a copy if needed
LABELED_IMAGES = '../images/labeled_images/'
LABELS = '../images/labels/'

# Load the images
images = os.scandir(UNLABELED_IMAGES)
display_image_name = None
cv_image = None
display_image_id = None

# Get the names and position in the file of previously labeled images
labeled_image_names = {}
with open(LABELS + "labeled_image_names.txt", "r") as f:
    for line_num, name in enumerate(f, start=0):
        name = name.strip()
        labeled_image_names[name] = line_num * 10

# Specify canvas dimensions
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080

IMAGE_SIZE = 960
CROP_SIZE = 100

# Create canvas
c = tk.Canvas(width = CANVAS_WIDTH, height = CANVAS_HEIGHT)
c.pack()

WIDTH_OFFSET = (CANVAS_WIDTH - (IMAGE_SIZE + CROP_SIZE)) // 2
HEIGHT_OFFSET = (CANVAS_HEIGHT - (IMAGE_SIZE + CROP_SIZE)) // 2

# Label positions based on the center crop (x - WIDTH_OFFSET, y - HEIGHT_OFFSET)
positions = {'elbow_right': [],
             'elbow_left': [],
             'wrist_right': [],
             'wrist_left': [],
             }

# Keep track of what type of label was added last
label_stack = []  

# Remember the number of labeled images total (previously labeled + newly labeled in this session)
num_labeled_images = 0

# Add the image + labels to the data
def add_labeled_image(labeled_image: dict):
    """
    Saves the image to labeled_images folder and the labels to the corresponding file in the labels folder.
    Both single and multiple labels are saved separately in the case that I want to experiment with a different model.

    Parameters:
        labeled_image (dict): Dictionary containing the image, elbow_positions and  wrist positions.
    """
    global num_labeled_images

    # Retreive labels from the dictionary
    image = labeled_image['image']
    positions = {"elbow_right": labeled_image["elbow_right"],
                 "wrist_right": labeled_image["wrist_right"],
                 "elbow_left": labeled_image["elbow_left"],
                 "wrist_left": labeled_image["wrist_left"],}

    # Write the image to file
    image_name = f'image{num_labeled_images}.png'
    cv2.imwrite(LABELED_IMAGES + image_name, image)
    num_labeled_images += 1

    # Save the multiple elbow and wrist annotations
    labels = image_utils.normalise_multiple_labels(positions)

    # Write the labels to file
    data = [image_name] + labels
    with open(LABELS + 'annotations_multi.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    

def get_center_crop() -> np.ndarray:
    """
    Extracts a center crop from the displayed image.

    This function calculates the correct crop of the highlighted part of the display image based on the image size.

    Note: This function relies on the global varaibles 'cv_image' and 'display_image_id'.

    Returns:
        np.ndarray: The center crop image as a NumPy array.
    """

    # Get the position, width and height of the image
    x_center, y_center = c.coords(display_image_id)
    height, width = cv_image.shape[:2]

    # Get the offset of the crop on the image
    offset_x = int(WIDTH_OFFSET - (x_center - width // 2))
    offset_y = int(HEIGHT_OFFSET - (y_center - height // 2))
    center_image = image_utils.crop_image(cv_image, start_x=offset_x, start_y=offset_y, width=IMAGE_SIZE+CROP_SIZE, height=IMAGE_SIZE+CROP_SIZE)

    return center_image


def check_labeled_images():
    """
    This function checks if an image with the same name hasn't been labeled already.

    If the an image with the same name has been labeled it asks the user whether they want to proceed.

    Returns:
        bool: A boolean value indicating if saving the image should proceed.
    """
    if display_image_name in labeled_image_names:
        line = labeled_image_names[display_image_name]
        response = input((f"The image with name '{display_image_name}' was already labeled at some point.\n"
                          f"It was labeled as image number {line // 10}. That should correspond to images image{line}.png - image{line + 9}.png.\n"
                          f"I suggest to check that this is a different image with the same name. Do you still want to proceed? Y/N: "
        ))
        
        if response.strip().lower() == 'y':
            print("Continuing normally.")
            return True
        else:
            print("Did not save this image.")
            return False
    return True


def write_image_name_to_file():
    """
    Write the original image name to a file.
    """
    if display_image_name not in labeled_image_names:
        labeled_image_names[display_image_name] = num_labeled_images
        with open(LABELS + "labeled_image_names.txt", "a") as f:
            f.write(display_image_name + '\n')
    else:  # If the user proceeds to label an image with the same name add a repeated tag at the end
        with open(LABELS + "labeled_image_names.txt", "a") as f:
            f.write(display_image_name + " (repeated)" + '\n')


def delete_display_image():
    """
    This function deletes the current dispaly image from the images/unlabeled_images folder.
    """
    image_path = UNLABELED_IMAGES + display_image_name

    try:
        # Check if the file exists
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Image {display_image_name} removed successfully.")
        else:
            print(f"Image {display_image_name} does not exist.")
    except Exception as e:
        print(f"An error occurred while removing the image: {e}")


def save_labels():
    """
    This function saves the 10-crop images and labels of the current image and displays the next one.

    Note:
        This function relies on the following global variables:
        - `positions`: Dictionary with values being lists of keypoint positionss.
        - `cv_image`: Current image being labeled.
    """
    continue_answer = check_labeled_images()

    if not continue_answer:
        return

    # Perform the crops and save the image and labels
    center_image = get_center_crop()
    labeled_images = image_utils.ten_crop(center_image, positions)
    for labeled_image in labeled_images:
        add_labeled_image(labeled_image)
    
    print("10 images and labels saved successfully!")

    write_image_name_to_file()

    delete_display_image()

    get_next_image()  # Load and display the next image
    

def draw_labels():
    """
    Draw all elbow and wrist labels on the canvas.
    """
    c.delete('elbow_right', 'wrist_right', 'elbow_left', 'wrist_left')

    labels = [
            {"positions": positions["elbow_right"], "fill": "red", "tag": "elbow_right"},
            {"positions": positions["wrist_right"], "fill": "green", "tag": "wrist_right"},
            {"positions": positions["elbow_left"], "fill": "hotpink", "tag": "elbow_left"},
            {"positions": positions["wrist_left"], "fill": "blue", "tag": "wrist_left"}
    ]

    for label in labels:
        for (x, y) in label["positions"]:
            x = x + WIDTH_OFFSET
            y = y + HEIGHT_OFFSET
            c.create_oval(x-3, y-3, x+3, y+3, fill=label["fill"], tags=label["tag"])


def reset():
    # Clears the elbow and wrist labels and deltes the elbow and wrist indicators
    global positions, label_stack
    label_stack = []
    positions = {'elbow_right': [],
             'elbow_left': [],
             'wrist_right': [],
             'wrist_left': [],
             }
    c.delete('elbow_right', 'wrist_right', 'elbow_left', 'wrist_left')
    print("All keypoints reset!")

def on_backspace(event):
    # On backspace press call undo_label
    undo_label()

def on_enter(event):
    # On backspace press call undo_label
    save_labels()

def undo_label():
    # Undos the last placed label
    global positions, label_stack
    if label_stack:
        name = label_stack.pop()
        positions[name].pop()
        draw_labels()
        print("Last keypoint removed!")
    else:
        print("There is no more labels to undo!")

def place_keypoint(event, name, colour):
    # Store the position of the elbow and highlight it on the canvas
    global positions
    x, y = event.x, event.y
    c.create_oval(x-3, y-3, x+3, y+3, fill=colour, tags=name)       
    positions[name].append((x - WIDTH_OFFSET, y - HEIGHT_OFFSET))
    label_stack.append(name)
    print("Keypoint added!")

def drag_start(event):
    """
    Initialize mouse position for dragging.

    Parameters:
        event: An event object containing the x and y positions of the initial mouse click.
    """
    global x_start, y_start
    x_start, y_start = event.x, event.y


def drag(event):
    """
    Update mouse position and move display image accordingly.

    Parameters:
        event: An event object containing the x and y positions of the mouse movement.

    Note: This function relies on a global varaible 'display_image_id'.
    """
    global x_start, y_start
    new_x, new_y = event.x, event.y
    c.move(display_image_id, new_x - x_start, new_y - y_start)
    x_start, y_start = new_x, new_y


def get_next_image():
    """
    This function reads in and dispalys the next image.
    """
    global cv_image, display_image_id, display_image, display_image_name
    print("Im getting the next image!")

    reset()  # Reset labels
    display_image_name = next(images).name
    cv_image = cv2.imread(UNLABELED_IMAGES + display_image_name)
    display_image = tk.PhotoImage(file = UNLABELED_IMAGES + display_image_name)
    c.delete("display_image_tag")
    display_image_id = c.create_image(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2, anchor='c', image=display_image, tags="display_image_tag")
    c.tag_lower("display_image_tag")
    print("Ready to label!")


def count_labeled_images():
    # Count the number of labeled images so far.
    global num_labeled_images
    with os.scandir(LABELED_IMAGES) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.png'):
                num_labeled_images += 1


def draw_on_canvas():
    # This function draws all the baseline objects on the canvas. 
    # This includes the highlighted center crop 516x516 area and the legend in the top left.
    print("Creating canvas!")

    # Highlight the center crop 
    c.create_rectangle(0, 0, CANVAS_WIDTH, HEIGHT_OFFSET, fill='black', stipple='gray75')
    c.create_rectangle(0, 0, WIDTH_OFFSET, CANVAS_HEIGHT, fill='black', stipple='gray75')
    c.create_rectangle(0, CANVAS_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT - HEIGHT_OFFSET, fill='black', stipple='gray75')
    c.create_rectangle(CANVAS_WIDTH, 0, CANVAS_WIDTH - WIDTH_OFFSET, CANVAS_HEIGHT, fill='black', stipple='gray75')

    # Draw the legend
    c.create_rectangle(0,0,85,50, fill='white')
    c.create_oval(4, 4, 10, 10, fill='red', tags='legend')
    c.create_text(48, 7, text='= elbow right', tags='legend')
    c.create_oval(4, 14, 10, 20, fill='green', tags='legend')
    c.create_text(45, 17, text='= wrist right', tags='legend')

    c.create_oval(4, 24, 10, 30, fill='hotpink', tags='legend')
    c.create_text(45, 27, text='= elbow left', tags='legend')
    c.create_oval(4, 34, 10, 40, fill='blue', tags='legend')
    c.create_text(42, 37, text='= wrist left', tags='legend')


def main():
    # Draw base objects
    draw_on_canvas()

    # Count the labeled images so far
    count_labeled_images()

    # Read in and dispaly the first image
    get_next_image()

    # Specify mouse button binds
    c.focus_set()
    c.bind('<Button-1>', lambda event: place_keypoint(event, "elbow_right", "red"))
    c.bind('<Button-3>', lambda event: place_keypoint(event, "wrist_right", "green"))
    c.bind("q", lambda event: place_keypoint(event, "elbow_left", "hotpink"))
    c.bind("w", lambda event: place_keypoint(event, "wrist_left", "blue"))

    c.bind_all('<Button-2>', drag_start)
    c.bind_all('<B2-Motion>', drag)

    # Bind the Backspace key
    c.bind_all("<BackSpace>", on_backspace)

    c.bind_all("<Return>", on_enter)

    # Create buttons
    button1 = tk.Button(text="Reset", command=reset)
    button1.pack()
    button2 = tk.Button(text="Undo", command=undo_label)
    button2.pack()
    button3 = tk.Button(text="Next", command=save_labels)
    button3.pack()

    c.mainloop()

if __name__ == "__main__":
    main()