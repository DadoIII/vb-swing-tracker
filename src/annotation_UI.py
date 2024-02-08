from typing import List
import tkinter as tk
import numpy as np
import cv2
import os
import csv
import image_utils

# Basic paths
UNLABELED_IMAGES = '../unlabeled_images/'
LABELED_IMAGES = '../labeled_images/'
LABELS = '../labels/'

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
        labeled_image_names[name] = line_num

# Specify canvas dimensions
CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 800

# Create canvas
c = tk.Canvas(width = CANVAS_WIDTH, height = CANVAS_HEIGHT)
c.pack()

WIDTH_OFFSET = (CANVAS_WIDTH - 516) // 2
HEIGHT_OFFSET = (CANVAS_HEIGHT - 516) // 2

# Label positions based on the center crop (x - WIDTH_OFFSET, y - HEIGHT_OFFSET)
elbow_pos = []
wrist_pos = []

# Remember the number of labeled images total (previously labeled + newly labeled in this session)
num_labeled_images = 0


def elbow(event):
    # Store the position of the elbow and highlight it on the canvas
    x, y = event.x, event.y
    c.delete('elbow')
    c.create_oval(x-3, y-3, x+3, y+3, fill='red', tags='elbow')
    elbow_pos.append((x - WIDTH_OFFSET, y - HEIGHT_OFFSET))


def wrist(event):
    # Store the position of the wrist and highlight it on the canvas
    x, y = event.x, event.y
    c.delete('wrist')
    c.create_oval(x-3, y-3, x+3, y+3, fill='green', tags='wrist')
    wrist_pos.append((x - WIDTH_OFFSET, y - HEIGHT_OFFSET))


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
    elbow_pos = labeled_image['elbow_pos']
    wrist_pos = labeled_image['wrist_pos']

    # Write the image to file
    image_name = f'image{num_labeled_images}.png'
    cv2.imwrite(LABELED_IMAGES + image_name, image)
    num_labeled_images += 1

    # Save the single elbow and wrist annotations
    data = [image_name]
    labels = image_utils.normalise_single_labels(elbow_pos, wrist_pos)
    data += labels

    # Write the labels to file
    with open(LABELS + 'annotations_single.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

    # Save the multiple elbow and wrist annotations
    labels = image_utils.normalise_multiple_labels(elbow_pos, wrist_pos)

    # Write the labels to file
    data = [image_name] + np.ndarray.flatten(labels).tolist()
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
    center_image = image_utils.crop_image(cv_image, start_x=offset_x, start_y=offset_y, width=516, height=516)

    return center_image


def check_existing_label():
    """
    This function checks if an image with the same name hasn't been labeled already.

    If the an image with the same name has been labeled it asks the user whether they want to proceed.

    Returns:
        bool: A boolean value indicating if saving the image should proceed.
    """
    print(display_image_name, list(labeled_image_names.keys()))
    if display_image_name in labeled_image_names:
        line = labeled_image_names[display_image_name]
        response = input((f"The image with name '{display_image_name}' was already labeled at some point.\n"
                          f"It was labeled as image number {line}. That should correspond to images image{line * 10}.png - image{line * 10 + 9}.png.\n"
                          f"I suggest to check that this is a different image with the same name. Do you still want to proceed? Y/N: "
        ))
        
        if response.strip().lower() == 'y':
            print("Continuing normally.")
            return True
        else:
            print("Did not label this image.")
            return False
    return True


def write_label_to_file():
    """
    Write the original image name to a file
    """
    if display_image_name not in labeled_image_names:
        labeled_image_names[display_image_name] = num_labeled_images
        with open(LABELS + "labeled_image_names.txt", "a") as f:
            f.write(display_image_name + '\n')
    else:  # If the user proceeds to label an image with the same name add a repeated tag at the end
        with open(LABELS + "labeled_image_names.txt", "a") as f:
            f.write(display_image_name + " (repeated)" + '\n')


def save_labels():
    """
    This function saves the 10-crop images and labels of the current image and displays the next one.

    Note:
        This function relies on the following global variables:
        - `elbow_pos`: List of elbow positions.
        - `wrist_pos`: List of wrist positions.
        - `cv_image`: Current image being labeled.
    """
    continue_answer = check_existing_label()

    if not continue_answer:
        return
    
    write_label_to_file()

    # Perform the crops and save the image and labels
    center_image = get_center_crop()
    labeled_images = image_utils.ten_crop(center_image, elbow_pos, wrist_pos)
    for labeled_image in labeled_images:
        add_labeled_image(labeled_image)
    
    # TODO: Remove the image from unlabeled images

    get_next_image()  # Load and display the next image
    

def reset():
    # Clears the elbow and wrist labels and deltes the elbow and wrist indicators
    global elbow_pos, wrist_pos
    elbow_pos, wrist_pos = [], []
    c.delete('elbow', 'wrist')


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
    # Read in and dispaly the next image
    global cv_image, display_image_id, display_image, display_image_name

    reset()  # Reset labels
    display_image_name = next(images).name
    cv_image = cv2.imread(UNLABELED_IMAGES + display_image_name)
    display_image = tk.PhotoImage(file = UNLABELED_IMAGES + display_image_name)
    c.delete("display_image_tag")
    display_image_id = c.create_image(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2, anchor='c', image=display_image, tags="display_image_tag")
    c.tag_lower("display_image_tag")


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

    # Highlight the center crop 
    c.create_rectangle(0, 0, CANVAS_WIDTH, HEIGHT_OFFSET, fill='black', stipple='gray75')
    c.create_rectangle(0, 0, WIDTH_OFFSET, CANVAS_HEIGHT, fill='black', stipple='gray75')
    c.create_rectangle(0, CANVAS_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT - HEIGHT_OFFSET, fill='black', stipple='gray75')
    c.create_rectangle(CANVAS_WIDTH, 0, CANVAS_WIDTH - WIDTH_OFFSET, CANVAS_HEIGHT, fill='black', stipple='gray75')

    # Draw the legend
    c.create_rectangle(0,0,60,25, fill='white')
    c.create_oval(4, 4, 10, 10, fill='red', tags='legend')
    c.create_text(35, 7, text='= elbow', tags='legend')
    c.create_oval(4, 14, 10, 20, fill='green', tags='legend')
    c.create_text(32, 17, text='= wrist', tags='legend')


def main():
    # Draw base objects
    draw_on_canvas()

    # Count the labeled images so far
    count_labeled_images()

    # Read in and dispaly the first image
    get_next_image()

    # Specify mouse button binds
    c.bind('<Button-1>', elbow)
    c.bind('<Button-3>', wrist)

    c.bind('<Button-2>', drag_start)
    c.bind('<B2-Motion>', drag)

    # Create buttons
    button1 = tk.Button(text="Reset", command=reset)
    button1.pack()
    button2 = tk.Button(text="Next", command=save_labels)
    button2.pack()

    c.mainloop()

if __name__ == "__main__":
    main()