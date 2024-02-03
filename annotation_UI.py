from tkinter import *
import os
import csv
import numpy as np
import image_utils

# Basic paths
UNLABELED_IMAGES = './unlabeled-images/'
LABELED_IMAGES = './labeled-images/'
LABELS = './labels/'

# Create canvas
CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 800
c = Canvas(width = CANVAS_WIDTH, height = CANVAS_HEIGHT)
c.pack()

# Highlight the center crop 
WIDTH_OFFSET = (CANVAS_WIDTH - 516) // 2
HEIGHT_OFFSET = (CANVAS_HEIGHT - 516) // 2
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

# Label positions based on the center crop (x - WIDTH_OFFSET, y - HEIGHT_OFFSET)
elbow_pos = []
wrist_pos = []

# Find out how many labeled images there are
num_labeled_images = 0
with os.scandir(LABELED_IMAGES) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.endswith('.png'):
            num_labeled_images += 1
next_labeled_index = num_labeled_images

images = os.scandir(UNLABELED_IMAGES)
display_image = PhotoImage(file = UNLABELED_IMAGES + next(images).name)
display_image_id = c.create_image(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2, anchor='c', image=display_image, tags="display_image_tag")
c.tag_lower("display_image_tag")


# Label the position of the elbow
def elbow(event):
    x, y = event.x, event.y
    c.delete('elbow')
    c.create_oval(x-3, y-3, x+3, y+3, fill='red', tags='elbow')
    elbow_pos.append((x - WIDTH_OFFSET, y - HEIGHT_OFFSET))


# Label the position of the wrist
def wrist(event):
    x, y = event.x, event.y
    c.delete('wrist')
    c.create_oval(x-3, y-3, x+3, y+3, fill='green', tags='wrist')
    wrist_pos.append((x - WIDTH_OFFSET, y - HEIGHT_OFFSET))


# Add the image + labels to the data
def add_labeled_image(labeled_image: dict):
    global next_labeled_index

    # Retreive labels from the dictionary
    image = labeled_image['image']
    elbow_pos = labeled_image['elbow_pos']
    wrist_pos = labeled_image['wrist_pos']

    # Write the image to file
    image_name = f'image{next_labeled_index}.png'
    image.write(LABELED_IMAGES + image_name, format='png')
    next_labeled_index += 1

    # ===== Save the single elbow and wrist annotations =====

    data = [image_name]

    if elbow_pos == []:
        data += [0, 0, 0]
    else:
        normalised_elbow_x = elbow_pos[0][0] / 416
        normalised_elbow_y = elbow_pos[0][1] / 416
        data += [1, normalised_elbow_x, normalised_elbow_y]

    if wrist_pos == []:
        data += [0, 0, 0]
    else:
        normalised_wrist_x = wrist_pos[0][0] / 416
        normalised_wrist_y = wrist_pos[0][1] / 416
        data += [1, normalised_wrist_x, normalised_wrist_y]

    # Write the labels to file
    with open(LABELS + 'annotations_single.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

    # ===== Save the multiple elbow and wrist annotations =====
    
    # Create empty 13 x 13 x 6 labels (13 x 13 yolo object detection cells and 6 labels each)
    labels = np.zeros((13, 13, 6))

    # Format and normalise labels
    for x, y in elbow_pos:
        # Figure out which box the label belongs to
        box_x = x // 32  
        box_y = y // 32
        print("elbow", x, y, box_x, box_y)
        # Normalise the width and height within the box
        value_x = (x % 32) / 32
        value_y = (y % 32) / 32
        labels[box_x,box_y,:3] = [1, value_x, value_y]
    
    for x, y in wrist_pos:
        # Figure out which box the label belongs to
        box_x = x // 32  
        box_y = y // 32
        print("wrist", x, y, box_x, box_y)
        # Normalise the width and height within the box
        value_x = (x % 32) / 32
        value_y = (y % 32) / 32
        labels[box_x,box_y,3:] = [1, value_x, value_y]

    # Write the labels to file
    data = [image_name] + np.ndarray.flatten(labels).tolist()
    with open(LABELS + 'annotations_multi.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


# Creates a new image taken from the center 516x516 square
def get_center_crop():
    # Get the width and height of the image
    x_center, y_center = c.coords(display_image_id)
    bbox = c.bbox(display_image_id)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # Get the offset of the crop on the image
    offset_x = int(WIDTH_OFFSET - (x_center - width // 2))
    offset_y = int(HEIGHT_OFFSET - (y_center - height // 2))
    center_image = image_utils.crop_image(display_image, start_x=offset_x, start_y=offset_y, width=516, height=516)
    return center_image


# Saves the 10 crop images and labels and displays the next image
def next_image():
    center_image = get_center_crop()
    labeled_images = image_utils.ten_crop(center_image, elbow_pos, wrist_pos)
    for labeled_image in labeled_images:
        add_labeled_image(labeled_image)
    print("Go next!")


# Deletes the elbow and wrist labels
def reset():
    global elbow_pos, wrist_pos
    elbow_pos, wrist_pos = [], []
    c.delete('elbow', 'wrist')


# Initialize mouse position
def drag_start(event):
    global x_start, y_start
    x_start, y_start = event.x, event.y


# Update mouse position and move display image
def drag(event):
    global x_start, y_start
    new_x, new_y = event.x, event.y
    c.move(display_image_id, new_x - x_start, new_y - y_start)
    x_start, y_start = new_x, new_y


c.bind('<Button-1>', elbow)
c.bind('<Button-3>', wrist)

c.bind('<Button-2>', drag_start)
c.bind('<B2-Motion>', drag)

button1 = Button(text="Reset", command=reset)
button1.pack()
button2 = Button(text="Next", command=next_image)
button2.pack()

c.mainloop()