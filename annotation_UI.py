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
elbow_pos = None
wrist_pos = None

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


# Method to label the position of the elbow
def elbow(event):
    global elbow_pos
    x, y = event.x, event.y
    c.delete('elbow')
    c.create_oval(x-3, y-3, x+3, y+3, fill='red', tags='elbow')
    elbow_pos = (x - WIDTH_OFFSET, y - HEIGHT_OFFSET)


# Method to label the position of the wrist
def wrist(event):
    global wrist_pos
    x, y = event.x, event.y
    c.delete('wrist')
    c.create_oval(x-3, y-3, x+3, y+3, fill='green', tags='wrist')
    wrist_pos = (x - WIDTH_OFFSET, y - HEIGHT_OFFSET)


# Add the image + labels to the data
def add_labeled_image(labeled_image: dict):
    global next_labeled_index

    # Retreive labels from the dictionary
    image = labeled_image['image']
    has_elbow = labeled_image['has_elbow']
    elbow_pos = labeled_image['elbow_pos']
    has_wrist = labeled_image['has_wrist']
    wrist_pos = labeled_image['wrist_pos']

    # Format and normalise labels
    image_name = f'image{next_labeled_index}.png'
    normalised_elbow_x = elbow_pos[0] / 416
    normalised_elbow_y = elbow_pos[1] / 416
    normalised_wrist_x = wrist_pos[0] / 416
    normalised_wrist_y = wrist_pos[1] / 416

    # Write the image to file
    image.write(LABELED_IMAGES + image_name, format='png')

    # Write the labels to file
    data = [image_name, has_elbow,  normalised_elbow_x, normalised_elbow_y, has_wrist, normalised_wrist_x, normalised_wrist_y]
    with open('annotations.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    next_labeled_index += 1


# Creates a new image taken from the center 516x516 square
def get_center_crop():
    x_center, y_center = c.coords(display_image_id)
    bbox = c.bbox(display_image_id)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
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
    elbow_pos, wrist_pos = None, None
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