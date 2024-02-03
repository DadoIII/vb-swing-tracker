import tkinter as tk

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def horizontal_flip(img: tk.PhotoImage):
    width = img.width()
    height = img.height()
    flipped_img = tk.PhotoImage(width = width, height = height)

    for y in range(height):
        for x in range(width):
            flipped_img.put(rgb_to_hex(img.get(x, y)), (width - x, y))
    return flipped_img

def compile_to_dict(image: tk.PhotoImage, elbow_pos: [(int, int)], wrist_pos: [(int, int)]):
    dic = {}
    dic['elbow_pos'] = []
    dic['wrist_pos'] = []
    
    # Save image 
    dic['image'] = image
    
    # Save eblow positions
    for x, y in elbow_pos:
        if x >= 0 and x < image.width() and y >= 0 and y < image.height():  # Check for elbow out of bounds after crop 
            dic['elbow_pos'].append((x,y))

    # Save wrist positions
    for x, y in wrist_pos:
        if x >= 0 and x < image.width() and y >= 0 and y < image.height():  # Check for elbow out of bounds after crop 
            dic['wrist_pos'].append((x,y))
    
    return dic

def ten_crop(img: tk.PhotoImage, elbow_pos: [(int, int)], wrist_pos: [(int, int)]):
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
        cropped_image = crop_image(img, x_start, y_start, 416, 416)

        cropped_elbow_pos = [(x - x_start, y - y_start) for x, y in elbow_pos]  # Get the new elbow positions after the crop 
        cropped_wrist_pos = [(x - x_start, y - y_start) for x, y in wrist_pos]  # Get the new wrist positions after the crop
        dic = compile_to_dict(cropped_image, cropped_elbow_pos, cropped_wrist_pos)
        labeled_images.append(dic)

        # Flip each image + elbow and wrist positions horizontally 
        fliped_elbow_pos = [(416 - x, y) for x, y in cropped_elbow_pos]  # Get the new elbow positions after the flip
        fliped_wrist_pos = [(416 - x, y) for x, y in cropped_wrist_pos]  # Get the new wrist positions after the flip
        dic = compile_to_dict(horizontal_flip(cropped_image), fliped_elbow_pos, fliped_wrist_pos)
        labeled_images.append(dic)

    return labeled_images


def crop_image(img: tk.PhotoImage, start_x , start_y, width, height):
    # Make sure the crop is in bounds
    if start_x < 0 or start_y < 0:
        raise Exception(f"Crop is out of bounds for the image. Cannot start at pixel: ({start_x}, {start_y}).")
    if start_x + width > img.width():
        raise Exception(f"Crop is out of bounds for the image. Cannot take pixel: ({start_x+width},y) from an image with width: {img.width()}.")
    if start_y + height > img.height():
        raise Exception(f"Crop is out of bounds for the image. Cannot take pixel: (x,{start_y+height}) from an image with height: {img.height()}.")
    
    cropped_image = tk.PhotoImage(width = width, height = height)
    for (i, y) in enumerate(range(start_y, start_y + height)):
        for (j, x) in enumerate(range(start_x, start_x + width)):
            cropped_image.put(rgb_to_hex(img.get(x, y)), (j, i))

    return cropped_image