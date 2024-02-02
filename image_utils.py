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

def compile_to_dict(image: tk.PhotoImage, elbow_pos: (int, int) or None, wrist_pos: (int, int) or None):
    dic = {}
    # Save image 
    dic['image'] = image
    
    # Save eblow position
    if elbow_pos == None: 
        dic['has_elbow'] = 0
        dic['elbow_pos'] = (0, 0)
    else:
        dic['has_elbow'] = int(all([elbow_pos[0] >= 0,  # Check for elbow out of bounds after crop
                                    elbow_pos[0] < image.width(), 
                                    elbow_pos[1] >= 0, 
                                    elbow_pos[1] < image.height()]))
        dic['elbow_pos'] = elbow_pos
    
    # Save wrist position
    if wrist_pos == None:
        dic['has_wrist'] = 0
        dic['wrist_pos'] = (0, 0)
    else:
        dic['has_wrist'] = int(all([wrist_pos[0] >= 0,  # Check for wrist out of bounds after crop
                                    wrist_pos[0] < image.width(), 
                                    wrist_pos[1] >= 0, 
                                    wrist_pos[1] < image.height()]))
        dic['wrist_pos'] = wrist_pos
    return dic

def ten_crop(img: tk.PhotoImage, elbow_pos: (int, int) or None, wrist_pos: (int, int) or None):
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
    for (x, y) in crop_starts:
        cropped_image = crop_image(img, x, y, 416, 416)
        if elbow_pos != None:  # Get the new elbow position after the crop
            cropped_elbow_pos = (elbow_pos[0] - x, elbow_pos[1] - y)
        if wrist_pos != None:  # Get the new wrist position after the crop
            cropped_wrist_pos = (wrist_pos[0] - x, wrist_pos[1] - y)
        dic = compile_to_dict(cropped_image, cropped_elbow_pos, cropped_wrist_pos)
        labeled_images.append(dic)

        # Flip each image + elbow and wrist positions horizontally
        if elbow_pos != None:  # Get the new elbow position after the flip
            fliped_elbow_pos = (416 - cropped_elbow_pos[0], cropped_elbow_pos[1])
        if wrist_pos != None:  # Get the new wrist position after the flip
            fliped_wrist_pos = (416 - cropped_wrist_pos[0], cropped_wrist_pos[1])
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