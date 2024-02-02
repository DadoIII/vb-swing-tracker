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

def ten_crop(img: tk.PhotoImage, elbow_pos: (int, int), wrist_pos: (int, int)):
    labeled_images = []

    # (x, y) start of crops
    crop_starts = [
        (0, 0),  # Top-left 
        (0, 100),  #Bottom-left
        (100, 0),  #Top-right
        (100, 100),  #Bottom-right
        (50, 50)  #Center
    ]

    for (x, y) in crop_starts:  #TODO make a horizontal flip for each of the crops
        cropped_image = crop_image(img, x, y, 416, 416)
        dic = {}
        # Save image 
        dic['image'] = cropped_image
        
        # Save eblow position
        if elbow_pos == None: 
            dic['has_elbow'] = 0
            dic['elbow_pos'] = (0, 0)
        else:
            dic['has_elbow'] = int(all([elbow_pos[0] - x < 416,  # Check for elbow out of bounds after crop
                                        elbow_pos[0] - x >= 0, 
                                        elbow_pos[1] - y < 416, 
                                        elbow_pos[1] - y >= 0]))
            dic['elbow_pos'] = (elbow_pos[0] - x, elbow_pos[1] - y)
        
        # Save wrist position
        if wrist_pos == None:
            dic['has_wrist'] = 0
            dic['wrist_pos'] = (0, 0)
        else:
            dic['has_wrist'] = int(all([wrist_pos[0] - x < 416,  # Check for wrist out of bounds after crop
                                        wrist_pos[0] - x >= 0, 
                                        wrist_pos[1] - y < 416, 
                                        wrist_pos[1] - y >= 0]))
            dic['wrist_pos'] = (wrist_pos[0] - x, wrist_pos[1] - y)

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