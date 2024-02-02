from tkinter import *
import numpy as np
c = Canvas(width = 800, height = 600)
c.pack()


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


img = PhotoImage(file = "./almost.png")
c.create_image(0,0, anchor='nw', image=img)


def horizontal_flip(img):
    width = img.width()
    height = img.height()
    flipped_img = PhotoImage(width = width, height = height)

    for y in range(height):
        for x in range(width):
            flipped_img.put(rgb_to_hex(img.get(x, y)), (width - x, y))
    return flipped_img

flipped_img = horizontal_flip(img)
#c.create_image(0, 250, anchor='nw', image=flipped_img)


def average_rgb(rgbs):
    values = {'r': [],
            'g': [],
            'b': []}
    for r, g, b in rgbs:
        values['r'].append(r)
        values['g'].append(g)
        values['b'].append(b)

    return (int(np.mean(values['r'])), int(np.mean(values['g'])), int(np.mean(values['b'])))


def downscale(img, new_width, new_height):
    width = img.width()
    height = img.height()
    temp_img = PhotoImage(width = new_width, height = height)
    
    group_size = width // new_width  # How many pixels to average each step
    extra_pixels = width % new_width  # How many extra pixels are left with this group sizing
    if extra_pixels > 0:
        extra_avg = new_width / extra_pixels  # How often do we need to average +1 extra pixel


    print(f"Grup size: {group_size}, extra pixels: {extra_pixels}")

    for y in range(height):
        i = 0
        count = 0
        while i < width:
            pixels = []  # List of pixel values to average
            for j in range(group_size):
                pixels.append(img.get(i+j, y))
            i += group_size

            if extra_pixels > 0 and (count + 1) % extra_avg <= 1:  # Average extra pixel 
                pixels.append(img.get(i+group_size, y))
                i += 1

            avg_pixel = rgb_to_hex(average_rgb(pixels))
            temp_img.put(avg_pixel, (count, y))
            count += 1


    downscaled_img = PhotoImage(width = new_width, height = new_height)
    
    group_size = height // new_height  # How many pixels to average each step
    extra_pixels = height % new_height  # How many extra pixels are left with this group sizing
    if extra_pixels > 0:
        extra_avg = new_height / extra_pixels  # How often do we need to average +1 extra pixel

    for x in range(new_width):
        i = 0
        count = 0
        while i < height:
            pixels = []  # List of pixel values to average
            for j in range(group_size):
                pixels.append(temp_img.get(x, i+j))
            i += group_size

            if extra_pixels > 0 and (count + 1) % extra_avg <= 1:  # Average extra pixel 
                pixels.append(temp_img.get(x, i+group_size))
                i += 1
            
            avg_pixel = rgb_to_hex(average_rgb(pixels))
            downscaled_img.put(avg_pixel, (x, count))
            count += 1

    return downscaled_img

small_img = downscale(img, 286, 103)
c.create_image(0, 250, anchor='nw', image=small_img)

small_img2 = downscale(img, 300, 120)
c.create_image(0, 400, anchor='nw', image=small_img2)

c.mainloop()