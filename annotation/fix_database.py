# This code helps when some of the images were manually removed from the database.
# It will rename the remaining pictures to leave no gaps in the numbering of the names and adjust the labels accordingly.
# Might not work properly if you delete only some of the crops from one image, to make this work delete ALL of the 10 crops

from pathlib import Path
import os
import csv

new_images = '../images/labeled_images'
old_images = '../images/old_labeled_images'
original_name_file = '../images/labels/labeled_image_names.txt'
image_labels_file = '../images/labels/annotations_multi.csv'

def extract_index(file_name):
    return int(file_name[5:-4])

def main():
    # Read in original image names
    lines = []
    with open(original_name_file, 'r') as file:
        lines = file.readlines()
    original_image_names = [line.strip() for line in lines]

    # Read in the image labels
    image_labels = []  
    with open(image_labels_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            image_labels.append(row)

    # Empty the original name file
    with open(original_name_file, 'w') as file:
        file.truncate(0)

    # Empty the annotations file
    with open(image_labels_file, 'w', newline='') as csvfile:
        pass

    os.rename(new_images, old_images)
    os.makedirs(new_images)

    image_count = 0
    last_image_name = None
    folder_path = Path(old_images)
    file_paths = sorted(folder_path.iterdir(), key=lambda x: extract_index(x.name))  # Sort images by index

    # For each labeled image
    for file_path in file_paths:
        if file_path.is_file():
            image_index = int(file_path.name[5:-4])

            # Rename the file
            os.rename(os.path.join(old_images, file_path.name), os.path.join(new_images, f"image{image_count}.png"))

            # Write the original name to a file
            with open(original_name_file, 'a') as file:
                image_name = original_image_names[image_index // 10]
                if last_image_name != image_name:
                    file.write(image_name + '\n')
                    last_image_name = image_name

            # Open the CSV file in write mode
            with open(image_labels_file, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                image_labels[image_index][0] = f"image{image_count}.png"
                csvwriter.writerow(image_labels[image_index])

            image_count += 1

    os.rmdir(old_images)
 
if __name__ == "__main__":
    main()