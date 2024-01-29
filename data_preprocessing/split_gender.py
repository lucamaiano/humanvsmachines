import os
import re
import shutil
import random

# Source folder containing the images(change if necessary)
source_folder = "/RealFaces_w_StableDiffusion/datasets/png_images/train/fake/fake_new"

# Destination folder for even-numbered images(change if necessary)
destination_folder = "/RealFaces_w_StableDiffusion/test_southAmerica/test/fake/fake_new"

# Path to the prompts.txt file(change if necessary)
prompts_file_path = '/RealFaces_w_StableDiffusion/prompts.txt'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Read prompts.txt and find the line with "women"
women_line_number = []
with open(prompts_file_path, 'r') as prompts_file:
    for line_number, line in enumerate(prompts_file, start=1):
        if 'hispanic' in line:
            women_line_number.append(line_number)


for line in women_line_number:
    # Traverse the source folder
    for root, _, files in os.walk(source_folder):
        for file in files:
            parts = file.split("-")
            if len(parts) > 1 and parts[0].isdigit():
                image_number = int(parts[0])
                #print(image_number)
                
                if image_number in women_line_number:
                    images = 10
                    for img in range(images):
                        source_file_path = os.path.join(root, file)
                        destination_file_path = os.path.join(destination_folder, file)
                        for img in range(images):
                            # Move the file to the destination folder
                            shutil.copy(source_file_path, destination_file_path)
                            print(f"Copied {file} to {destination_folder}")
else: 
    print('Shit it did not work')
    



# # Traverse the source folder
# for root, _, files in os.walk(source_folder):
#     for file in files:
#         # Split the filename by dash and get the first part
#         parts = file.split("-")
#         if len(parts) > 1 and parts[0].isdigit():
#             first_integer = int(parts[0])
#             if first_integer % 2 == 0:
#                 source_file_path = os.path.join(root, file)
#                 destination_file_path = os.path.join(destination_folder, file)
                
#                 # Move the file to the destination folder
#                 shutil.move(source_file_path, destination_file_path)
#                 print(f"Moved {file} to {destination_folder}")