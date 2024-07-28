import os
from pathlib import Path

# Define the mapping of class indices
class_mapping = {0: 1, 1: 2, 2: 0}

# Directory containing the annotation files
annotation_dir = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\RealFieldStrawberries.v1i.yolov8(1)\valid\labels'

# Function to update class indices
def update_class_indices(annotation_path, class_mapping):
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        if class_id in class_mapping:
            parts[0] = str(class_mapping[class_id])
        new_lines.append(' '.join(parts))
    
    with open(annotation_path, 'w') as file:
        file.write('\n'.join(new_lines))

# Iterate through all annotation files in the directory
for annotation_file in Path(annotation_dir).glob('*.txt'):
    update_class_indices(annotation_file, class_mapping)

print("Class indices updated successfully.")
