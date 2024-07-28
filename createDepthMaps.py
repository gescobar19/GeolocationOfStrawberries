import os
import numpy as np
from PIL import Image

def create_depth_map(image_size, annotations, min_depth, max_depth):
    depth_map = np.zeros(image_size, dtype=np.float32)
    
    for ann in annotations:
        x_center, y_center, width, height, depth = ann
        
        # Convert YOLO format to pixel coordinates
        x1 = int((x_center - width / 2) * image_size[1])
        x2 = int((x_center + width / 2) * image_size[1])
        y1 = int((y_center - height / 2) * image_size[0])
        y2 = int((y_center + height / 2) * image_size[0])
        
        # Debugging prints
        print(f"YOLO Coordinates: ({x_center}, {y_center}, {width}, {height}), Depth: {depth}")
        print(f"Pixel Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Normalize depth value to the range [0, 1] for visualization
        depth_normalized = (depth - min_depth) / (max_depth - min_depth)
        depth_normalized = np.clip(depth_normalized, 0, 1)  # Ensure depth is within [0, 1]
        
        # Debugging depth normalization
        print(f"Depth Normalized: {depth_normalized}")
        
        # Set depth values within bounding box
        depth_map[y1:y2, x1:x2] = depth_normalized
    
    return depth_map

def process_images(input_folder, annotations_folder, output_folder, min_depth, max_depth):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            annotation_path = os.path.join(annotations_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
            output_path = os.path.join(output_folder, filename.replace('.jpg', '_depth.png').replace('.png', '_depth.png').replace('.jpeg', '_depth.png'))
            
            image_size = Image.open(image_path).size
            annotations = []
            
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            x_center, y_center, width, height, depth = map(float, parts)
                            annotations.append((x_center, y_center, width, height, depth))
                            # Debugging print for each annotation
                            print(f"Annotation: {x_center}, {y_center}, {width}, {height}, {depth}")
            
            depth_map = create_depth_map(image_size, annotations, min_depth, max_depth)
            
            # Debugging information
            print(f"Processing {filename}")
            print(f"Depth Map Min: {np.min(depth_map)}, Max: {np.max(depth_map)}")
            
            # Save the depth map as an image
            depth_image = Image.fromarray((depth_map * 255).astype(np.uint8))
            depth_image.save(output_path)
            print(f"Saved depth map for {filename} to {output_path}")


# Example usage
input_folder = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\StrawberryFixedCam4Depth\train\images'
annotations_folder = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\StrawberryFixedCam4Depth\train\labels'
output_folder = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\StrawberryFixedCam4Depth\trainDepthMaps'
min_depth = 68  # minimum depth value in cm
max_depth = 73  # maximum depth value in cm

process_images(input_folder, annotations_folder, output_folder, min_depth, max_depth)
