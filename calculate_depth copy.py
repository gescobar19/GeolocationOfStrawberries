import os
import cv2
import torch
import numpy as np
from torchvision.transforms import Compose
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO

# Function to detect strawberries and estimate depth
def detect_and_estimate_depth(yolo_model, depth_model, image_path):
    # Load YOLOv8 model
    yolo_model = YOLO("yolov8n.pt")  # Load your YOLOv8 model path here
    yolo_model.eval()

    # Load DepthAnythingV2 model
    depth_model = DepthAnythingV2(encoder='vitl')  # Adjust parameters based on your DepthAnythingV2 setup
    depth_model.load_state_dict(torch.load(r"C:\Users\EEsco\Downloads\ultralytics-main(3)\depth_anything_v2_vitb.pth", map_location='cpu'))
    depth_model.eval()

    # Load image using OpenCV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).float() / 255.0

    # Perform detection with YOLOv8
    results = yolo_model(image_tensor.unsqueeze(0))

    # Extract bounding boxes and crop detected strawberries
    for detection in results.pred:
        class_id, confidence, box = detection['class'], detection['conf'], detection['bbox']
        x_min, y_min, x_max, y_max = map(int, box)

        # Crop detected region (strawberry) from the original image
        strawberry_region = image[y_min:y_max, x_min:x_max]

        # Perform depth estimation using DepthAnythingV2
        depth_map = depth_model.infer_image(strawberry_region)

        # Calculate average depth (you may need to adjust this based on DepthAnythingV2 output)
        average_depth = np.mean(depth_map)

        # Draw bounding box and depth information on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f"Depth: {average_depth:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# Function to process a folder of images
def process_folder(yolo_model, depth_model, folder_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        output_image = detect_and_estimate_depth(yolo_model, depth_model, image_path)
        output_file = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_depth.jpg")
        cv2.imwrite(output_file, output_image)
        print(f"Processed: {image_file} -> {output_file}")

# Example usage
if __name__ == '__main__':
    folder_path = r"C:\Users\EEsco\Downloads\ultralytics-main(3)\Depth-Strawberries.v1i.yolov8\test\images"
    output_dir = r"C:\Users\EEsco\Downloads\ultralytics-main(3)\depthMaps"

    yolo_model = YOLO(r'C:\Users\EEsco\Downloads\ultralytics-main(3)\runs\detect\train6\weights\best.pt')  # Assuming yolov8n.pt is in the current directory or specified path
    depth_model = DepthAnythingV2()  # Assuming depth_anything_v2.pth is in the current directory or specified path

    process_folder(yolo_model, depth_model, folder_path, output_dir)
