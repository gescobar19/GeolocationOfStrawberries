import torch
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO(r'C:\Users\EEsco\Downloads\ultralytics-main(3)\Depth-Anything-V2\metric_depth\runs\detect\train4\weights\best.pt') # Replace with your model path

# Load an image
img_path = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\StrawberryFixedCam4.v2i.yolov8(2)\test\images\WIN_20240621_19_49_00_Pro_row-1-column-2_jpg.rf.95b8a3891fc2aa219f0c8766df32135c.jpg'  # Replace with your image path
img = cv2.imread(img_path)

# Perform detection
results = model(img)

# Draw bounding boxes and labels on the image
for result in results:
    for box in result.boxes:
        # Extract box coordinates and label
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf.item()
        class_name = model.names[int(box.cls.item())]
        label = f"{class_name} {conf:.2f}"

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the label
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the image with detections
output_path = 'detected_image2.jpg'  # Replace with your desired output path
cv2.imwrite(output_path, img)

print(f"Image saved with detections at {output_path}")
