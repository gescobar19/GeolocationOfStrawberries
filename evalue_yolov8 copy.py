from ultralytics import YOLO

if __name__ == '__main__':
    # Load the trained YOLOv8 model
    model_path = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\Depth-Anything-V2\metric_depth\runs\detect\train4\weights\best.pt'
    model = YOLO(model_path)

    # Path to the single image you want to detect objects in
    image_path = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\StrawberryFixedCam4.v2i.yolov8(2)\valid\images\WIN_20240619_14_31_18_Pro_row-1-column-2_jpg.rf.42ea69c68e6a3b2f000d0eaf9912aea5.jpg'  # Replace with your image path

    # Perform detection on the single image
    results = model(image_path)

    # Print the detected results (optional)
    print(results.pandas().xyxy[0])  # Assuming only one batch, get detections dataframe

    # You can also save the results to a file if needed
    results_file = 'detection_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Detections for image {image_path}:\n")
        for _, detection in results.pandas().xyxy[0].iterrows():
            f.write(f"Class: {int(detection['class'])}, Confidence: {float(detection['confidence']):.4f}, "
                    f"Bounding Box: {detection[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()}\n")

    print(f"Detections saved to {results_file}")
