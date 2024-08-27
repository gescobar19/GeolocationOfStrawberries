from ultralytics import YOLO

if __name__ == '__main__':
    # Load the trained YOLOv8 model
    model = YOLO(r'C:\Users\EEsco\Downloads\ultralytics-main(3)\Depth-Anything-V2\metric_depth\runs\detect\train4\weights\best.pt')

    # Evaluate the model on the validation set
    results = model.val(data= r'C:\Users\EEsco\Downloads\ultralytics-main(3)\StrawberryFixedCam4.v2i.yolov8(2)\data.yaml')

    # Define the path to save the results
    results_file = 'evaluation_results5.txt'

    # Open the file in write mode and save the results
    with open(results_file, 'w') as f:
        # Write overall metrics
        f.write(f"Precision: {results.box.mp:.4f}\n")
        f.write(f"Recall: {results.box.mr:.4f}\n")
        f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")

    print(f"Evaluation results saved to {results_file}")