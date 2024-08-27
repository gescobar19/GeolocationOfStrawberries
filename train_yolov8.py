from ultralytics import YOLO

def train_model():
    # Define the path to your dataset YAML file
    dataset_path = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\MixedRealFake\data.yaml'
    
    # Load a pretrained YOLOv8 model or any model you have
    model = YOLO(r'C:\Users\EEsco\Downloads\ultralytics-main(3)\runs\detect\train3\weights\best.pt')  # Replace with the path to your model if needed
    
    # Train the model
    model.train(data=dataset_path, imgsz=640, epochs=100, device='0', lr0=0.0001)  # Use GPUs 0 and 1

if __name__ == '__main__':
    train_model()
