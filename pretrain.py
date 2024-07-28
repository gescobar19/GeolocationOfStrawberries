from ultralytics import YOLO # Import your YOLOv8 model class

def pretrain_model():
    # Define the path to your pre-training dataset YAML file
    pretrain_dataset_path = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\Strawberry-Detection.v9i.yolov8\data.yaml'
    
    # Load a pretrained YOLOv8 model or any model you have
    model = YOLO('yolov8n.pt')  # Replace with the path to your pretrained model if needed
    
    # Train the model on the pre-training dataset
    model.train(data=pretrain_dataset_path, imgsz=640, epochs=50, device='0')  # Use GPU 0

if __name__ == '__main__':
    pretrain_model()
