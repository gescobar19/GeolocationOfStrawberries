from ultralytics import YOLO  # Import your YOLOv8 model class

def pretrain_model():
    # Define the path to your pre-training dataset YAML file
    pretrain_dataset_path = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\Strawberry-Detection.v9i.yolov8\data.yaml'
    
    # Load a pretrained YOLOv8 model or any model you have
    model = YOLO('yolov8n.pt')  # Replace with the path to your pretrained model if needed
    
    # Train the model on the pre-training dataset
    model.train(data=pretrain_dataset_path, imgsz=640, epochs=50, device='0')  # Use GPU 0

def fine_tune_model():
    # Define the path to your fine-tuning dataset YAML file
    #fine_tune_dataset_path = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\StrawberryFixedCam4.v2i.yolov8(2)\data.yaml'

    # Define the path to the mixed dataset YAML file
    mixed_dataset_path = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\Strawberry-Mixed-Dataset\data.yaml'

    # Create the mixed dataset YAML file by combining pre-training and fine-tuning datasets
    #create_mixed_dataset(pretrain_dataset_path, fine_tune_dataset_path, mixed_dataset_path)

    # Load the pretrained model
    model = YOLO(r'C:\Users\EEsco\Downloads\ultralytics-main(3)\runs\detect\train\weights\best.pt')
    
    # Train the model on the mixed dataset
    model.train(data=mixed_dataset_path, imgsz=640, epochs=50, device='0')  # Use GPU 0

def create_mixed_dataset(pretrain_dataset_path, fine_tune_dataset_path, mixed_dataset_path):
    # This function should create a YAML file that combines the two datasets
    # For simplicity, let's assume it combines the datasets by including paths from both

    with open(pretrain_dataset_path, 'r') as file:
        pretrain_data = file.read()

    with open(fine_tune_dataset_path, 'r') as file:
        fine_tune_data = file.read()

    mixed_data = pretrain_data + "\n" + fine_tune_data

    with open(mixed_dataset_path, 'w') as file:
        file.write(mixed_data)

if __name__ == '__main__':
    pretrain_dataset_path = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\Strawberry-Detection.v9i.yolov8\data.yaml'
    #pretrain_model()
    fine_tune_model()
