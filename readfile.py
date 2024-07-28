import os

# Replace with your specific file path
file_path = 'StrawberryFixedCam4WithDepth.v2i.yolov8/train/labels/WIN_20240619_14_03_17_Pro_row-1-column-2_jpg.rf.1a08e1798bf704118cc448a06375582f.txt'

# Check if file exists and try to open it
if os.path.exists(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        print("File read successfully!")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print("File does not exist.")
