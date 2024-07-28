import os
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

def get_ground_truth(label_file, img_width, img_height):
    if not os.path.exists(label_file):
        print(f"Label file not found: {label_file}")
        return np.array([])

    try:
        labels = np.loadtxt(label_file)
    except Exception as e:
        print(f"Error reading label file {label_file}: {e}")
        return np.array([])

    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=0)

    ground_truth_boxes = []
    for label in labels:
        if len(label) != 5:
            print(f"Skipping invalid label: {label}")
            continue

        class_id, x_center, y_center, width, height = label
        if class_id != 0:  # Changed to class_id == 0 for mature strawberries
            continue

        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        ground_truth_boxes.append([x1, y1, x2, y2])

    return np.array(ground_truth_boxes)

def evaluate_detection(predicted_boxes, ground_truth_boxes, iou_threshold, scores):
    tp = 0
    fp = 0
    fn = 0

    true_labels = []
    pred_labels = []

    matched_ground_truth = np.zeros(len(ground_truth_boxes), dtype=bool)

    for pred_box in predicted_boxes:
        best_iou = 0
        best_match = -1
        for i, gt_box in enumerate(ground_truth_boxes):
            iou = bbox_overlap_ratio(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_match = i
        if best_iou >= iou_threshold:
            if not matched_ground_truth[best_match]:
                tp += 1
                matched_ground_truth[best_match] = True
                true_labels.append(1)
                pred_labels.append(1)
            else:
                fp += 1
                true_labels.append(0)
                pred_labels.append(1)
        else:
            fp += 1
            true_labels.append(0)
            pred_labels.append(1)
    fn = np.sum(~matched_ground_truth)
    true_labels.extend([0] * fn)
    pred_labels.extend([0] * fn)

    return tp, fp, fn, scores, true_labels, pred_labels

def bbox_overlap_ratio(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def plot_precision_recall(tp_array, fp_array, fn_array):
    precision_array = np.array(tp_array) / (np.array(tp_array) + np.array(fp_array))
    recall_array = np.array(tp_array) / (np.array(tp_array) + np.array(fn_array))

    plt.figure()
    plt.plot(recall_array, precision_array, '-o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

# Load the YOLOv8 model
model = YOLO(r'C:\Users\EEsco\Downloads\ultralytics-main(3)\runs\detect\train11\weights\best.pt')

# Define test image and label directories
current_dir = os.getcwd()
print(f'Current working directory: {current_dir}')

test_image_dir = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\StrawberryTestImgs\test\images'
test_label_dir = r'C:\Users\EEsco\Downloads\ultralytics-main(3)\StrawberryTestImgs\test\labels'
output_dir = 'detection_results'

Path(output_dir).mkdir(parents=True, exist_ok=True)

iou_threshold = 0.5

test_images = list(Path(test_image_dir).glob('*.jpg'))

total_tp = 0
total_fp = 0
total_fn = 0

all_tp = []
all_fp = []
all_fn = []

all_scores = []
all_true_labels = []
all_pred_labels = []

for image_path in test_images:
    image_file = str(image_path)
    label_file = str(Path(test_label_dir) / image_path.with_suffix('.txt').name)

    im = cv2.imread(image_file)
    if im is None:
        print(f'Error: Failed to read image file {image_file}')
        continue

    results = model(im)

    bboxes = []
    scores = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls.item())
            if class_id == 0:  # Only consider mature strawberries
                bboxes.append([x1, y1, x2, y2])
                scores.append(box.conf.item())
                label = f"mature {box.conf.item():.2f}"  # Set label name to "mature"
                # Define colors
                rect_color = (34, 139, 34)  # Neutral green
                text_color = (255, 255, 255)  # White text
                background_color = (0, 0, 0)  # Black background

                # Draw rectangle
                cv2.rectangle(im, (x1, y1), (x2, y2), rect_color, 2)

                # Calculate text size and position
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                text_x, text_y = x1, y1 - 10
                box_coords = ((text_x, text_y), (text_x + text_width, text_y - text_height - baseline))
                
                # Draw background rectangle for text
                cv2.rectangle(im, box_coords[0], box_coords[1], background_color, cv2.FILLED)
                # Draw text
                cv2.putText(im, label, (text_x, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)


    detected_boxes = np.array(bboxes)

    output_file_name = str(Path(output_dir) / image_path.name)
    cv2.imwrite(output_file_name, im)

    bbox_file = str(Path(output_dir) / f'{image_path.stem}_bboxes.txt')
    np.savetxt(bbox_file, detected_boxes)

    ground_truth_boxes = get_ground_truth(label_file, im.shape[1], im.shape[0])

    tp, fp, fn, scores, true_labels, pred_labels = evaluate_detection(detected_boxes, ground_truth_boxes, iou_threshold, scores)

    total_tp += tp
    total_fp += fp
    total_fn += fn

    all_tp.append(tp)
    all_fp.append(fp)
    all_fn.append(fn)

    all_scores.extend(scores)
    all_true_labels.extend(true_labels)
    all_pred_labels.extend(pred_labels)

    print(f'Image: {image_path.name}')
    print(f'True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}')

if (total_tp + total_fp) > 0:
    precision = total_tp / (total_tp + total_fp)
else:
    precision = 0.0

if (total_tp + total_fn) > 0:
    recall = total_tp / (total_tp + total_fn)
else:
    recall = 0.0

if (precision + recall) > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0.0

with open(Path(output_dir) / 'evaluation_results.txt', 'w') as file:
    file.write(f'True Positives: {total_tp}\n')
    file.write(f'False Positives: {total_fp}\n')
    file.write(f'False Negatives: {total_fn}\n')
    file.write(f'Precision: {precision:.2f}\n')
    file.write(f'Recall: {recall:.2f}\n')
    file.write(f'F1 Score: {f1_score:.2f}\n')

plot_precision_recall(all_tp, all_fp, all_fn)
plot_confusion_matrix(all_true_labels, all_pred_labels)
