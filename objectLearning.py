"""
Note: in order to get class names (to be used according to the labels the detectors output), you can use:
classes = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
For Pytorch’s Faster R-CNN and 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',  pretrained=True)
results = model(img)
classes = results.names
For TorchHub’s YOLO. Note that in both cases, you get a dictionary as a result.
"""
"""
CS3485 - Lab 5 (R-NNN Section)
Object Detection with Faster R-CNN

This section uses a pre-trained Faster R-CNN model to:
1. Perform qualitative detection on 5 shared real-world images
2. Evaluate performance on 5 bus dataset images using confidence + IoU metrics
"""

import torch
import numpy as np
import os
import cv2
import time
import glob
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pre-trained Faster R-CNN and COCO labels
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.to(device)
model.eval()
CLASSES = weights.meta["categories"]



def prepare_image(path):
    image = Image.open(path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)
    return image, image_tensor

def detect_objects(image_tensor, threshold=0.5):
    with torch.no_grad():
        output = model([image_tensor])[0]

    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()

    #Keeping only predictions above the threshold
    keep = scores >= threshold
    return boxes[keep], labels[keep], scores[keep]

def display_results(image, boxes, labels, scores):
    img_array = np.array(image).copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{CLASSES[label]}: {score:.2f}"
        cv2.putText(img_array, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
    plt.imshow(img_array)
    plt.axis('off')
    plt.show()

# ========== Part 1: Qualitative Analysis ==========

def run_real_world_analysis():
    image_paths = [
        "Media/Crows.jpg",
        "Media/Dog Park.jpg",
        "Media/Petting Zoo.jpg",
        "Media/Street.jpg",
        "Media/Cars Movie.png"
    ]

    for path in image_paths:
        try:
            img, tensor = prepare_image(path)
            boxes, labels, scores = detect_objects(tensor)
            print(f"\nResults for {path}:")
            for label, score in zip(labels, scores):
                print(f"  - {CLASSES[label]}: {score:.2f}")
            display_results(img, boxes, labels, scores)
        except Exception as e:
            print(f"Error with {path}: {e}")

# ========== Part 2: Quantitative Evaluation ==========

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area)

def build_bus_dataset(folder_path, limit=5):
    #Automatically gathering image files from the dataset directory
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
    dataset = []
    for path in image_paths[:limit]:
        dataset.append({
            'img_path': path,
            'gt_boxes': [[50, 50, 200, 200]]  # dummy box; replace with real boxes if available
        })
    return dataset

def evaluate_bus_dataset(dataset, conf_thresh=0.5, iou_thresh=0.5):
    total, passed_conf, passed_iou, total_time = 0, 0, 0, 0

    for sample in dataset:
        try:
            img, tensor = prepare_image(sample['img_path'])

            start = time.time()
            boxes, labels, scores = detect_objects(tensor, threshold=conf_thresh)
            total_time += time.time() - start

            if len(scores) > 0:
                passed_conf += 1

            found_match = False
            for gt in sample['gt_boxes']:
                for pred_box in boxes:
                    if compute_iou(gt, pred_box) >= iou_thresh:
                        found_match = True
                        break
                if found_match:
                    break
            if found_match:
                passed_iou += 1

            total += 1
        except Exception as e:
            print(f"Error processing {sample['img_path']}: {e}")

    if total == 0:
        print("No images evaluated. Check your folder path.")
        return

    print(f"\n--- Bus Dataset Evaluation ({total} images) ---")
    print(f"Confidence Accuracy: {passed_conf / total:.2f}")
    print(f"IoU Accuracy: {passed_iou / total:.2f}")
    print(f"Average Inference Time: {total_time / total:.2f} sec")


if __name__ == "__main__":
    print("=== Running Qualitative Analysis ===")
    run_real_world_analysis()

    print("\n=== Running Quantitative Evaluation ===")
    bus_data = build_bus_dataset("archive/images/images/")
    evaluate_bus_dataset(bus_data)
