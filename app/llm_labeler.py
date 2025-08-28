import torch
from ultralytics import YOLO
from typing import Dict, List
import numpy as np

# Load a lightweight pretrained model for label refinement
label_model = YOLO("yolov8n.pt")  # could be smaller for speed

def refine_labels(img: np.ndarray, boxes: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
    """
    Refine existing bounding boxes using YOLO predictions.
    """
    if not boxes:
        # If no boxes, generate initial boxes
        pred = label_model.predict(img, verbose=False)[0]
        boxes = {}
        for b in pred.boxes:
            cls = label_model.names[int(b.cls[0].item())]
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            boxes.setdefault(cls, []).append([x1, y1, x2, y2])
    else:
        # Optional: refine boxes using model confidence
        pred = label_model.predict(img, verbose=False)[0]
        for b in pred.boxes:
            cls = label_model.names[int(b.cls[0].item())]
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            # add boxes if new object detected
            boxes.setdefault(cls, []).append([x1, y1, x2, y2])
    return boxes
