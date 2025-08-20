from ultralytics import YOLO
from PIL import Image
import io

# Load YOLO model (replace with your model path)
model = YOLO("models/current.pt")

def run_inference(image_bytes: bytes):
    """
    Run YOLO inference on the uploaded image bytes
    and return detections as a list of dicts.
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run YOLO prediction
    results = model.predict(image)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # bounding box
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            detections.append({
                "bbox": [x1.item(), y1.item(), x2.item(), y2.item()],
                "label": label,
                "confidence": conf
            })

    return detections
