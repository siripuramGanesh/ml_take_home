# app/services/dataset.py
import json
from pathlib import Path
from typing import List
from PIL import Image
import io
from app.schemas.collect import BoundingBox

NEW_DATA_DIR = Path("data/new")
(NEW_DATA_DIR / "images").mkdir(parents=True, exist_ok=True)
(NEW_DATA_DIR / "labels").mkdir(parents=True, exist_ok=True)
# We’ll also keep your JSON for auditing, but training will use YOLO .txt labels.

def _to_yolo_line(bb: BoundingBox, img_w: int, img_h: int, class_map: dict[str, int]) -> str:
    # bbox = [x_min, y_min, x_max, y_max] in pixels
    x_min, y_min, x_max, y_max = bb.bbox
    w = x_max - x_min
    h = y_max - y_min
    x_center = x_min + w / 2
    y_center = y_min + h / 2

    # normalize to 0..1
    x_c = x_center / img_w
    y_c = y_center / img_h
    ww  = w / img_w
    hh  = h / img_h

    cls_id = class_map.get(bb.label, 0)  # default 0 if unseen
    return f"{cls_id} {x_c:.6f} {y_c:.6f} {ww:.6f} {hh:.6f}"

def save_data(image_bytes: bytes, filename: str, annotations: List[BoundingBox], message: str = ""):
    """
    Save uploaded image + annotations.
    - Image -> data/new/images/<filename>
    - JSON  -> data/new/<filename>.json (audit only)
    - YOLO  -> data/new/labels/<stem>.txt (for training)
    Returns: (saved_image_path, num_annotations)
    """
    # ensure subdirs
    (NEW_DATA_DIR / "images").mkdir(parents=True, exist_ok=True)
    (NEW_DATA_DIR / "labels").mkdir(parents=True, exist_ok=True)

    # 1) Save image
    img_path = NEW_DATA_DIR / "images" / filename
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    # Get image size
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = img.size

    # 2) Save human-readable JSON (audit/debug)
    json_path = NEW_DATA_DIR / f"{Path(filename).stem}.json"
    with open(json_path, "w") as f:
        json.dump([a.dict() for a in annotations], f, indent=2)

    # 3) Save optional message
    if message.strip():
        msg_path = NEW_DATA_DIR / f"{Path(filename).stem}_msg.txt"
        with open(msg_path, "w") as f:
            f.write(message)

    # 4) Write YOLO label file
    # NOTE: simple class_map; extend with a persistent map if you have many classes
    # Example: map labels to incremental IDs by name (stable map is recommended in real projects)
    # For now, a tiny heuristic map:
    class_map = {}
    next_id = 0
    for a in annotations:
        if a.label not in class_map:
            class_map[a.label] = next_id
            next_id += 1

    yolo_label_path = NEW_DATA_DIR / "labels" / f"{Path(filename).stem}.txt"
    with open(yolo_label_path, "w") as f:
        for a in annotations:
            f.write(_to_yolo_line(a, img_w, img_h, class_map) + "\n")

    return str(img_path), len(annotations)
