from pathlib import Path
from typing import List, Dict
import json, random, threading, cv2, numpy as np
from torch.utils.data import Dataset
from timm.data import create_transform
from PIL import Image
from .llm_labeler import refine_labels
from .llm_semantic import validate_labels_semantic
import asyncio

DATA_DIR = Path("data")
NEW_DIR = DATA_DIR / "new"
COCO_DIR = DATA_DIR / "coco"
NEW_DIR.mkdir(parents=True, exist_ok=True)
(NEW_DIR / "images").mkdir(exist_ok=True)
(NEW_DIR / "labels").mkdir(exist_ok=True)

NEW_INDEX = NEW_DIR / "index.json"
INDEX_LOCK = threading.Lock()

# -----------------------------
# COCO annotations
# -----------------------------
COCO_ANN_FILE = COCO_DIR / "annotations" / "instances_train2017.json"
with open(COCO_ANN_FILE) as f:
    coco_data = json.load(f)
img_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}
coco_labels = {}
for ann in coco_data["annotations"]:
    img_id = ann["image_id"]
    bbox = ann["bbox"]
    coco_labels.setdefault(img_id, []).append([int(v) for v in bbox])
COCO_IMAGES = sorted((COCO_DIR/"images").glob("*.*"))

# -----------------------------
# Decode image
# -----------------------------
def decode_image(image_npy_b64: str | None, image_bytes_b64: str | None) -> np.ndarray:
    import base64, io
    if image_npy_b64:
        arr = base64.b64decode(image_npy_b64)
        return np.load(io.BytesIO(arr))
    if image_bytes_b64:
        data = np.frombuffer(base64.b64decode(image_bytes_b64), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        return img
    raise ValueError("No image provided")

# -----------------------------
# Clip bounding boxes
# -----------------------------
def clip_boxes_to_image(boxes, img_shape):
    H, W = img_shape[:2]
    clipped = {}
    for cls, b_list in boxes.items():
        clipped[cls] = [
            [
                max(0, min(W, x1)),
                max(0, min(H, y1)),
                max(0, min(W, x2)),
                max(0, min(H, y2)),
            ]
            for x1, y1, x2, y2 in b_list
        ]
    return clipped

# -----------------------------
# Save new sample
# -----------------------------
def save_new_sample(img: np.ndarray, boxes: Dict[str, List[List[int]]], message: str) -> dict:
    # Refine boxes with VLM
    boxes = refine_labels(img, boxes)

    # Validate labels with LLM
    boxes = asyncio.run(validate_labels_semantic(boxes))

    # Clip boxes to image dimensions
    boxes = clip_boxes_to_image(boxes, img.shape)

    # Save image and JSON (existing logic)
    import time
    sid = str(int(time.time() * 1000))
    img_path = NEW_DIR / "images" / f"{sid}.png"
    lbl_path = NEW_DIR / "labels" / f"{sid}.json"
    cv2.imwrite(str(img_path), img)

    sample = {"id": sid, "image": str(img_path), "boxes": boxes, "message": message}

    with INDEX_LOCK:
        index = []
        if NEW_INDEX.exists():
            try:
                index = json.loads(NEW_INDEX.read_text())
            except Exception:
                index = []
        index.append(sample)
        NEW_INDEX.write_text(json.dumps(index, indent=2))

    with open(lbl_path, "w") as f:
        json.dump(sample, f)

    return sample

# -----------------------------
# CV2 augmentations
# -----------------------------
def augment_cv2(image: np.ndarray) -> np.ndarray:
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image

# -----------------------------
# Full augmentation: CV2 + TIMM
# -----------------------------
def full_augment(image: np.ndarray):
    image = augment_cv2(image)
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transform = create_transform(
        input_size=640,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic"
    )
    img = transform(img_pil)
    return img

# -----------------------------
# MixedDataset with COCO labels and strict ratio
# -----------------------------
class MixedDataset(Dataset):
    RATIOS = [(0.1,0.9),(0.25,0.75),(0.5,0.5),(0.75,0.25),(0.9,0.1)]

    def __init__(self, step=0, transform=None):
        self.step = step % len(self.RATIOS)
        self.alpha_new, self.alpha_coco = self.RATIOS[self.step]
        self.new_index = json.loads(NEW_INDEX.read_text()) if NEW_INDEX.exists() else []
        self.coco_images = COCO_IMAGES
        self.transform = transform
        # Precompute pools
        n_new = max(1, int(len(self.new_index) * self.alpha_new))
        n_coco = max(1, int(len(self.coco_images) * self.alpha_coco))
        self.new_pool = self.new_index[:n_new]
        self.coco_pool = self.coco_images[:n_coco]

    def __len__(self):
        return len(self.new_pool) + len(self.coco_pool)

    def __getitem__(self, idx):
        # Alternate sampling strictly
        if idx % 2 == 0 and self.new_pool:
            sample = self.new_pool[idx % len(self.new_pool)]
            img = cv2.imread(sample["image"])
            boxes = sample["boxes"]
            source = "new"
        else:
            img_path = self.coco_pool[idx % len(self.coco_pool)]
            img = cv2.imread(str(img_path))
            img_id = next((k for k,v in img_id_to_file.items() if v==img_path.name), None)
            boxes = {"coco": coco_labels.get(img_id, [])}
            source = "coco"

        if img is None:
            raise ValueError(f"Failed to read image: {sample if source=='new' else img_path}")

        img = full_augment(img)
        return img, boxes
