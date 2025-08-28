from pathlib import Path
from typing import List, Dict
import json
import random
import cv2
import numpy as np
import threading
from torch.utils.data import Dataset
from timm.data import create_transform
from PIL import Image

DATA_DIR = Path("data")
NEW_DIR = DATA_DIR / "new"
COCO_DIR = DATA_DIR / "coco"

(NEW_DIR / "images").mkdir(parents=True, exist_ok=True)
(NEW_DIR / "labels").mkdir(parents=True, exist_ok=True)

NEW_INDEX = NEW_DIR / "index.json"
INDEX_LOCK = threading.Lock()

# -----------------------------
# Image decoding
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
# Save new sample
# -----------------------------
def save_new_sample(img: np.ndarray, boxes: Dict[str, List[List[int]]], message: str) -> dict:
    import time

    sid = str(int(time.time() * 1000))
    img_path = NEW_DIR / "images" / f"{sid}.png"
    lbl_path = NEW_DIR / "labels" / f"{sid}.json"
    cv2.imwrite(str(img_path), img)

    sample = {"id": sid, "image": str(img_path), "boxes": boxes, "message": message}

    # Thread-safe index update
    with INDEX_LOCK:
        index = []
        if NEW_INDEX.exists():
            try:
                index = json.loads(NEW_INDEX.read_text())
            except Exception:
                index = []
        index.append(sample)
        NEW_INDEX.write_text(json.dumps(index, indent=2))

    # Save individual label file
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
# Mixed dataset
# -----------------------------
class MixedDataset(Dataset):
    RATIOS = [(0.1,0.9),(0.25,0.75),(0.5,0.5),(0.75,0.25),(0.9,0.1)]

    def __init__(self, step: int = 0, transform=None):
        self.step = step % len(self.RATIOS)
        self.alpha_new, self.alpha_coco = self.RATIOS[self.step]

        self.new_index = json.loads(NEW_INDEX.read_text()) if NEW_INDEX.exists() else []

        self.coco_images = sorted((COCO_DIR/"images").glob("*.jpg")) + sorted((COCO_DIR/"images").glob("*.png"))

        self.transform = transform or create_transform(
            input_size=640,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic"
        )

    def __len__(self):
        return len(self.coco_images) + len(self.new_index)

    def __getitem__(self, idx):
        if random.random() < self.alpha_new and self.new_index:
            sample = random.choice(self.new_index)
            img = cv2.imread(sample["image"])
            boxes = sample["boxes"]
        else:
            img_path = random.choice(self.coco_images)
            img = cv2.imread(str(img_path))
            boxes = {}  # placeholder for COCO labels

        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        img = augment_cv2(img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if self.transform:
            img = self.transform(img)

        return img, boxes
