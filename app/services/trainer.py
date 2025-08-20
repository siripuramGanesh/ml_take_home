# app/services/trainer.py
import asyncio
from pathlib import Path
from ultralytics import YOLO
import random
import tempfile

COCO_IMG = Path("data/coco/images")
COCO_LBL = Path("data/coco/labels")
NEW_IMG  = Path("data/new/images")
NEW_LBL  = Path("data/new/labels")
MODEL_PATH = Path("models/current.pt")

# Load model (if starting from a base model, point to yolov8n.pt or similar on first run)
model = YOLO(str(MODEL_PATH)) if MODEL_PATH.exists() else YOLO("yolov8n.pt")

CO_TRAINING_RATIOS = [
    (0.10, 0.90),
    (0.25, 0.75),
    (0.50, 0.50),
    (0.75, 0.25),
    (0.90, 0.10),
]

def _pair_images_with_labels(img_dir: Path, lbl_dir: Path):
    imgs = []
    for img_path in img_dir.glob("*.*"):
        stem = img_path.stem
        lbl = lbl_dir / f"{stem}.txt"
        if lbl.exists():
            imgs.append(str(img_path))
    return imgs

def _sample_mixed_paths(new_ratio: float, coco_ratio: float):
    new_imgs  = _pair_images_with_labels(NEW_IMG, NEW_LBL)
    coco_imgs = _pair_images_with_labels(COCO_IMG, COCO_LBL)

    if not new_imgs and not coco_imgs:
        return []

    # Number of samples per pool (proportional; at least 1 if pool not empty)
    n_new  = max(1, int(len(new_imgs)  * new_ratio))  if new_imgs  else 0
    n_coco = max(1, int(len(coco_imgs) * coco_ratio)) if coco_imgs else 0

    sampled = []
    if new_imgs:
        sampled += random.sample(new_imgs, min(n_new, len(new_imgs)))
    if coco_imgs:
        sampled += random.sample(coco_imgs, min(n_coco, len(coco_imgs)))
    random.shuffle(sampled)
    return sampled

async def continuous_training():
    """
    Background loop:
      - Build a mixed train list using the current (new:COCO) ratio
      - Train 1-2 epochs with strong augmentation + cosine (cyclical) LR
      - Save weights back to models/current.pt
    """
    while True:
        for new_ratio, coco_ratio in CO_TRAINING_RATIOS:
            train_paths = _sample_mixed_paths(new_ratio, coco_ratio)
            if not train_paths:
                print("Trainer: No data found yet. Sleeping…")
                await asyncio.sleep(10)
                continue

            print(f"Trainer: {int(new_ratio*100)}% new / {int(coco_ratio*100)}% COCO on {len(train_paths)} images")

            # Ultralytics accepts a data dict with a list for 'train'
            data_cfg = {
                "train": train_paths,
                # (Optional) you can add a small val set; for simplicity, reuse a subset
                "val": train_paths[:max(1, len(train_paths)//5)],
                # You can add 'nc' and 'names' if you maintain a stable class map
            }

            # Strong but reasonable augmentations + cosine LR (cyclical)
            model.train(
                data=data_cfg,
                epochs=1,           # keep tight for background loop; increase later
                imgsz=640,
                batch=8,
                workers=2,          # bump if you have CPU headroom
                cos_lr=True,        # cosine schedule ~ cyclical
                lr0=0.01,           # initial LR
                lrf=0.01,           # final LR multiplier for cosine
                degrees=10.0,
                translate=0.1,
                scale=0.5,          # up to 1.5x
                shear=2.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,         # enable mosaic
                mixup=0.1,          # slight mixup
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                close_mosaic=3,     # turn mosaic off last N epochs (here small since epochs=1)
                exist_ok=True,
                verbose=False,
            )

            # Save/overwrite the active model
            # (Ultralytics keeps runs/, but we explicitly export best to our path)
            model.save(str(MODEL_PATH))
            print(f"Trainer: saved updated model -> {MODEL_PATH}")

            # brief pause before next ratio
            await asyncio.sleep(5)

        # Longer pause after a full schedule cycle
        await asyncio.sleep(30)
