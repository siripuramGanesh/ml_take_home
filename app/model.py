import numpy as np
import torch
from torch.optim.lr_scheduler import CyclicLR
from ultralytics import YOLO
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = CHECKPOINT_DIR / "detector.pt"

class Detector:
    def __init__(self, weights: str | None = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = YOLO(weights or "yolov8n.pt")
        self.model.to(device)
        self.device = device

        self.optimizer = torch.optim.SGD(
            self.model.model.parameters(), lr=1e-4, momentum=0.9
        )
        self.scheduler = CyclicLR(
            self.optimizer,
            base_lr=1e-5,
            max_lr=1e-2,
            step_size_up=2000,
            cycle_momentum=False
        )

        # Load checkpoint if exists
        if CHECKPOINT_FILE.exists():
            checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
            self.model.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            print("[Detector] Loaded checkpoint.")

    def save_checkpoint(self):
        torch.save({
            "model_state": self.model.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
        }, CHECKPOINT_FILE)
        print("[Detector] Checkpoint saved.")

    def predict(self, image: np.ndarray):
        res = self.model.predict(image, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            cls = self.model.names.get(cls_id, str(cls_id))
            out.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":conf,"cls":cls})
        return out

    def train_step(self, imgs, boxes=None):
        if not imgs:
            return
        self.model.model.train()
        self.optimizer.zero_grad()
        results = self.model.model(imgs)
        loss = results["loss"]
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.save_checkpoint()  # persist checkpoint
        return loss.item()
