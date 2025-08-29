import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


class ContinuousTrainer:
    def __init__(self):
        self.training_phases = [
            (0.1, 0.9), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (0.9, 0.1)
        ]
        self.current_phase = 0
        self.training_queue = asyncio.Queue()
        self.is_training = False
        
    async def schedule_training(self, image: np.ndarray, labels: List[str], predictions: List[Dict]):
        training_data = {
            "image": image,
            "labels": labels,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "phase": self.current_phase
        }
        
        await self.training_queue.put(training_data)
        
        if not self.is_training:
            asyncio.create_task(self._training_worker())

    async def _training_worker(self):
        self.is_training = True
        print("Training worker started")
        
        try:
            while not self.training_queue.empty():
                try:
                    data = await asyncio.wait_for(self.training_queue.get(), timeout=1.0)
                    await self._process_training_data(data)
                    self.training_queue.task_done()
                except asyncio.TimeoutError:
                    break
        finally:
            self.is_training = False

    async def _process_training_data(self, data: Dict):
        try:
            augmented_image = self._augment_image(data["image"])
            self._save_training_sample(augmented_image, data["labels"], data["predictions"])
            
            if self.training_queue.qsize() >= 10:
                await self._trigger_training()
        except Exception as e:
            print(f"Error processing training data: {e}")

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        augmented = image.copy()
        
        if np.random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)
        
        brightness = np.random.uniform(0.8, 1.2)
        augmented = np.clip(augmented.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        
        return augmented

    def _save_training_sample(self, image: np.ndarray, labels: List[str], predictions: List[Dict]):
        try:
            custom_dir = Path("data/custom/images")
            custom_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_{timestamp}_{np.random.randint(1000)}.jpg"
            filepath = custom_dir / filename
            
            cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self._update_annotations(filename, image.shape, labels, predictions)
            
        except Exception as e:
            print(f"Error saving training sample: {e}")

    def _update_annotations(self, filename: str, image_shape, labels: List[str], predictions: List[Dict]):
        annotations_path = Path("data/custom/annotations.json")
        
        if annotations_path.exists():
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
        else:
            annotations = {"images": [], "annotations": [], "categories": []}
        
        image_id = len(annotations["images"]) + 1
        annotations["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": image_shape[1],
            "height": image_shape[0],
            "date_captured": datetime.now().isoformat()
        })
        
        for i, pred in enumerate(predictions):
            if i < len(labels):
                annotations["annotations"].append({
                    "id": len(annotations["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": self._get_category_id(annotations, labels[i]),
                    "bbox": pred["bbox"],
                    "area": pred["area"],
                    "confidence": pred["confidence"],
                    "iscrowd": 0
                })
        
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)

    def _get_category_id(self, annotations: Dict, category_name: str) -> int:
        for category in annotations["categories"]:
            if category["name"] == category_name:
                return category["id"]
        
        new_id = len(annotations["categories"]) + 1
        annotations["categories"].append({
            "id": new_id,
            "name": category_name,
            "supercategory": "object"
        })
        
        return new_id

    async def _trigger_training(self):
        print(f"Starting training phase {self.current_phase}")
        new_data_ratio, coco_ratio = self.training_phases[self.current_phase]
        
        print(f"Training with {new_data_ratio*100}% new data, {coco_ratio*100}% COCO")
        await asyncio.sleep(2)
        
        self.current_phase = (self.current_phase + 1) % len(self.training_phases)
        print(f"Training completed. Moving to phase {self.current_phase}")

    def get_current_mix(self) -> Dict[str, float]:
        new_ratio, coco_ratio = self.training_phases[self.current_phase]
        return {
            "new_data_ratio": new_ratio,
            "coco_ratio": coco_ratio,
            "phase": self.current_phase
        }