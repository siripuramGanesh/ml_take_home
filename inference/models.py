import asyncio
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YOLOModel:
    def __init__(self, model_size: str = "yolov8n.pt"):
        self.model_size = model_size
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = {"total_inferences": 0, "avg_confidence": 0.0}
        
    def _load_model(self):
        try:
            model = YOLO(self.model_size)
            print(f"✅ Loaded YOLO model: {self.model_size}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return self._create_fallback_model()

    async def predict_async(self, image: np.ndarray) -> List[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        
        if image.shape[0] > 1280 or image.shape[1] > 1280:
            image = cv2.resize(image, (640, 640))
        
        try:
            results = await loop.run_in_executor(
                None, self.model, image, verbose=False
            )
            
            formatted_results = self._format_results(results[0])
            self._update_metrics(formatted_results)
            
            return formatted_results
            
        except Exception as e:
            print(f"Inference error: {e}")
            return []

    def _format_results(self, result) -> List[Dict[str, Any]]:
        formatted = []
        
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                confidence = float(box.conf.item())
                class_id = int(box.cls.item())
                
                formatted.append({
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}",
                    "area": float((box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
                })
        
        return formatted

    def _update_metrics(self, results: List[Dict]):
        self.metrics["total_inferences"] += 1
        
        if results:
            confidences = [r["confidence"] for r in results]
            self.metrics["avg_confidence"] = (
                self.metrics["avg_confidence"] * 0.9 + np.mean(confidences) * 0.1
            )

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()

    def _create_fallback_model(self):
        print("🔄 Creating fallback mock model...")
        return MockModel()

class MockModel:
    def __init__(self):
        self.metrics = {"total_inferences": 0, "avg_confidence": 0.7}
    
    async def predict_async(self, image: np.ndarray):
        await asyncio.sleep(0.1)
        self.metrics["total_inferences"] += 1
        
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        bbox_size = min(width, height) // 4
        
        return [{
            "bbox": [center_x - bbox_size, center_y - bbox_size,
                    center_x + bbox_size, center_y + bbox_size],
            "confidence": 0.85,
            "class_id": 0,
            "class_name": "object",
            "area": bbox_size * bbox_size * 4
        }]
    
    def get_metrics(self):
        return self.metrics