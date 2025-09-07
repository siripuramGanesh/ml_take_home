import torch
import torch.nn as nn
from ultralytics import YOLO
import asyncio
from typing import List, Dict, Any
import numpy as np
import cv2
from pathlib import Path

# Import our new components
from training_pipeline.dynamic_head import DynamicYOLO
from training_pipeline.continual_loss import DynamicLoss
from training_pipeline.protective_hooks import setup_protective_hooks

class YOLOModel:
    def __init__(self, model_size: str = "yolov8n.pt", max_new_classes: int = 20):
        self.model_size = model_size
        self.max_new_classes = max_new_classes
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = {"total_inferences": 0, "avg_confidence": 0.0}
        self.class_mapping = {}  # Maps class IDs to names
        self.new_classes = set()  # Track newly added classes
        
    def _load_model(self):
        """Load YOLO model and wrap with dynamic head"""
        try:
            # Load original YOLO
            base_model = YOLO(self.model_size)
            print(f"✅ Loaded base YOLO model: {self.model_size}")
            
            # Wrap with dynamic head
            dynamic_model = DynamicYOLO(base_model, max_new_classes=self.max_new_classes)
            print(f"✅ Initialized dynamic head for {self.max_new_classes} new classes")
            
            # Setup protective hooks
            setup_protective_hooks(dynamic_model)
            print("✅ Protective gradient hooks installed")
            
            return dynamic_model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return self._create_fallback_model()

    async def predict_async(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference with dynamic model"""
        loop = asyncio.get_event_loop()
        
        if image.shape[0] > 1280 or image.shape[1] > 1280:
            image = cv2.resize(image, (640, 640))
        
        try:
            # Convert to tensor and run inference
            tensor_image = self._preprocess_image(image)
            with torch.no_grad():
                results = await loop.run_in_executor(
                    None, lambda: self.model(tensor_image)
                )
            
            formatted_results = self._format_dynamic_results(results)
            self._update_metrics(formatted_results)
            
            return formatted_results
            
        except Exception as e:
            print(f"Inference error: {e}")
            return []

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to model input tensor"""
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        return tensor

    def _format_dynamic_results(self, results) -> List[Dict[str, Any]]:
        """Format dynamic model output for API response"""
        boxes, scores, class_ids = results
        
        formatted = []
        for i in range(boxes.shape[1]):  # Iterate through detections
            if scores[0, i] > 0.25:  # Confidence threshold
                box = boxes[0, i].tolist()
                confidence = float(scores[0, i])
                class_id = int(class_ids[0, i])
                
                # Determine if this is a new or original class
                is_new_class = class_id >= 80  # COCO has 80 classes
                class_type = "new" if is_new_class else "original"
                
                formatted.append({
                    "bbox": box,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": self._get_class_name(class_id),
                    "class_type": class_type,
                    "area": (box[2] - box[0]) * (box[3] - box[1])
                })
        
        return formatted

    def _get_class_name(self, class_id: int) -> str:
        """Get class name, handling both original and new classes"""
        if class_id in self.class_mapping:
            return self.class_mapping[class_id]
        elif class_id < 80:  # Original COCO class
            return f"coco_class_{class_id}"
        else:  # New class
            return f"new_class_{class_id - 80}"

    def add_new_class(self, class_name: str, initial_samples: int = 10):
        """Register a new class for detection"""
        if class_name in self.new_classes:
            print(f"⚠️ Class {class_name} already exists")
            return
        
        # Assign next available class ID
        new_class_id = 80 + len(self.new_classes)  # Start after COCO classes
        
        if new_class_id >= 80 + self.max_new_classes:
            print(f"❌ Maximum new classes ({self.max_new_classes}) reached")
            return
        
        self.new_classes.add(class_name)
        self.class_mapping[new_class_id] = class_name
        
        print(f"✅ Added new class: {class_name} (ID: {new_class_id})")
        print(f"📊 {len(self.new_classes)}/{self.max_new_classes} new classes used")

    def get_model_info(self) -> Dict:
        """Get information about model architecture"""
        return {
            "base_model": self.model_size,
            "max_new_classes": self.max_new_classes,
            "current_new_classes": len(self.new_classes),
            "new_classes_list": list(self.new_classes),
            "architecture": "dynamic_dual_head",
            "protected_layers": True
        }

    # ... keep existing _update_metrics, _create_fallback_model, etc. ...

class MockModel:
    """Mock model for testing - updated for dynamic structure"""
    def __init__(self):
        self.metrics = {"total_inferences": 0, "avg_confidence": 0.7}
        self.new_classes = set()
        self.class_mapping = {}
    
    async def predict_async(self, image: np.ndarray):
        await asyncio.sleep(0.1)
        self.metrics["total_inferences"] += 1
        
        # Simulate both original and new class detections
        height, width = image.shape[:2]
        
        # Original class detection
        orig_detection = {
            "bbox": [width//4, height//4, width//2, height//2],
            "confidence": 0.85,
            "class_id": 0,  # person
            "class_name": "person",
            "class_type": "original",
            "area": (width//2 - width//4) * (height//2 - height//4)
        }
        
        # New class detection (if any)
        detections = [orig_detection]
        
        if self.new_classes:
            new_detection = {
                "bbox": [width//2, height//2, 3*width//4, 3*height//4],
                "confidence": 0.65,
                "class_id": 80,  # first new class
                "class_name": next(iter(self.new_classes)),
                "class_type": "new", 
                "area": (3*width//4 - width//2) * (3*height//4 - height//2)
            }
            detections.append(new_detection)
        
        return detections
    
    def add_new_class(self, class_name: str):
        self.new_classes.add(class_name)
        print(f"✅ [Mock] Added new class: {class_name}")
    
    def get_metrics(self):
        return self.metrics