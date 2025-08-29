import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


class COCOHandler:
    def __init__(self, coco_path: str = "data/coco", custom_data_path: str = "data/custom"):
        self.coco_path = Path(coco_path)
        self.custom_data_path = Path(custom_data_path)
        self.custom_annotations = self._initialize_custom_dataset()
        
    def _initialize_custom_dataset(self) -> Dict:
        """Initialize custom dataset structure"""
        custom_annotation_path = self.custom_data_path / "annotations.json"
        
        if custom_annotation_path.exists():
            with open(custom_annotation_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "images": [],
                "annotations": [],
                "categories": [],
                "info": {
                    "description": "Custom Object Detection Dataset",
                    "url": "",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "ML TakeHome System",
                    "date_created": datetime.now().isoformat()
                }
            }

    def add_image(self, image: np.ndarray, message: str = "") -> int:
        """Add new image to custom dataset"""
        try:
            image_id = len(self.custom_annotations["images"]) + 1
            image_filename = f"custom_{image_id:06d}.jpg"
            image_path = self.custom_data_path / "images" / image_filename
            
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            self.custom_annotations["images"].append({
                "id": image_id,
                "width": image.shape[1],
                "height": image.shape[0],
                "file_name": image_filename,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": datetime.now().isoformat(),
                "message": message
            })
            
            self._save_custom_annotations()
            return image_id
            
        except Exception as e:
            print(f"Error adding image: {e}")
            raise

    def add_annotation(self, image_id: int, label: str, bbox: List[Tuple[int, int]]):
        """Add annotation for an image"""
        try:
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            
            coco_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = (x_max - x_min) * (y_max - y_min)
            
            category_id = self._get_or_create_category(label)
            
            annotation_id = len(self.custom_annotations["annotations"]) + 1
            
            self.custom_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": coco_bbox,
                "area": area,
                "segmentation": [],
                "iscrowd": 0
            })
            
            self._save_custom_annotations()
            
        except Exception as e:
            print(f"Error adding annotation: {e}")
            raise

    def _get_or_create_category(self, label: str) -> int:
        """Get existing category ID or create new one"""
        for category in self.custom_annotations["categories"]:
            if category["name"].lower() == label.lower():
                return category["id"]
        
        category_id = len(self.custom_annotations["categories"]) + 1
        self.custom_annotations["categories"].append({
            "id": category_id,
            "name": label,
            "supercategory": "object"
        })
        
        return category_id

    def _save_custom_annotations(self):
        """Save custom annotations to file"""
        self.custom_data_path.mkdir(parents=True, exist_ok=True)
        annotation_path = self.custom_data_path / "annotations.json"
        
        with open(annotation_path, 'w') as f:
            json.dump(self.custom_annotations, f, indent=2)

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            "coco_images": self._count_coco_images(),
            "coco_annotations": self._count_coco_annotations(),
            "coco_categories": self._count_coco_categories(),
            "custom_images": len(self.custom_annotations.get("images", [])),
            "custom_annotations": len(self.custom_annotations.get("annotations", [])),
            "custom_categories": len(self.custom_annotations.get("categories", []))
        }

    def _count_coco_images(self) -> int:
        """Count COCO images"""
        coco_images_dir = self.coco_path / "train2017"
        if coco_images_dir.exists():
            return len(list(coco_images_dir.glob("*.jpg")))
        return 0

    def _count_coco_annotations(self) -> int:
        """Count COCO annotations"""
        coco_annotations = self.coco_path / "instances_train2017.json"
        if coco_annotations.exists():
            try:
                with open(coco_annotations, 'r') as f:
                    data = json.load(f)
                    return len(data.get("annotations", []))
            except:
                pass
        return 0

    def _count_coco_categories(self) -> int:
        """Count COCO categories"""
        coco_annotations = self.coco_path / "instances_train2017.json"
        if coco_annotations.exists():
            try:
                with open(coco_annotations, 'r') as f:
                    data = json.load(f)
                    return len(data.get("categories", []))
            except:
                pass
        return 0

    def get_training_data(self, phase: int) -> Dict:
        """Get training data mix based on current phase"""
        phases = [(0.1, 0.9), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (0.9, 0.1)]
        new_ratio, coco_ratio = phases[phase]
        
        training_data = {
            "images": [],
            "annotations": [],
            "categories": self._merge_categories()
        }
        
        return training_data

    def _merge_categories(self) -> List[Dict]:
        """Merge categories from both datasets"""
        merged = []
        seen = set()
        
        coco_categories = self._load_coco_categories()
        for category in coco_categories:
            if category["name"] not in seen:
                merged.append(category)
                seen.add(category["name"])
        
        for category in self.custom_annotations.get("categories", []):
            if category["name"] not in seen:
                merged.append(category)
                seen.add(category["name"])
        
        return merged

    def _load_coco_categories(self) -> List[Dict]:
        """Load COCO categories"""
        coco_annotations = self.coco_path / "instances_train2017.json"
        if coco_annotations.exists():
            try:
                with open(coco_annotations, 'r') as f:
                    data = json.load(f)
                    return data.get("categories", [])
            except:
                pass
        return []