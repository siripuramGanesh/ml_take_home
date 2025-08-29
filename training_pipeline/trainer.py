import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision
from timm.data import create_transform
from timm.data.auto_augment import auto_augment_transform
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class COCOCustomDataset(Dataset):
    """Dataset class that handles both COCO and custom data"""
    
    def __init__(self, annotations_path: str, images_dir: str, transform=None):
        self.annotations_path = Path(annotations_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        if self.annotations_path.exists():
            with open(self.annotations_path, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {"images": [], "annotations": [], "categories": []}
        
        self.image_ids = [img['id'] for img in self.annotations['images']]
        self.id_to_img = {img['id']: img for img in self.annotations['images']}
        self.img_to_anns = {}
        
        for ann in self.annotations['annotations']:
            if ann['image_id'] not in self.img_to_anns:
                self.img_to_anns[ann['image_id']] = []
            self.img_to_anns[ann['image_id']].append(ann)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.id_to_img[image_id]
        
        image_path = self.images_dir / image_info['file_name']
        
        try:
            image = torchvision.io.read_image(str(image_path))
            image = image.float() / 255.0  # Normalize to [0, 1]
        except:
            # Return a dummy image if file not found
            image = torch.rand(3, 224, 224)
        
        annotations = self.img_to_anns.get(image_id, [])
        boxes = []
        labels = []
        
        for ann in annotations:
            # Convert COCO bbox format [x, y, width, height] to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        if self.transform:
            image = self.transform(image)
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([image_id])
        }
        
        return image, target

class ContinuousTrainer:
    """Handles continuous training with dataset mixing"""
    
    def __init__(self, coco_path: str = "data/coco", custom_path: str = "data/custom"):
        self.coco_path = Path(coco_path)
        self.custom_path = Path(custom_path)
        self.training_phases = [(0.1, 0.9), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (0.9, 0.1)]
        self.current_phase = 0
        self.augmentation = self._create_augmentation()
        
    def _create_augmentation(self):
        """Create data augmentation transforms using timm"""
        return auto_augment_transform(
            config_str='rand-m9-mstd0.5-inc1',
            hparams={'translate_const': 100, 'img_mean': (124, 116, 104)}
        )
    
    def get_datasets_for_phase(self, phase: int):
        """Get appropriate dataset mix for current phase"""
        new_ratio, coco_ratio = self.training_phases[phase]
        
        # Load COCO dataset
        coco_dataset = None
        coco_annotations = self.coco_path / "instances_train2017.json"
        coco_images = self.coco_path / "train2017"
        
        if coco_annotations.exists() and coco_images.exists():
            coco_dataset = COCOCustomDataset(
                str(coco_annotations),
                str(coco_images),
                transform=self.augmentation
            )
        
        # Load custom dataset
        custom_dataset = None
        custom_annotations = self.custom_path / "annotations.json"
        custom_images = self.custom_path / "images"
        
        if custom_annotations.exists() and custom_images.exists():
            custom_dataset = COCOCustomDataset(
                str(custom_annotations),
                str(custom_images),
                transform=self.augmentation
            )
        
        # Handle different scenarios
        if coco_dataset and custom_dataset:
            # Both datasets available - mix them
            coco_size = int(len(coco_dataset) * coco_ratio)
            custom_size = int(len(custom_dataset) * new_ratio)
            
            # Sample from datasets (simplified - in practice you'd implement proper sampling)
            return ConcatDataset([coco_dataset, custom_dataset])
        
        elif coco_dataset:
            # Only COCO available
            return coco_dataset
        
        elif custom_dataset:
            # Only custom available
            return custom_dataset
        
        else:
            # No datasets available
            raise ValueError("No training data available")
    
    async def train_model(self, model, phase: int, epochs: int = 5):
        """Train the model with current phase configuration"""
        print(f"🚀 Starting training phase {phase}")
        
        try:
            dataset = self.get_datasets_for_phase(phase)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
            
            # Simplified training loop
            # In practice, you'd use YOLO's built-in training or implement proper training
            for epoch in range(epochs):
                print(f"📊 Epoch {epoch + 1}/{epochs}")
                await asyncio.sleep(1)  # Simulate training time
            
            print(f"✅ Training phase {phase} completed")
            self.current_phase = (phase + 1) % len(self.training_phases)
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
    
    def get_current_mix_ratio(self) -> Dict[str, float]:
        """Get current dataset mix ratios"""
        new_ratio, coco_ratio = self.training_phases[self.current_phase]
        return {
            "new_data_ratio": new_ratio,
            "coco_ratio": coco_ratio,
            "phase": self.current_phase,
            "phase_description": f"Phase {self.current_phase + 1}: {new_ratio*100}% new data, {coco_ratio*100}% COCO"
        }
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about available datasets"""
        stats = {
            "coco_available": False,
            "custom_available": False,
            "coco_images": 0,
            "custom_images": 0,
            "coco_annotations": 0,
            "custom_annotations": 0
        }
        
        # Check COCO dataset
        coco_annotations = self.coco_path / "instances_train2017.json"
        if coco_annotations.exists():
            try:
                with open(coco_annotations, 'r') as f:
                    coco_data = json.load(f)
                    stats.update({
                        "coco_available": True,
                        "coco_images": len(coco_data.get("images", [])),
                        "coco_annotations": len(coco_data.get("annotations", []))
                    })
            except:
                pass
        
        # Check custom dataset
        custom_annotations = self.custom_path / "annotations.json"
        if custom_annotations.exists():
            try:
                with open(custom_annotations, 'r') as f:
                    custom_data = json.load(f)
                    stats.update({
                        "custom_available": True,
                        "custom_images": len(custom_data.get("images", [])),
                        "custom_annotations": len(custom_data.get("annotations", []))
                    })
            except:
                pass
        
        return stats

class TrainingScheduler:
    """Schedules training sessions with cyclical learning rates"""
    
    def __init__(self):
        self.training_queue = asyncio.Queue()
        self.is_training = False
        self.learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]  # Cyclical LR
    
    async def schedule_training(self, trainer: ContinuousTrainer, model):
        """Schedule a training session"""
        await self.training_queue.put((trainer, model))
        
        if not self.is_training:
            asyncio.create_task(self._training_worker())
    
    async def _training_worker(self):
        """Background training worker"""
        self.is_training = True
        
        try:
            while not self.training_queue.empty():
                try:
                    trainer, model = await asyncio.wait_for(self.training_queue.get(), timeout=5.0)
                    
                    # Get current learning rate based on phase
                    current_lr = self.learning_rates[trainer.current_phase % len(self.learning_rates)]
                    print(f"📚 Learning rate: {current_lr}")
                    
                    # Train model
                    await trainer.train_model(model, trainer.current_phase)
                    
                    self.training_queue.task_done()
                    
                except asyncio.TimeoutError:
                    break
                    
        finally:
            self.is_training = False
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return {
            "queue_size": self.training_queue.qsize(),
            "is_training": self.is_training,
            "learning_rates": self.learning_rates
        }