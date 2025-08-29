import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class DatasetManager:
    """Manages dataset operations and statistics"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.coco_path = self.base_path / "coco"
        self.custom_path = self.base_path / "custom"
        
    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information"""
        return {
            "coco": self._get_coco_info(),
            "custom": self._get_custom_info(),
            "total": self._get_total_info()
        }
    
    def _get_coco_info(self) -> Dict:
        """Get COCO dataset information"""
        info = {
            "available": False,
            "images": 0,
            "annotations": 0,
            "categories": 0
        }
        
        if (self.coco_path / "instances_train2017.json").exists():
            try:
                with open(self.coco_path / "instances_train2017.json", 'r') as f:
                    coco_data = json.load(f)
                    info.update({
                        "available": True,
                        "images": len(coco_data.get("images", [])),
                        "annotations": len(coco_data.get("annotations", [])),
                        "categories": len(coco_data.get("categories", []))
                    })
            except:
                pass
        
        return info
    
    def _get_custom_info(self) -> Dict:
        """Get custom dataset information"""
        custom_annotations = self.custom_path / "annotations.json"
        info = {
            "available": False,
            "images": 0,
            "annotations": 0,
            "categories": 0
        }
        
        if custom_annotations.exists():
            try:
                with open(custom_annotations, 'r') as f:
                    custom_data = json.load(f)
                    info.update({
                        "available": True,
                        "images": len(custom_data.get("images", [])),
                        "annotations": len(custom_data.get("annotations", [])),
                        "categories": len(custom_data.get("categories", []))
                    })
            except:
                pass
        
        return info
    
    def _get_total_info(self) -> Dict:
        """Get total dataset information"""
        coco_info = self._get_coco_info()
        custom_info = self._get_custom_info()
        
        return {
            "total_images": coco_info["images"] + custom_info["images"],
            "total_annotations": coco_info["annotations"] + custom_info["annotations"],
            "total_categories": max(coco_info["categories"], custom_info["categories"]),
            "coco_available": coco_info["available"],
            "custom_available": custom_info["available"]
        }
    
    def cleanup_dataset(self, max_age_days: int = 30):
        """Clean up old dataset files"""
        custom_images = self.custom_path / "images"
        if custom_images.exists():
            for image_file in custom_images.glob("*.jpg"):
                if image_file.stat().st_mtime < (datetime.now().timestamp() - max_age_days * 86400):
                    image_file.unlink()
                    print(f"Removed old image: {image_file.name}")
    
    def export_dataset(self, export_path: str, format: str = "coco"):
        """Export dataset in specified format"""
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if format == "coco":
            self._export_coco_format(export_dir)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_coco_format(self, export_dir: Path):
        """Export in COCO format"""
        # This would merge COCO and custom data
        # For now, just copy custom annotations
        import shutil
        
        custom_annotations = self.custom_path / "annotations.json"
        if custom_annotations.exists():
            shutil.copy2(custom_annotations, export_dir / "annotations.json")
        
        custom_images = self.custom_path / "images"
        if custom_images.exists():
            images_dest = export_dir / "images"
            images_dest.mkdir(exist_ok=True)
            for image_file in custom_images.glob("*.jpg"):
                shutil.copy2(image_file, images_dest / image_file.name)