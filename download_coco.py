import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


class COCODownloader:
    def __init__(self):
        self.data_dir = Path("data")
        self.coco_dir = self.data_dir / "coco"
        self.custom_dir = self.data_dir / "custom" / "images"
        
    def setup_directories(self):
        """Create necessary directories"""
        self.coco_dir.mkdir(parents=True, exist_ok=True)
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Created directory structure")
    
    def download_file(self, url, destination):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=f"Downloading {destination.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))
            
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def extract_zip(self, zip_path, extract_to):
        """Extract zip file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def download_coco(self, download_images=True, download_annotations=True):
        """Download COCO dataset"""
        coco_files = []
        
        if download_images:
            coco_files.append((
                "http://images.cocodataset.org/zips/train2017.zip",
                self.coco_dir / "train2017.zip"
            ))
        
        if download_annotations:
            coco_files.append((
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                self.coco_dir / "annotations_trainval2017.zip"
            ))
        
        for url, destination in coco_files:
            if destination.exists():
                print(f"⚠️  Already exists: {destination.name}")
                continue
            
            print(f"⬇️  Downloading: {url}")
            if self.download_file(url, destination):
                print(f"📦 Extracting: {destination.name}")
                if self.extract_zip(destination, self.coco_dir):
                    destination.unlink()
                    print(f"✅ Completed: {destination.name}")
    
    def organize_coco_files(self):
        """Organize extracted COCO files"""
        annotations_dir = self.coco_dir / "annotations"
        if annotations_dir.exists():
            for file in annotations_dir.iterdir():
                if file.is_file():
                    file.rename(self.coco_dir / file.name)
            annotations_dir.rmdir()
            print("✅ Organized annotation files")
    
    def create_minimal_test_set(self):
        """Create minimal test dataset"""
        import numpy as np
        from PIL import Image, ImageDraw
        
        test_dir = self.data_dir / "test_images"
        test_dir.mkdir(exist_ok=True)
        
        objects = ["person", "car", "dog"]
        
        for i, obj in enumerate(objects):
            img = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            
            draw.text((20, 20), f"Test: {obj}", fill=(255, 0, 0))
            draw.rectangle([50, 50, 150, 150], outline=(0, 255, 0), width=2)
            
            pil_img.save(test_dir / f"test_{obj}_{i}.jpg")
        
        print(f"✅ Created test images in {test_dir}")
    
    def check_coco_availability(self):
        """Check if COCO dataset is properly set up"""
        required_files = [
            self.coco_dir / "instances_train2017.json",
            self.coco_dir / "train2017"
        ]
        
        for file in required_files:
            if not file.exists():
                return False
        return True
    
    def run(self, download_options):
        """Main method to run the downloader"""
        print("🚀 Setting up COCO dataset...")
        self.setup_directories()
        
        if download_options["download_coco"]:
            self.download_coco(
                download_images=download_options["download_images"],
                download_annotations=download_options["download_annotations"]
            )
            self.organize_coco_files()
        
        if self.check_coco_availability():
            print("🎉 COCO dataset is ready for training!")
        else:
            print("⚠️  COCO dataset not fully available")
            print("🛠️  Creating minimal test dataset...")
            self.create_minimal_test_set()

def main():
    parser = argparse.ArgumentParser(description="Download and setup COCO dataset")
    parser.add_argument("--download-full", action="store_true", 
                       help="Download full COCO dataset (images + annotations)")
    parser.add_argument("--download-annotations-only", action="store_true",
                       help="Download only COCO annotations (241MB)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip downloading, just setup directories")
    
    args = parser.parse_args()
    
    downloader = COCODownloader()
    
    download_options = {
        "download_coco": not args.skip_download,
        "download_images": args.download_full,
        "download_annotations": args.download_full or args.download_annotations_only
    }
    
    downloader.run(download_options)

if __name__ == "__main__":
    main()