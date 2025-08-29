# ML TakeHome Project

Production-ready inference and data collection pipeline with continuous learning.

## Features

- YOLO inference endpoint with continuous training
- Data collection endpoint in COCO format
- Python client with CLI interface
- Progressive dataset mixing (COCO + custom data)
- Data augmentation with timm
- Asynchronous FastAPI servers
# Install requirements
pip install -r requirements.txt

## Installation

```bash
# Install from git
pip install git+https://github.com/your-username/ml_take_home.git

# Or install locally
git clone https://github.com/your-username/ml_take_home.git
cd ml_take_home
pip install -e .

# Download full COCO dataset (18GB+)
python download_coco.py --download-full

# Or download only annotations (241MB)
python download_coco.py --download-annotations-only

#Start servers:

# Or skip download for development
python download_coco.py --skip-download

# Inference server (port 8000)
uvicorn inference.app:app --reload --port 8000

# Data collector (port 8001)
uvicorn data_collector.app:app --reload --port 8001

#Use CLI client
# Run inference
ml-takehome infer -i image.jpg -l cat -l dog

# Collect data with bounding boxes
ml-takehome collect -i image.jpg -b bboxes.json -m "Description"

# Use test images
ml-takehome infer -i data/test_images/test_person_0.jpg -l person

#Project Structure
ml-takehome/
├── inference/           # YOLO inference API
├── data_collector/      # Data collection API  
├── client/              # Python client + CLI
├── training_pipeline/   # Continuous training
├── data/               # Datasets (COCO + custom)
└── tests/              # Test cases