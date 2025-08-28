ml_take_home – YOLOv8 Incremental Training & Data Collector
Description

This is a production-ready object detection service built with YOLOv8, FastAPI, and Python.
It supports incremental training with new labeled data, co-training with COCO, and LLM/VLM-assisted label refinement.

# Features

YOLOv8 inference via async FastAPI endpoint (/api/infer)

Data collection endpoint (/api/collect) for incremental training

Async co-training with adjustable ratios of new vs COCO data:
10/90, 25/75, 50/50, 75/25, 90/10

CV2 + TIMM augmentation for robust training

LLM/VLM-assisted labeling and semantic validation

Python client (MLClient) and CLI (cli.py) for sending images and bounding boxes

Safe error handling in client/CLI

MCP manifest support (mcp.json)

Installation
# Clone repo
git clone <repo-url>
cd ml_take_home

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -e .

# Running the Server
uvicorn app.main:app --reload


Health check: GET /api/health

Inference: POST /api/infer

Data collection: POST /api/collect

Make sure to create the minimal COCO dataset folder structure if not using full COCO:
data/coco/annotations/instances_train2017.json with an empty JSON object works for demo.

***CLI Usage***
***Send a single image with bounding boxes:***
python cli.py --image path/to/image.png --bbox '{"label": [[x1,y1,x2,y2]]}' --message "your annotation message"



***Run inference on the same image:***
python cli.py --image path/to/image.png --infer

***Python Client Usage***

import asyncio
import cv2
from client.api import MLClient

async def main():
    client = MLClient()
    
    # Load image
    img = cv2.imread("path/to/image.png")
    
    # Define bounding boxes
    boxes = {"label": [[x1, y1, x2, y2]]}  # replace with your label and coordinates
    
    # Send to data collector
    res = await client.act(img, boxes, "your annotation message")
    print("Collector response:", res)
    
    # Run inference
    inference = await client.infer(img)
    print("Inference result:", inference)

asyncio.run(main())
