import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np
from PIL import Image


class MLClient:
    def __init__(self, base_url: str = "http://localhost:8000", collector_url: str = "http://localhost:8001"):
        self.inference_url = base_url.rstrip('/')
        self.collector_url = collector_url.rstrip('/')
        self.client = httpx.AsyncClient()
    
    async def infer(self, image_path: str, labels: Optional[List[str]] = None) -> Dict:
        """Run inference on an image"""
        try:
            with open(image_path, "rb") as f:
                files = {"image": f}
                data = {"labels": json.dumps(labels)} if labels else {}
                response = await self.client.post(
                    f"{self.inference_url}/inference",
                    files=files,
                    data=data,
                    timeout=30.0
                )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def infer_base64(self, image_path: str, labels: Optional[List[str]] = None) -> Dict:
        """Run inference using base64 encoding"""
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            data = {
                "image_data": image_data,
                "labels": json.dumps(labels) if labels else "[]"
            }
            
            response = await self.client.post(
                f"{self.inference_url}/inference/base64",
                json=data,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def collect(self, image_path: str, bounding_boxes: Dict[str, List], message: str = "") -> Dict:
        """Collect data with bounding boxes"""
        try:
            with open(image_path, "rb") as f:
                files = {"image": f}
                data = {
                    "bounding_boxes": json.dumps(bounding_boxes),
                    "message": message
                }
                response = await self.client.post(
                    f"{self.collector_url}/collect",
                    files=files,
                    data=data,
                    timeout=30.0
                )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def collect_base64(self, image_path: str, bounding_boxes: Dict[str, List], message: str = "") -> Dict:
        """Collect data using base64 encoding"""
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            data = {
                "image_data": image_data,
                "bounding_boxes": json.dumps(bounding_boxes),
                "message": message
            }
            
            response = await self.client.post(
                f"{self.collector_url}/collect/base64",
                json=data,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def get_model_status(self) -> Dict:
        """Get model status and metrics"""
        try:
            response = await self.client.get(f"{self.inference_url}/model/status", timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        try:
            response = await self.client.get(f"{self.collector_url}/dataset/stats", timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Synchronous client for convenience
class SyncMLClient:
    def __init__(self, base_url: str = "http://localhost:8000", collector_url: str = "http://localhost:8001"):
        self.client = MLClient(base_url, collector_url)
    
    def infer(self, image_path: str, labels: Optional[List[str]] = None) -> Dict:
        """Synchronous inference"""
        return asyncio.run(self.client.infer(image_path, labels))
    
    def collect(self, image_path: str, bounding_boxes: Dict[str, List], message: str = "") -> Dict:
        """Synchronous data collection"""
        return asyncio.run(self.client.collect(image_path, bounding_boxes, message))
    
    def get_model_status(self) -> Dict:
        """Synchronous model status"""
        return asyncio.run(self.client.get_model_status())
    
    def get_dataset_stats(self) -> Dict:
        """Synchronous dataset stats"""
        return asyncio.run(self.client.get_dataset_stats())