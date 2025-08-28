import httpx
import base64
import numpy as np
import cv2

API_URL = "http://localhost:8000/api"

class MLClient:
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def act(self, image: np.ndarray, bounding_boxes: dict, message: str):
        """Send image + bounding boxes to the collector endpoint"""
        try:
            img_bytes = self._encode_image(image)
            payload = {
                "image_bytes_b64": img_bytes,
                "bounding_boxes": bounding_boxes,
                "message": message
            }
            r = await self.client.post(f"{self.base_url}/collect", json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print("Error sending to collector:", e)
            return {"ok": False, "error": str(e)}

    async def infer(self, image: np.ndarray):
        """Send image to inference endpoint"""
        try:
            img_bytes = self._encode_image(image)
            payload = {"image_bytes_b64": img_bytes}
            r = await self.client.post(f"{self.base_url}/infer", json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print("Error running inference:", e)
            return {"boxes": [], "error": str(e)}

    def _encode_image(self, image: np.ndarray) -> str:
        """Convert image to base64 bytes for API"""
        _, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer).decode("utf-8")
