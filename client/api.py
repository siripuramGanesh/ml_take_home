import httpx
import base64
import numpy as np

API_URL = "http://localhost:8000/api"

class MLClient:
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def act(self, image: np.ndarray, bounding_boxes: dict, message: str):
        """
        Send image + bounding boxes to /collect
        """
        img_bytes = self._encode_image(image)
        payload = {
            "image_bytes_b64": img_bytes,
            "bounding_boxes": bounding_boxes,
            "message": message
        }
        r = await self.client.post(f"{self.base_url}/collect", json=payload)
        r.raise_for_status()
        return r.json()

    async def infer(self, image: np.ndarray):
        """
        Send image to /infer
        Returns list of bounding boxes
        """
        img_bytes = self._encode_image(image)
        payload = {"image_bytes_b64": img_bytes}
        r = await self.client.post(f"{self.base_url}/infer", json=payload)
        r.raise_for_status()
        return r.json()

    def _encode_image(self, image: np.ndarray) -> str:
        import cv2
        _, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer).decode("utf-8")
