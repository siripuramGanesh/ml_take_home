from pydantic import BaseModel
from typing import List

class Detection(BaseModel):
    bbox: list[float]
    label: str
    confidence: float

class InferResponse(BaseModel):
    detections: List[Detection]
