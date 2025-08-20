from pydantic import BaseModel
from typing import List,Dict,Any

class BoundingBox(BaseModel):
    bbox: List[float]
    label: str
    confidence: float=1.0

class CollectRequest(BaseModel):
    annotations: List[BoundingBox]
    message: str=" "

class CollectResponse(BaseModel):
    status: str
    image_file: str
    num_annotations: int
