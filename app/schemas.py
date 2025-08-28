from pydantic import BaseModel, model_validator
from typing import Dict, List, Optional

# Collector
class DataCollectorRequest(BaseModel):
    image_npy_b64: Optional[str] = None
    image_bytes_b64: Optional[str] = None
    bounding_boxes: Dict[str, List[List[int]]]
    message: str = ""

    @model_validator(mode="before")
    def check_image(cls, values):
        if not (values.get("image_npy_b64") or values.get("image_bytes_b64")):
            raise ValueError("At least one image field must be provided")
        return values

class Ok(BaseModel):
    ok: bool

# Inference
class DetectionBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: str

class InferenceRequest(BaseModel):
    image_npy_b64: Optional[str] = None
    image_bytes_b64: Optional[str] = None

    @model_validator(mode="before")
    def check_image(cls, values):
        if not (values.get("image_npy_b64") or values.get("image_bytes_b64")):
            raise ValueError("At least one image field must be provided")
        return values

class InferenceResponse(BaseModel):
    boxes: List[DetectionBox]
