from fastapi import APIRouter, HTTPException
from ..schemas import InferenceRequest, InferenceResponse, DetectionBox
from ..dataset import decode_image
from ..training import trainer

router = APIRouter()

@router.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    try:
        img = decode_image(req.image_npy_b64, req.image_bytes_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    boxes = trainer.detector.predict(img)
    return InferenceResponse(boxes=[DetectionBox(**b) for b in boxes])
