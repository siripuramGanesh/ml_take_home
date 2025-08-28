from fastapi import APIRouter, HTTPException
from ..schemas import DataCollectorRequest, Ok
from ..dataset import decode_image, save_new_sample
from ..training import trainer
import asyncio

router = APIRouter()

@router.post("/collect", response_model=Ok)
async def collect(req: DataCollectorRequest):
    # Decode image from either npy or bytes
    try:
        img = decode_image(req.image_npy_b64, req.image_bytes_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Save new sample with LLM/VLM refinement
    try:
        save_new_sample(img, req.bounding_boxes, req.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid bounding boxes: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save sample: {e}")

    # Enqueue a training batch asynchronously
    try:
        await trainer.enqueue_mixed_batch()
    except Exception as e:
        print("[Collect] Warning: Failed to enqueue batch:", e)

    return Ok(ok=True)
