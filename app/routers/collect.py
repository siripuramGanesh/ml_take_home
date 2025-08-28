from fastapi import APIRouter, HTTPException
from ..schemas import DataCollectorRequest, Ok
from ..dataset import decode_image, save_new_sample
from ..training import trainer

router = APIRouter()

@router.post("/collect", response_model=Ok)
async def collect(req: DataCollectorRequest):
    try:
        img = decode_image(req.image_npy_b64, req.image_bytes_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    save_new_sample(img, req.bounding_boxes, req.message)
    await trainer.enqueue_mixed_batch()
    return Ok(ok=True)
