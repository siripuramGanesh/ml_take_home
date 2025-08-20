from fastapi import APIRouter,UploadFile, File
from app.schemas.infer import InferResponse
from app.services.yolo_service import run_inference

router=APIRouter()

@router.post("/infer",response_model=InferResponse)
async def infer(image: UploadFile=File(...)):
    image_bytes=await image.read()
    detections=run_inference(image_bytes)
    return {"detections": detections}