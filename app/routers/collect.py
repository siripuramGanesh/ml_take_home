from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.collect import CollectRequest, CollectResponse, BoundingBox
from app.services.dataset import save_data
import json

router = APIRouter()

@router.post("/collect", response_model=CollectResponse)
async def collect(
    image: UploadFile = File(...),
    annotations: str = Form(...),  # JSON string of BoundingBox list
    message: str = Form(" ")
):
    """
    Accepts an uploaded image with bounding boxes and optional message.
    Saves the data to `data/new/`.
    """
    try:
        # Parse annotations JSON into list of BoundingBox
        annotations_list = [BoundingBox(**ann) for ann in json.loads(annotations)]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid annotations format: {e}")

    # Read image bytes
    image_bytes = await image.read()

    # Save image + annotations
    image_file, num_annotations = save_data(
        image_bytes=image_bytes,
        filename=image.filename,
        annotations=annotations_list,
        message=message
    )

    return {
        "status": "success",
        "image_file": image_file,
        "num_annotations": num_annotations
    }
