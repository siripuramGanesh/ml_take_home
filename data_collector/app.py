from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from io import BytesIO
import json
import cv2
from typing import Dict, List
import base64

from .coco_handler import COCOHandler

app = FastAPI(title="Data Collector API", version="1.0.0")
coco_handler = COCOHandler()

@app.post("/collect")
async def collect_data(
    image: UploadFile = File(...),
    bounding_boxes: str,
    message: str = ""
):
    try:
        image_data = await image.read()
        pil_image = Image.open(BytesIO(image_data))
        np_image = np.array(pil_image)
        
        if np_image.shape[2] == 4:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
        elif np_image.shape[2] == 1:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
        
        try:
            bboxes = json.loads(bounding_boxes)
            if not isinstance(bboxes, dict):
                raise ValueError("Bounding boxes must be a dictionary")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        
        _validate_bboxes(bboxes, np_image.shape)
        
        image_id = coco_handler.add_image(np_image, message)
        
        for label, boxes in bboxes.items():
            for bbox in boxes:
                coco_handler.add_annotation(image_id, label, bbox)
        
        return JSONResponse({
            "status": "success", 
            "image_id": image_id,
            "message": "Data added to dataset"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.post("/collect/base64")
async def collect_data_base64(
    image_data: str,
    bounding_boxes: str,
    message: str = ""
):
    try:
        image_bytes = base64.b64decode(image_data)
        np_image = np.array(Image.open(BytesIO(image_bytes)))
        
        bboxes = json.loads(bounding_boxes)
        
        image_id = coco_handler.add_image(np_image, message)
        
        for label, boxes in bboxes.items():
            for bbox in boxes:
                coco_handler.add_annotation(image_id, label, bbox)
        
        return {"status": "success", "image_id": image_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/stats")
async def dataset_stats():
    stats = coco_handler.get_stats()
    return JSONResponse(stats)

@app.get("/dataset/images/count")
async def image_count():
    stats = coco_handler.get_stats()
    return {"total_images": stats["custom_images"]}

def _validate_bboxes(bboxes: Dict, image_shape: tuple):
    height, width = image_shape[:2]
    
    for label, boxes in bboxes.items():
        if not isinstance(boxes, list):
            raise HTTPException(status_code=400, detail=f"Boxes for {label} must be a list")
        
        for i, bbox in enumerate(boxes):
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Box {i} for {label} must be a list of 4 coordinates"
                )
            
            for coord in bbox:
                if not (0 <= coord[0] <= width and 0 <= coord[1] <= height):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Coordinate {coord} out of image bounds"
                    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)