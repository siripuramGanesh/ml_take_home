import asyncio
import base64
from io import BytesIO
from typing import List, Optional

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .models import YOLOModel
from .training import ContinuousTrainer

app = FastAPI(title="YOLO Inference API", version="1.0.0")
model = YOLOModel()
trainer = ContinuousTrainer()

@app.post("/inference")
async def inference_endpoint(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    labels: Optional[List[str]] = None
):
    try:
        image_data = await image.read()
        pil_image = Image.open(BytesIO(image_data))
        np_image = np.array(pil_image)
        
        if np_image.shape[2] == 4:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
        elif np_image.shape[2] == 1:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
        
        results = await model.predict_async(np_image)
        
        if labels:
            background_tasks.add_task(
                trainer.schedule_training, 
                np_image, labels, results
            )
        
        return JSONResponse({
            "predictions": results,
            "image_shape": np_image.shape,
            "status": "success"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model/status")
async def model_status():
    return {
        "training_phase": trainer.current_phase,
        "dataset_mix": trainer.get_current_mix(),
        "performance_metrics": model.get_metrics(),
        "model_type": "YOLOv8"
    }

@app.post("/inference/base64")
async def inference_base64(
    background_tasks: BackgroundTasks,
    image_data: str,
    labels: Optional[List[str]] = None
):
    try:
        image_bytes = base64.b64decode(image_data)
        np_image = np.array(Image.open(BytesIO(image_bytes)))
        
        results = await model.predict_async(np_image)
        
        if labels:
            background_tasks.add_task(
                trainer.schedule_training, 
                np_image, labels, results
            )
        
        return {"predictions": results, "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)