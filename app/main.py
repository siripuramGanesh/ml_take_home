from fastapi import FastAPI
from app.routers import infer,collect
import asyncio
from app.services.trainer import continuos_training

app=FastAPI()

app.include_router(infer.router)
app.include_router(collect.router)

@app.on_event("startup")
async def start_background_training():
    asyncio.create_task(continuos_training())
