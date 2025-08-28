from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
import json

from .routers import collect, infer
from .training import trainer

@asynccontextmanager
async def lifespan(app: FastAPI):
    await trainer.start()
    yield

app = FastAPI(title="TaskFlow-ML", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(collect.router, prefix="/api", tags=["collector"])
app.include_router(infer.router, prefix="/api", tags=["inference"])

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/mcp/manifest.json")
async def mcp_manifest():
    manifest_path = Path(__file__).parent.parent / "mcp.json"
    if not manifest_path.exists():
        return JSONResponse(status_code=404, content={"error": "MCP manifest not found"})
    return JSONResponse(content=json.loads(manifest_path.read_text()))
