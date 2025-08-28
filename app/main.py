from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
import json

# Import routers and trainer
from .routers import collect, infer
from .training import trainer

# -----------------------------
# Lifespan for startup/shutdown
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background training loop
    await trainer.start()
    yield
    # Optionally: stop background training if needed
    # await trainer.stop()

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI(title="TaskFlow-ML", version="0.1.0", lifespan=lifespan)

# -----------------------------
# CORS middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Include routers
# -----------------------------
app.include_router(collect.router, prefix="/api", tags=["collector"])
app.include_router(infer.router, prefix="/api", tags=["inference"])

# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok"}

# -----------------------------
# MCP manifest endpoint
# -----------------------------
@app.get("/mcp/manifest.json")
async def mcp_manifest():
    """
    Return the MCP (Model Context Protocol) manifest.
    LLMs can discover your service and endpoints automatically.
    """
    manifest_path = Path(__file__).parent.parent / "mcp.json"
    if not manifest_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "MCP manifest not found"}
        )
    manifest = json.loads(manifest_path.read_text())
    return JSONResponse(content=manifest)
