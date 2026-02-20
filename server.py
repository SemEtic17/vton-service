import os
import io
import time
import base64
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel, HttpUrl
from PIL import Image

from fashn_vton import TryOnPipeline

app = FastAPI(title="Diva Haus VTON Service")

class TryOnRequest(BaseModel):
    person_image_url: HttpUrl
    garment_image_url: HttpUrl
    category: str = "tops"
    garment_photo_type: str = "model"
    num_samples: int = 1
    num_timesteps: int = 30
    guidance_scale: float = 1.5
    seed: int = 42
    segmentation_free: bool = True


pipeline = None

def ensure_pipeline_initialized():
    """Lazily initialize the TryOn pipeline. This avoids blocking the ASGI
    startup sequence and lets the health endpoint respond quickly while the
    heavy model loads on demand."""
    global pipeline
    if pipeline is None:
        weights_dir = os.getenv("VTON_WEIGHTS_DIR", "./weights")
        device = os.getenv("VTON_DEVICE", None)
        logging.getLogger('uvicorn.error').info(f"[server] initializing FASHN VTON pipeline (weights_dir={weights_dir}, device={device})")
        pipeline = TryOnPipeline(weights_dir=weights_dir, device=device)
        logging.getLogger('uvicorn.error').info("[server] pipeline ready")


@app.post("/vton")
def run_tryon(req: TryOnRequest):
    try:
        # Ensure the heavy ML pipeline is initialized on first request.
        ensure_pipeline_initialized()

        # download images
        resp = requests.get(req.person_image_url)
        resp.raise_for_status()
        person_img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        resp = requests.get(req.garment_image_url)
        resp.raise_for_status()
        garment_img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        start_time = time.time()
        result = pipeline(
            person_image=person_img,
            garment_image=garment_img,
            category=req.category,
            garment_photo_type=req.garment_photo_type,
            num_samples=req.num_samples,
            num_timesteps=req.num_timesteps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            segmentation_free=req.segmentation_free,
        )
        elapsed = int((time.time() - start_time) * 1000)

        if not result or len(result.images) == 0:
            raise RuntimeError("pipeline returned no images")

        output_buf = io.BytesIO()
        result.images[0].save(output_buf, format="PNG")
        output_buf.seek(0)
        b64 = base64.b64encode(output_buf.read()).decode("utf-8")

        return {
            "ok": True,
            "previewBase64": b64,
            "processingTimeMs": elapsed,
            "modelVersion": "fashn-v1.5",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}
