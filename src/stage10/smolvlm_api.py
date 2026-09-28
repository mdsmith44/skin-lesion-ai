"""Internal, visual-only HTTP service for the pinned Stage 10 SmolVLM2 model."""

from contextlib import asynccontextmanager
from io import BytesIO
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile

from src.stage10.smolvlm_description import SmolVLMDescriber


MAX_IMAGE_BYTES = 10 * 1024 * 1024
logger = logging.getLogger(__name__)


def create_app(describer_factory=None) -> FastAPI:
    """Construct the service; load one describer when its lifespan starts."""
    factory = describer_factory or SmolVLMDescriber

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.describer = factory()
        except Exception:
            logger.exception("SmolVLM2 model failed to load")
            app.state.describer = None
        yield
        app.state.describer = None

    app = FastAPI(title="Stage 10 visual-description service", lifespan=lifespan)
    app.state.describer = None

    @app.get("/health")
    async def health(request: Request):
        loaded = request.app.state.describer is not None
        return JSONResponse(
            {"application": "ready", "smolvlm2_model": "ready" if loaded else "unavailable"},
            status_code=200 if loaded else 503,
        )

    @app.post("/describe")
    async def describe(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "multipart/form-data":
            raise HTTPException(415, "Expected multipart/form-data with one image file.")
        try:
            async with request.form(max_files=1, max_fields=0) as form:
                parts = form.multi_items()
                if len(parts) != 1 or parts[0][0] != "image" or not isinstance(parts[0][1], UploadFile):
                    raise HTTPException(400, "Expected exactly one uploaded image named image.")
                image_bytes = await parts[0][1].read(MAX_IMAGE_BYTES + 1)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, "Invalid multipart image upload.") from exc
        if not image_bytes:
            raise HTTPException(400, "Uploaded image is empty.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Uploaded image exceeds the size limit.")
        try:
            with Image.open(BytesIO(image_bytes)) as opened:
                image = opened.convert("RGB")
                image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, "Uploaded file is not a readable image.") from exc

        describer = request.app.state.describer
        if describer is None:
            raise HTTPException(503, "SmolVLM2 model is unavailable.")
        try:
            # Synchronous generation keeps this single-worker service serial.
            return describer.describe(image)
        except Exception as exc:
            logger.exception("SmolVLM2 generation failed")
            raise HTTPException(500, "Description generation failed.") from exc

    return app


app = create_app()
