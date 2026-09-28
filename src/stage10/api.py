"""Research-only HTTP classification API backed by the Stage 10 Triton model."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from src.stage10.triton_classifier import (
    InvalidImageError, TritonClassifier, TritonResponseError, TritonUnavailableError,
)


MAX_IMAGE_BYTES = 10 * 1024 * 1024


def create_app(classifier: TritonClassifier | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail startup if the required endpoint has not been configured.
        app.state.classifier = classifier if classifier is not None else TritonClassifier.from_env()
        yield

    app = FastAPI(title="Stage 10 research classifier", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request):
        try:
            server_ready, model_ready = await run_in_threadpool(request.app.state.classifier.readiness)
        except Exception:
            server_ready, model_ready = False, False
        body = {
            "application": "ready",
            "triton_server": "ready" if server_ready else "unavailable",
            "resnet18_fp16": "ready" if model_ready else "unavailable",
        }
        return JSONResponse(body, status_code=200 if server_ready and model_ready else 503)

    @app.post("/classify")
    async def classify(request: Request):
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
            return await run_in_threadpool(request.app.state.classifier.classify, image_bytes)
        except InvalidImageError as exc:
            raise HTTPException(400, str(exc)) from exc
        except TritonUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        except TritonResponseError as exc:
            raise HTTPException(502, "Triton returned an invalid inference response.") from exc
        except Exception as exc:
            raise HTTPException(500, "Classification failed.") from exc

    return app


app = create_app()
