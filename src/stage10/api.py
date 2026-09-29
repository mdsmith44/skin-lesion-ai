"""Research-only Stage 10 gateway for classification, descriptions, and reports."""

from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from src.stage10.nemotron_client import (
    NemotronClient, NemotronResponseError, NemotronUnavailableError,
)
from src.stage10.smolvlm_client import (
    SmolVLMClient, SmolVLMResponseError, SmolVLMUnavailableError,
)
from src.stage10.triton_classifier import (
    InvalidImageError, TritonClassifier, TritonResponseError, TritonUnavailableError,
)


MAX_IMAGE_BYTES = 10 * 1024 * 1024


def create_app(classifier: TritonClassifier | None = None,
               smolvlm_client: SmolVLMClient | None = None,
               nemotron_client: NemotronClient | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail startup if the required endpoint has not been configured.
        app.state.classifier = classifier if classifier is not None else TritonClassifier.from_env()
        app.state.smolvlm_client = (
            smolvlm_client if smolvlm_client is not None else SmolVLMClient.from_env()
        )
        app.state.nemotron_client = (
            nemotron_client if nemotron_client is not None else NemotronClient.from_env()
        )
        yield

    app = FastAPI(title="Stage 10 research gateway", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request):
        try:
            server_ready, model_ready = await run_in_threadpool(request.app.state.classifier.readiness)
        except Exception:
            server_ready, model_ready = False, False
        try:
            smolvlm_ready = await run_in_threadpool(request.app.state.smolvlm_client.readiness)
        except Exception:
            smolvlm_ready = False
        try:
            nemotron_ready = await run_in_threadpool(request.app.state.nemotron_client.readiness)
        except Exception:
            nemotron_ready = False
        body = {
            "application": "ready",
            "triton_server": "ready" if server_ready else "unavailable",
            "resnet18_fp16": "ready" if model_ready else "unavailable",
            "smolvlm2_service": "ready" if smolvlm_ready else "unavailable",
            "nemotron_report_service": "ready" if nemotron_ready else "unavailable",
        }
        ready = server_ready and model_ready and smolvlm_ready and nemotron_ready
        return JSONResponse(body, status_code=200 if ready else 503)

    async def read_image_upload(request: Request) -> bytes:
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
        return image_bytes

    @app.post("/classify")
    async def classify(request: Request):
        image_bytes = await read_image_upload(request)
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

    @app.post("/describe")
    async def describe(request: Request):
        image_bytes = await read_image_upload(request)
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, "Uploaded file is not a readable image.") from exc
        try:
            return await run_in_threadpool(request.app.state.smolvlm_client.describe, image_bytes)
        except SmolVLMUnavailableError as exc:
            raise HTTPException(503, "SmolVLM description service is unavailable.") from exc
        except SmolVLMResponseError as exc:
            raise HTTPException(502, "SmolVLM description service returned an invalid response.") from exc
        except Exception as exc:
            raise HTTPException(500, "Description failed.") from exc

    @app.post("/report")
    async def report(request: Request):
        image_bytes = await read_image_upload(request)
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, "Uploaded file is not a readable image.") from exc
        try:
            return await run_in_threadpool(request.app.state.nemotron_client.report, image_bytes)
        except NemotronUnavailableError as exc:
            raise HTTPException(503, "Nemotron report service is unavailable.") from exc
        except NemotronResponseError as exc:
            raise HTTPException(502, "Nemotron report service returned an invalid response.") from exc
        except Exception as exc:
            raise HTTPException(500, "Reporting failed.") from exc

    return app


app = create_app()
