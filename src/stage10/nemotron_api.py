"""Research-only HTTP reporting backed by Triton and the frozen Stage 9 contract."""

from contextlib import asynccontextmanager
from io import BytesIO
import logging
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from src.stage9.reporting import LIMITATION, PROMPT_VERSION, template_report
from src.stage10.triton_classifier import (
    InvalidImageError, TritonClassifier, TritonResponseError, TritonUnavailableError,
)
from src.stage10.triton_report_adapter import InvalidClassifierEvidence, report_prompt_evidence


MAX_IMAGE_BYTES = 10 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


def _load_reporter():
    # Keep the large model import and CUDA initialization out of module import.
    from src.stage9.nemotron_reporting import NemotronReporter
    return NemotronReporter()


def create_app(classifier=None, reporter_factory=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier if classifier is not None else TritonClassifier.from_env()
        app.state.report_lock = threading.Lock()
        app.state.reporter = None
        try:
            app.state.reporter = await run_in_threadpool(reporter_factory or _load_reporter)
        except Exception:
            # A failed load is reflected in /health; no report is synthesized.
            LOGGER.exception("Nemotron reporter failed to load")
        yield

    app = FastAPI(title="Stage 10 grounded report service", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request):
        try:
            server_ready, model_ready = await run_in_threadpool(request.app.state.classifier.readiness)
        except Exception:
            server_ready, model_ready = False, False
        reporter_ready = request.app.state.reporter is not None
        body = {
            "application": "ready",
            "nemotron": "ready" if reporter_ready else "unavailable",
            "triton_server": "ready" if server_ready else "unavailable",
            "resnet18_fp16": "ready" if model_ready else "unavailable",
        }
        status = 200 if reporter_ready and server_ready and model_ready else 503
        return JSONResponse(body, status_code=status)

    @app.post("/report")
    async def report(request: Request):
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
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, "Uploaded file is not a readable image.") from exc

        reporter = request.app.state.reporter
        if reporter is None:
            raise HTTPException(503, "Nemotron reporter is unavailable.")
        try:
            classification = await run_in_threadpool(request.app.state.classifier.classify, image_bytes)
        except InvalidImageError as exc:
            raise HTTPException(400, "Uploaded file is not a readable image.") from exc
        except TritonUnavailableError as exc:
            raise HTTPException(503, "Triton classifier is unavailable.") from exc
        except TritonResponseError as exc:
            raise HTTPException(502, "Triton returned an invalid inference response.") from exc
        except Exception as exc:
            raise HTTPException(500, "Classification failed.") from exc
        try:
            evidence = report_prompt_evidence(classification, image_bytes)
        except InvalidClassifierEvidence as exc:
            raise HTTPException(502, "Classifier result is invalid for reporting.") from exc
        baseline = template_report(evidence)

        def generate():
            with request.app.state.report_lock:
                return reporter.report(evidence)

        try:
            generated = await run_in_threadpool(generate)
        except Exception as exc:
            raise HTTPException(503, "Nemotron generation failed.") from exc
        if (not isinstance(generated, dict)
                or generated.get("status") not in ("accepted", "rejected")
                or not isinstance(generated.get("raw_response"), str)
                or not isinstance(generated.get("reporter_provenance"), dict)
                or (generated["status"] == "accepted" and not isinstance(generated.get("report"), dict))
                or (generated["status"] == "rejected" and (generated.get("report") is not None
                    or not isinstance(generated.get("validation_error"), str)))):
            raise HTTPException(502, "Nemotron returned an invalid reporting result.")
        return {
            "classifier_evidence": classification,
            "prompt_evidence": evidence,
            "nemotron": {
                "model_provenance": generated["reporter_provenance"],
                "raw_generated_text": generated["raw_response"],
                "validated_report": generated["report"],
                "validation_status": generated["status"],
                "rejection_reason": generated.get("validation_error"),
            },
            "deterministic_template_baseline": baseline,
            "report_contract_version": PROMPT_VERSION,
            "limitation": LIMITATION,
        }

    return app


app = create_app()
