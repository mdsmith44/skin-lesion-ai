"""Stage 10 ResNet18 classification through Triton's HTTP V2 binary API."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from urllib.parse import urlparse

import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
import torch

from src.stage9.resnet18_inference import CLASS_NAMES, stage4_eval_preprocess


MODEL_NAME = "resnet18_fp16"
MODEL_VERSION = "1"
EXPECTED_ENGINE_SHA256 = "7370bba2e48f106824f7f9a4d4ab6d065a5cf2b512311de6a47c95a5ac844cb3"
HTTP_TIMEOUT = (2, 10)  # connect, read (seconds)
SCORE_INTERPRETATION = "Uncalibrated model scores; not clinical probabilities."
INTENDED_USE = "Research and education only; not clinical diagnosis."


class InvalidImageError(ValueError):
    """The uploaded bytes cannot be decoded as an image."""


class TritonUnavailableError(RuntimeError):
    """The Triton inference request did not complete successfully."""


class TritonResponseError(RuntimeError):
    """Triton returned a response outside the pinned model contract."""


class TritonClassifier:
    def __init__(self, base_url: str, *, transport=requests):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/"):
            raise ValueError("TRITON_HTTP_URL must be an HTTP(S) base URL without a path.")
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.preprocess = stage4_eval_preprocess()
        self.model_url = f"{self.base_url}/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}"

    @classmethod
    def from_env(cls):
        base_url = os.environ.get("TRITON_HTTP_URL")
        if not base_url:
            raise RuntimeError("TRITON_HTTP_URL is required.")
        return cls(base_url)

    def readiness(self) -> tuple[bool, bool]:
        """Check Triton server and model readiness without running inference."""
        def ready(url):
            try:
                return self.transport.get(url, timeout=HTTP_TIMEOUT).status_code == 200
            except requests.RequestException:
                return False

        return (
            ready(f"{self.base_url}/v2/health/ready"),
            ready(f"{self.model_url}/ready"),
        )

    def classify(self, image_bytes: bytes) -> dict:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).contiguous()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("Uploaded file is not a readable image.") from exc

        if tensor.shape != (1, 3, 224, 224) or tensor.dtype != torch.float32:
            raise RuntimeError("Unexpected preprocessing output.")
        payload = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        header = json.dumps({
            "inputs": [{
                "name": "normalized_rgb", "shape": [1, 3, 224, 224],
                "datatype": "FP32", "parameters": {"binary_data_size": len(payload)},
            }],
            "outputs": [{"name": "logits", "parameters": {"binary_data": False}}],
        }, separators=(",", ":")).encode("utf-8")
        try:
            response = self.transport.post(
                f"{self.model_url}/infer", data=header + payload,
                headers={"Content-Type": "application/octet-stream",
                         "Inference-Header-Content-Length": str(len(header))},
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise TritonUnavailableError("Triton inference is unavailable.") from exc
        if response.status_code != 200:
            raise TritonUnavailableError("Triton inference is unavailable.")
        logits = self._parse_logits(response)
        scores = torch.softmax(torch.from_numpy(logits), dim=0).tolist()
        predicted_index = int(np.argmax(logits))
        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "predicted_class_index": predicted_index,
            "class_order": list(CLASS_NAMES),
            "softmax_scores": dict(zip(CLASS_NAMES, scores)),
            "score_type": "uncalibrated_softmax",
            "score_interpretation": SCORE_INTERPRETATION,
            "intended_use": INTENDED_USE,
            "input": {"image_sha256": hashlib.sha256(image_bytes).hexdigest()},
            "classifier_provenance": {
                "runtime": "Triton HTTP V2", "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION, "triton_base_url": self.base_url,
                "expected_engine_sha256": EXPECTED_ENGINE_SHA256,
                "preprocessing": "Stage 4 evaluation transform",
                "input_name": "normalized_rgb", "output_name": "logits",
            },
        }

    @staticmethod
    def _parse_logits(response) -> np.ndarray:
        try:
            result = response.json()
        except (ValueError, TypeError) as exc:
            raise TritonResponseError("Triton returned invalid JSON.") from exc
        if not isinstance(result, dict) or result.get("model_name") != MODEL_NAME or result.get("model_version") != MODEL_VERSION:
            raise TritonResponseError("Unexpected Triton model identity or version.")
        outputs = result.get("outputs")
        if (not isinstance(outputs, list) or len(outputs) != 1 or
                not isinstance(outputs[0], dict) or
                outputs[0].get("name") != "logits" or
                outputs[0].get("datatype") != "FP32" or
                outputs[0].get("shape") != [1, 7]):
            raise TritonResponseError("Unexpected Triton output contract.")
        data = outputs[0].get("data")
        if (not isinstance(data, list) or len(data) != 7 or
                any(isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in data)):
            raise TritonResponseError("Triton must return seven finite logits.")
        try:
            logits = np.asarray(data, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TritonResponseError("Triton must return seven finite FP32 logits.") from exc
        if not np.isfinite(logits).all():
            raise TritonResponseError("Triton must return seven finite FP32 logits.")
        return logits
