"""CPU-only HTTP client for the internal Stage 10 SmolVLM description service."""

import json
import os
from urllib.parse import urlparse

import requests


HEALTH_TIMEOUT = (2, 5)  # connect, read (seconds)
DESCRIBE_TIMEOUT = (3, 180)  # generation is autoregressive


class SmolVLMUnavailableError(RuntimeError):
    """The internal description request could not complete."""


class SmolVLMResponseError(RuntimeError):
    """The internal service returned an invalid response."""


class SmolVLMClient:
    def __init__(self, base_url: str, *, transport=requests):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/"):
            raise ValueError("SMOLVLM_HTTP_URL must be an HTTP(S) base URL without a path.")
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    @classmethod
    def from_env(cls):
        base_url = os.environ.get("SMOLVLM_HTTP_URL")
        if not base_url:
            raise RuntimeError("SMOLVLM_HTTP_URL is required.")
        return cls(base_url)

    def readiness(self) -> bool:
        """Check the service's loaded-model status without generating text."""
        try:
            response = self.transport.get(f"{self.base_url}/health", timeout=HEALTH_TIMEOUT)
            if response.status_code != 200:
                return False
            body = response.json()
            return (isinstance(body, dict) and body.get("application") == "ready"
                    and body.get("smolvlm2_model") == "ready")
        except (requests.RequestException, ValueError, TypeError):
            return False

    def describe(self, image_bytes: bytes) -> dict:
        try:
            response = self.transport.post(
                f"{self.base_url}/describe",
                files={"image": ("image", image_bytes, "application/octet-stream")},
                timeout=DESCRIBE_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise SmolVLMUnavailableError("SmolVLM description service is unavailable.") from exc
        if not 200 <= response.status_code < 300:
            raise SmolVLMUnavailableError("SmolVLM description service returned an error.")
        try:
            body = response.json()
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise SmolVLMResponseError("SmolVLM description service returned invalid JSON.") from exc
        if not self._is_description_result(body):
            raise SmolVLMResponseError("SmolVLM description service returned an invalid result.")
        try:
            json.dumps(body, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SmolVLMResponseError("SmolVLM description service returned an invalid result.") from exc
        return body

    @staticmethod
    def _is_description_result(body) -> bool:
        """Check transport structure only; the upstream validator owns the text contract."""
        if not isinstance(body, dict):
            return False
        for key in ("model_identity", "model_revision", "raw_generated_text",
                    "prompt_contract_version"):
            if not isinstance(body.get(key), str):
                return False
        status = body.get("validation_status")
        reasons = body.get("rejection_reasons")
        validated = body.get("validated_description", object())
        if status not in ("accepted", "rejected") or not isinstance(reasons, list):
            return False
        if not all(isinstance(reason, str) for reason in reasons):
            return False
        return (isinstance(validated, str) if status == "accepted" else validated is None)
