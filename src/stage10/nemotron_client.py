"""CPU-only HTTP client for the internal Stage 10 Nemotron report service."""

import json
import os
from urllib.parse import urlparse

import requests


HEALTH_TIMEOUT = (2, 5)  # connect, read (seconds)
REPORT_TIMEOUT = (3, 300)  # allow bounded time for autoregressive generation


class NemotronUnavailableError(RuntimeError):
    """The internal report request did not complete successfully."""


class NemotronResponseError(RuntimeError):
    """The internal service returned a malformed successful response."""


class NemotronClient:
    def __init__(self, base_url: str, *, transport=requests):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/"):
            raise ValueError("NEMOTRON_HTTP_URL must be an HTTP(S) base URL without a path.")
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    @classmethod
    def from_env(cls):
        base_url = os.environ.get("NEMOTRON_HTTP_URL")
        if not base_url:
            raise RuntimeError("NEMOTRON_HTTP_URL is required.")
        return cls(base_url)

    def readiness(self) -> bool:
        """Check only /health; never trigger report generation."""
        try:
            response = self.transport.get(f"{self.base_url}/health", timeout=HEALTH_TIMEOUT)
            if response.status_code != 200:
                return False
            body = response.json()
            return (isinstance(body, dict)
                    and body.get("application") == "ready"
                    and body.get("nemotron") == "ready"
                    and body.get("triton_server") == "ready"
                    and body.get("resnet18_fp16") == "ready")
        except (requests.RequestException, ValueError, TypeError):
            return False

    def report(self, image_bytes: bytes) -> dict:
        try:
            response = self.transport.post(
                f"{self.base_url}/report",
                files={"image": ("image", image_bytes, "application/octet-stream")},
                timeout=REPORT_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise NemotronUnavailableError("Nemotron report service is unavailable.") from exc
        if not 200 <= response.status_code < 300:
            raise NemotronUnavailableError("Nemotron report service returned an error.")
        try:
            body = response.json()
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise NemotronResponseError("Nemotron report service returned invalid JSON.") from exc
        if not self._is_report_result(body):
            raise NemotronResponseError("Nemotron report service returned an invalid result.")
        try:
            json.dumps(body, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise NemotronResponseError("Nemotron report service returned an invalid result.") from exc
        return body

    @staticmethod
    def _is_report_result(body) -> bool:
        """Check the transport envelope only; the report service owns validation."""
        if not isinstance(body, dict):
            return False
        if (not isinstance(body.get("classifier_evidence"), dict)
                or not isinstance(body.get("prompt_evidence"), dict)
                or not isinstance(body.get("deterministic_template_baseline"), dict)
                or not isinstance(body.get("report_contract_version"), str)
                or not isinstance(body.get("limitation"), str)):
            return False
        result = body.get("nemotron")
        if not isinstance(result, dict):
            return False
        if (not isinstance(result.get("model_provenance"), dict)
                or not isinstance(result.get("raw_generated_text"), str)):
            return False
        status = result.get("validation_status")
        if status == "accepted":
            return isinstance(result.get("validated_report"), dict) and result.get("rejection_reason") is None
        if status == "rejected":
            return result.get("validated_report", object()) is None and isinstance(result.get("rejection_reason"), str)
        return False
