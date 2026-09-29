"""Mocked Stage 10 gateway-to-SmolVLM HTTP checks; no model is loaded."""

from io import BytesIO
import os
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image
import requests

from src.stage10.api import create_app
from src.stage10.smolvlm_client import DESCRIBE_TIMEOUT, HEALTH_TIMEOUT, SmolVLMClient


def synthetic_image():
    output = BytesIO()
    Image.new("RGB", (12, 10), (80, 50, 30)).save(output, format="PNG")
    return output.getvalue()


def description_result(status="accepted"):
    rejected = status == "rejected"
    return {
        "model_identity": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "model_revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        "raw_generated_text": "This is melanoma." if rejected else " A brown patch has uneven edges. ",
        "validated_description": None if rejected else "A brown patch has uneven edges.",
        "validation_status": status,
        "rejection_reasons": ["diagnostic_or_disease_term"] if rejected else [],
        "prompt_contract_version": "stage10_visual_description_v1",
        "lora_active": False,
        "generated_token_ids": [1, 2, 3],
    }


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self.body = body

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class FakeTransport:
    def __init__(self):
        self.health = FakeResponse(body={"application": "ready", "smolvlm2_model": "ready"})
        self.description = FakeResponse(body=description_result())
        self.gets = []
        self.posts = []

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        if isinstance(self.health, Exception):
            raise self.health
        return self.health

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if isinstance(self.description, Exception):
            raise self.description
        return self.description


class FakeClassifier:
    def readiness(self):
        return True, True

    def classify(self, _image_bytes):
        raise AssertionError("/describe must not call the classifier")


class GatewayDescribeTests(unittest.TestCase):
    def setUp(self):
        self.transport = FakeTransport()
        self.smolvlm = SmolVLMClient("http://smolvlm.example", transport=self.transport)
        self.nemotron = Mock()
        self.nemotron.readiness.return_value = True
        self.nemotron.report.side_effect = AssertionError("/describe must not call Nemotron")
        self.client = TestClient(create_app(FakeClassifier(), self.smolvlm, self.nemotron))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def post_image(self):
        return self.client.post(
            "/describe", files={"image": ("synthetic.png", synthetic_image(), "image/png")},
        )

    def test_ready_health_uses_internal_health_only(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "application": "ready", "triton_server": "ready", "resnet18_fp16": "ready",
            "smolvlm2_service": "ready", "nemotron_report_service": "ready",
        })
        self.assertEqual(self.transport.gets, [
            ("http://smolvlm.example/health", {"timeout": HEALTH_TIMEOUT}),
        ])
        self.assertEqual(self.transport.posts, [])

    def test_unavailable_or_malformed_health_is_503_without_generation(self):
        for health in (
            FakeResponse(status_code=503),
            FakeResponse(body={"application": "ready", "smolvlm2_model": "unavailable"}),
            FakeResponse(body=ValueError("private invalid JSON")),
            requests.ConnectionError("private connection detail"),
        ):
            with self.subTest(health=health):
                self.transport.health = health
                response = self.client.get("/health")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["smolvlm2_service"], "unavailable")
                self.assertNotIn("private", response.text)
        self.assertEqual(self.transport.posts, [])

    def test_accepted_and_rejected_results_pass_through_exactly(self):
        for status in ("accepted", "rejected"):
            with self.subTest(status=status):
                expected = description_result(status)
                self.transport.description = FakeResponse(body=expected)
                response = self.post_image()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), expected)
                url, request = self.transport.posts[-1]
                self.assertEqual(url, "http://smolvlm.example/describe")
                self.assertEqual(request["timeout"], DESCRIBE_TIMEOUT)
                self.assertEqual(request["files"]["image"][1], synthetic_image())
        self.assertGreater(DESCRIBE_TIMEOUT[1], 10)

    def test_timeout_and_connection_failure_are_503(self):
        for failure in (requests.Timeout("private timeout"),
                        requests.ConnectionError("private connection")):
            with self.subTest(failure=failure):
                self.transport.description = failure
                response = self.post_image()
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("private", response.text)

    def test_malformed_upstream_responses_are_502(self):
        for body in (
            ValueError("private invalid JSON"),
            ["not an object"],
            {"raw_generated_text": "incomplete"},
            {**description_result("rejected"), "validated_description": "should be null"},
            {**description_result(), "generated_token_ids": [float("nan")]},
        ):
            with self.subTest(body=body):
                self.transport.description = FakeResponse(body=body)
                response = self.post_image()
                self.assertEqual(response.status_code, 502)
                self.assertNotIn("private", response.text)

    def test_upstream_non_2xx_is_503(self):
        for status in (400, 500, 503):
            with self.subTest(status=status):
                self.transport.description = FakeResponse(status_code=status, body={"private": "detail"})
                response = self.post_image()
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("private", response.text)

    def test_invalid_uploads_do_not_reach_service(self):
        for files in (
            {"image": ("bad.txt", b"not an image", "text/plain")},
            {"other": ("synthetic.png", synthetic_image(), "image/png")},
            [("image", ("one.png", synthetic_image(), "image/png")),
             ("image", ("two.png", synthetic_image(), "image/png"))],
        ):
            with self.subTest(files=files):
                self.assertEqual(self.client.post("/describe", files=files).status_code, 400)
        self.assertEqual(self.client.post("/describe", content=b"not multipart").status_code, 415)
        self.assertEqual(self.transport.posts, [])

    def test_service_url_is_required_configuration(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "SMOLVLM_HTTP_URL"):
            SmolVLMClient.from_env()

    def test_gateway_startup_requires_service_url(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "SMOLVLM_HTTP_URL"):
            with TestClient(create_app(FakeClassifier())):
                pass


if __name__ == "__main__":
    unittest.main()
