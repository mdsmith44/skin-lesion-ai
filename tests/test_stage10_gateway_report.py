"""Mocked gateway-to-Nemotron HTTP checks; no models or services are used."""

from io import BytesIO
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import requests

from src.stage10.api import create_app
from src.stage10.nemotron_client import HEALTH_TIMEOUT, REPORT_TIMEOUT, NemotronClient


def synthetic_image():
    output = BytesIO()
    Image.new("RGB", (12, 10), (80, 50, 30)).save(output, format="PNG")
    return output.getvalue()


def report_result(status="accepted"):
    rejected = status == "rejected"
    report = {"image_id": "img_0123456789abcdef", "predicted_class": "bkl",
              "top_softmax_score": "0.617010", "summary": "synthetic baseline",
              "unknowns": ["visual_findings", "clinical_diagnosis", "patient_history"],
              "limitation": "Research and education only; not clinical diagnosis. Scores are uncalibrated."}
    return {
        "classifier_evidence": {"predicted_class": "bkl", "class_order": [
            "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]},
        "prompt_evidence": {"observed_metadata": {"image_id": report["image_id"]},
                            "classifier": {"predicted_class": "bkl", "top_softmax_score": "0.617010"}},
        "nemotron": {
            "model_provenance": {"model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                                 "revision": "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"},
            "raw_generated_text": "{bad json}" if rejected else " raw accepted JSON ",
            "validated_report": None if rejected else report,
            "validation_status": status,
            "rejection_reason": "Report keys do not match the closed schema." if rejected else None,
        },
        "deterministic_template_baseline": report,
        "report_contract_version": "stage9-grounded-v1",
        "limitation": report["limitation"],
        "upstream_extension": {"preserve": True},
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
        self.health = FakeResponse(body={"application": "ready", "nemotron": "ready",
                                         "triton_server": "ready", "resnet18_fp16": "ready"})
        self.result = FakeResponse(body=report_result())
        self.gets = []
        self.posts = []

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        if isinstance(self.health, Exception):
            raise self.health
        return self.health

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClassifier:
    def readiness(self):
        return True, True

    def classify(self, _image_bytes):
        return {"predicted_class": "bkl", "predicted_class_index": 2}


class FakeSmolVLM:
    def readiness(self):
        return True

    def describe(self, _image_bytes):
        return {"validation_status": "rejected", "validated_description": None,
                "raw_generated_text": "synthetic raw description"}


class GatewayReportTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_image()
        self.transport = FakeTransport()
        self.nemotron = NemotronClient("http://nemotron.example", transport=self.transport)
        self.client = TestClient(create_app(FakeClassifier(), FakeSmolVLM(), self.nemotron))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def post_image(self):
        return self.client.post("/report", files={"image": ("synthetic.png", self.data, "image/png")})

    def test_required_service_url_at_startup(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "NEMOTRON_HTTP_URL"):
            with TestClient(create_app(FakeClassifier(), FakeSmolVLM())):
                pass

    def test_health_checks_all_five_fields_without_generation(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "application": "ready", "triton_server": "ready", "resnet18_fp16": "ready",
            "smolvlm2_service": "ready", "nemotron_report_service": "ready",
        })
        self.assertEqual(self.transport.gets, [
            ("http://nemotron.example/health", {"timeout": HEALTH_TIMEOUT}),
        ])
        self.assertEqual(self.transport.posts, [])

    def test_unavailable_health_is_503_without_generation(self):
        for health in (
            FakeResponse(status_code=503),
            FakeResponse(body={"application": "ready", "nemotron": "unavailable",
                               "triton_server": "ready", "resnet18_fp16": "ready"}),
            FakeResponse(body=ValueError("private malformed health")),
            requests.ConnectionError("private connection"),
        ):
            with self.subTest(health=health):
                self.transport.health = health
                response = self.client.get("/health")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["nemotron_report_service"], "unavailable")
                self.assertNotIn("private", response.text)
        self.assertEqual(self.transport.posts, [])

    def test_successful_accepted_and_rejected_results_pass_through_unchanged(self):
        for status in ("accepted", "rejected"):
            with self.subTest(status=status):
                expected = report_result(status)
                self.transport.result = FakeResponse(body=expected)
                response = self.post_image()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), expected)
                url, request = self.transport.posts[-1]
                self.assertEqual(url, "http://nemotron.example/report")
                self.assertEqual(request["timeout"], REPORT_TIMEOUT)
                self.assertEqual(set(request), {"files", "timeout"})
                self.assertEqual(set(request["files"]), {"image"})
                self.assertEqual(request["files"]["image"][1], self.data)
        self.assertGreater(REPORT_TIMEOUT[1], 180)

    def test_caller_cannot_inject_evidence_or_extra_files(self):
        response = self.client.post(
            "/report", files={"image": ("synthetic.png", self.data, "image/png")},
            data={"predicted_class": "mel", "scores": "1", "split": "val",
                  "diagnosis": "injected", "smolvlm_output": "injected",
                  "prompt_evidence": "injected", "report": "injected"},
        )
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/report", files=[
            ("image", ("a.png", self.data, "image/png")),
            ("image", ("b.png", self.data, "image/png")),
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.transport.posts, [])

    def test_malformed_uploads_are_4xx(self):
        self.assertEqual(self.client.post("/report", content=b"not multipart").status_code, 415)
        self.assertEqual(self.client.post("/report", files={"image": ("bad.txt", b"bad", "text/plain")}).status_code, 400)
        self.assertEqual(self.client.post("/report", files={"other": ("a.png", self.data, "image/png")}).status_code, 400)
        self.assertEqual(self.client.post("/report", files={"image": ("empty.png", b"", "image/png")}).status_code, 400)
        self.assertEqual(self.transport.posts, [])

    def test_connection_timeout_and_upstream_http_failures_are_503(self):
        for failure in (
            requests.ConnectionError("private connection"),
            requests.Timeout("private timeout"),
            FakeResponse(status_code=500, body={"private": "internal"}),
            FakeResponse(status_code=503, body={"private": "internal"}),
        ):
            with self.subTest(failure=failure):
                self.transport.result = failure
                response = self.post_image()
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("private", response.text)

    def test_malformed_successful_responses_are_502(self):
        for body in (
            ValueError("private invalid JSON"),
            ["not an object"],
            {"nemotron": {}},
            {**report_result("rejected"), "nemotron": {**report_result("rejected")["nemotron"],
                                                         "validated_report": {"unexpected": True}}},
            {**report_result(), "upstream_extension": {"bad": float("nan")}},
        ):
            with self.subTest(body=body):
                self.transport.result = FakeResponse(body=body)
                response = self.post_image()
                self.assertEqual(response.status_code, 502)
                self.assertNotIn("private", response.text)

    def test_existing_classify_and_describe_paths_remain_separate(self):
        files = {"image": ("synthetic.png", self.data, "image/png")}
        classified = self.client.post("/classify", files=files)
        described = self.client.post("/describe", files=files)
        self.assertEqual(classified.status_code, 200)
        self.assertEqual(classified.json(), {"predicted_class": "bkl", "predicted_class_index": 2})
        self.assertEqual(described.status_code, 200)
        self.assertEqual(described.json()["raw_generated_text"], "synthetic raw description")
        self.assertEqual(self.transport.posts, [])


if __name__ == "__main__":
    unittest.main()
