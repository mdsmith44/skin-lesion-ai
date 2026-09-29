"""Offline contract tests for the split-free Stage 10 report service."""

from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading
import time
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from src.stage9.reporting import LABELS, PROMPT_VERSION, prompt_evidence, template_report, validate_report
from src.stage10.nemotron_api import create_app
from src.stage10.triton_classifier import EXPECTED_ENGINE_SHA256, TritonUnavailableError


def image_bytes():
    output = BytesIO()
    Image.new("RGB", (8, 8), (120, 80, 40)).save(output, format="PNG")
    return output.getvalue()


def classification(data):
    scores = [0.05, 0.1, 0.61701, 0.02, 0.1, 0.1, 0.01299]
    return {
        "predicted_class": "bkl", "predicted_class_index": 2,
        "class_order": list(LABELS), "softmax_scores": dict(zip(LABELS, scores)),
        "score_type": "uncalibrated_softmax",
        "input": {"image_sha256": hashlib.sha256(data).hexdigest()},
        "classifier_provenance": {
            "runtime": "Triton HTTP V2", "model_name": "resnet18_fp16", "model_version": "1",
            "expected_engine_sha256": EXPECTED_ENGINE_SHA256,
        },
    }


class FakeClassifier:
    def __init__(self):
        self.ready = (True, True)
        self.failure = None
        self.mutate = None
        self.calls = 0
        self.readiness_calls = 0

    def readiness(self):
        self.readiness_calls += 1
        return self.ready

    def classify(self, data):
        self.calls += 1
        if self.failure:
            raise self.failure
        result = classification(data)
        if self.mutate:
            self.mutate(result)
        return result


class FakeReporter:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.reject = False

    def report(self, evidence):
        self.calls.append(evidence)
        if self.failure:
            raise self.failure
        raw = json.dumps(template_report(evidence)) if not self.reject else '{"diagnosis":"invented"}'
        provenance = {
            "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "revision": "bf77c3174f68ad409e1c2aa60daeb46e32d1c606",
            "prompt_version": PROMPT_VERSION,
        }
        try:
            report = validate_report(raw, evidence)
        except ValueError as exc:
            return {"status": "rejected", "report": None, "validation_error": str(exc),
                    "raw_response": raw, "reporter_provenance": provenance}
        return {"status": "accepted", "report": report, "raw_response": raw,
                "reporter_provenance": provenance}


class NemotronApiTests(unittest.TestCase):
    def setUp(self):
        self.data = image_bytes()
        self.classifier = FakeClassifier()
        self.reporter = FakeReporter()
        self.loads = 0

        def factory():
            self.loads += 1
            return self.reporter

        self.client = TestClient(create_app(self.classifier, factory))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def post_image(self, **kwargs):
        return self.client.post(
            "/report", files={"image": ("uploaded.png", self.data, "image/png")}, **kwargs,
        )

    def test_startup_once_and_health_does_not_generate(self):
        self.assertEqual(self.loads, 1)
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "application": "ready", "nemotron": "ready",
            "triton_server": "ready", "resnet18_fp16": "ready",
        })
        self.assertEqual(self.reporter.calls, [])
        self.assertEqual(self.classifier.calls, 0)
        self.classifier.ready = (True, False)
        self.assertEqual(self.client.get("/health").status_code, 503)
        self.assertEqual(self.reporter.calls, [])
        self.assertEqual(self.post_image().status_code, 200)
        self.assertEqual(self.post_image().status_code, 200)
        self.assertEqual(self.loads, 1)

    def test_accepted_report_exact_projection_and_separate_baseline(self):
        response = self.post_image()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        evidence = body["prompt_evidence"]
        self.assertEqual(evidence, self.reporter.calls[0])
        self.assertEqual(evidence, prompt_evidence(evidence))
        self.assertEqual(set(evidence), {"observed_metadata", "classifier"})
        self.assertRegex(evidence["observed_metadata"]["image_id"], r"^img_[0-9a-f]{16}$")
        self.assertNotIn("split", json.dumps(evidence))
        self.assertNotIn("smolvlm", json.dumps(evidence).lower())
        self.assertEqual(evidence["classifier"], {
            "predicted_class": "bkl", "top_softmax_score": "0.617010",
        })
        self.assertEqual(body["classifier_evidence"]["class_order"], list(LABELS))
        self.assertEqual(body["classifier_evidence"]["score_type"], "uncalibrated_softmax")
        self.assertEqual(body["nemotron"]["validation_status"], "accepted")
        self.assertEqual(body["nemotron"]["validated_report"], template_report(evidence))
        self.assertEqual(body["nemotron"]["validated_report"],
                         validate_report(body["nemotron"]["raw_generated_text"], evidence))
        self.assertIsNone(body["nemotron"]["rejection_reason"])
        self.assertEqual(body["deterministic_template_baseline"], template_report(evidence))
        self.assertEqual(body["report_contract_version"], PROMPT_VERSION)
        self.assertIn("not clinical diagnosis", body["limitation"])

    def test_rejected_generation_is_completed_inference_without_template_substitution(self):
        self.reporter.reject = True
        response = self.post_image()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["nemotron"]["raw_generated_text"], '{"diagnosis":"invented"}')
        self.assertEqual(body["nemotron"]["validation_status"], "rejected")
        self.assertIsNone(body["nemotron"]["validated_report"])
        self.assertIn("Report keys", body["nemotron"]["rejection_reason"])
        self.assertEqual(body["deterministic_template_baseline"],
                         template_report(body["prompt_evidence"]))

    def test_upload_rejects_nonimages_and_caller_supplied_evidence(self):
        invalid = [
            self.client.post("/report", content=b"not multipart"),
            self.client.post("/report", files={"image": ("bad.txt", b"garbage", "text/plain")}),
            self.client.post("/report", files={"other": ("uploaded.png", self.data, "image/png")}),
            self.client.post("/report", files=[("image", ("a.png", self.data, "image/png")),
                                                ("image", ("b.png", self.data, "image/png"))]),
            self.client.post("/report", files={"image": ("uploaded.png", self.data, "image/png")},
                             data={"split": "val", "diagnosis": "injected", "smolvlm": "injected"}),
        ]
        self.assertEqual([response.status_code for response in invalid], [415, 400, 400, 400, 400])
        self.assertEqual(self.classifier.calls, 0)
        self.assertEqual(self.reporter.calls, [])

    def test_triton_failure_and_invalid_trusted_evidence(self):
        self.classifier.failure = TritonUnavailableError("private Triton detail")
        response = self.post_image()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)
        self.classifier.failure = None
        mutations = [
            lambda result: result.update(class_order=list(reversed(LABELS))),
            lambda result: result["softmax_scores"].update(bkl=float("nan")),
            lambda result: result.update(predicted_class="mel"),
            lambda result: result.update(predicted_class_index=4),
            lambda result: result.update(score_type="clinical_probability"),
            lambda result: result["input"].update(image_sha256="bad"),
            lambda result: result["classifier_provenance"].update(model_version="2"),
            lambda result: result["classifier_provenance"].update(expected_engine_sha256="bad"),
        ]
        for mutation in mutations:
            self.classifier.mutate = mutation
            with self.subTest(mutation=mutation):
                response = self.post_image()
                self.assertEqual(response.status_code, 502)
        self.assertEqual(self.reporter.calls, [])

    def test_nemotron_failure_is_service_error(self):
        self.reporter.failure = RuntimeError("private model detail")
        response = self.post_image()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)

    def test_failed_model_load_makes_health_and_report_unavailable(self):
        def failed_load():
            raise RuntimeError("private model load detail")

        with TestClient(create_app(self.classifier, failed_load)) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["nemotron"], "unavailable")
            response = client.post("/report", files={"image": ("uploaded.png", self.data, "image/png")})
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private", response.text)

    def test_generation_is_serialized(self):
        active = 0
        peak = 0
        counter_lock = threading.Lock()
        original = self.reporter.report

        def observed_report(evidence):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.02)
                return original(evidence)
            finally:
                with counter_lock:
                    active -= 1

        self.reporter.report = observed_report
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: self.post_image().status_code, range(3)))
        self.assertEqual(results, [200, 200, 200])
        self.assertEqual(peak, 1)


if __name__ == "__main__":
    unittest.main()
