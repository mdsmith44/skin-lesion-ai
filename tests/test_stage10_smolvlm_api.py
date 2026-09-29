"""Offline HTTP tests for the Stage 10 visual-description service."""

from copy import deepcopy
from io import BytesIO
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from src.stage10.description_contract import CONTRACT_VERSION, GENERATION, PROMPT
from src.stage10.smolvlm_api import create_app
from src.stage10.smolvlm_description import MODEL_ID, MODEL_REVISION


def image_bytes():
    output = BytesIO()
    Image.new("RGBA", (13, 17), (90, 40, 20, 127)).save(output, format="PNG")
    return output.getvalue()


def description_result(raw, validated, status, reasons):
    return {
        "model_identity": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "lora_active": False,
        "raw_generated_text": raw,
        "validated_description": validated,
        "validation_status": status,
        "rejection_reasons": reasons,
        "prompt_contract_version": CONTRACT_VERSION,
        "prompt": PROMPT,
        "generation_settings": dict(GENERATION),
        "input": {"kind": "pil_image", "size": [13, 17]},
    }


class FakeDescriber:
    def __init__(self, result):
        self.result = result
        self.images = []

    def describe(self, image):
        self.images.append(image.copy())
        return deepcopy(self.result)


class SmolVLMApiTests(unittest.TestCase):
    def test_startup_loads_once_and_health_never_generates(self):
        fake = FakeDescriber(description_result("Brown area.", "Brown area.", "accepted", []))
        constructions = []

        def factory():
            constructions.append(1)
            return fake

        app = create_app(factory)
        self.assertEqual(constructions, [])
        with TestClient(app) as client:
            for _ in range(2):
                response = client.get("/health")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {
                    "application": "ready", "smolvlm2_model": "ready",
                })
            self.assertEqual(constructions, [1])
            self.assertEqual(fake.images, [])
            for _ in range(2):
                response = client.post(
                    "/describe", files={"image": ("sample.png", image_bytes(), "image/png")},
                )
                self.assertEqual(response.status_code, 200)
            self.assertEqual(constructions, [1])
            self.assertEqual(len(fake.images), 2)
            self.assertTrue(all(image.mode == "RGB" and image.size == (13, 17)
                                for image in fake.images))

    def test_accepted_and_rejected_results_are_preserved_exactly(self):
        accepted = description_result(" Brown area. ", "Brown area.", "accepted", [])
        rejected = description_result(
            "This may be melanoma.", None, "rejected", ["diagnostic_or_disease_term"],
        )
        fake = FakeDescriber(accepted)
        with TestClient(create_app(lambda: fake)) as client:
            for result in (accepted, rejected):
                fake.result = result
                response = client.post(
                    "/describe", files={"image": ("sample.png", image_bytes(), "image/png")},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), result)

    def test_invalid_uploads_are_rejected_before_generation(self):
        fake = FakeDescriber(description_result("Brown area.", "Brown area.", "accepted", []))
        with TestClient(create_app(lambda: fake)) as client:
            for files in (
                {"image": ("bad.txt", b"not an image", "text/plain")},
                {"image": ("empty.png", b"", "image/png")},
                {"other": ("sample.png", image_bytes(), "image/png")},
                [("image", ("one.png", image_bytes(), "image/png")),
                 ("image", ("two.png", image_bytes(), "image/png"))],
            ):
                with self.subTest(files=files):
                    self.assertEqual(client.post("/describe", files=files).status_code, 400)
            self.assertEqual(client.post("/describe", content=b"not multipart").status_code, 415)
            self.assertEqual(fake.images, [])

    def test_failed_load_reports_unavailable_without_generation(self):
        calls = []

        def failing_factory():
            calls.append(1)
            raise RuntimeError("private cache path")

        with TestClient(create_app(failing_factory)) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["smolvlm2_model"], "unavailable")
            self.assertNotIn("private", response.text)
            response = client.post(
                "/describe", files={"image": ("sample.png", image_bytes(), "image/png")},
            )
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private", response.text)
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
