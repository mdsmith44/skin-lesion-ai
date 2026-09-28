"""Synthetic, offline tests for the Stage 10 HTTP classifier."""

from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import requests
import torch
from torchvision import transforms

from src.stage9.resnet18_inference import CLASS_NAMES, stage4_eval_preprocess
from src.stage10.api import create_app
from src.stage10.triton_classifier import (
    InvalidImageError, TritonClassifier, TritonResponseError, TritonUnavailableError,
)


def synthetic_image():
    pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    output = BytesIO()
    Image.fromarray(pixels, "RGB").save(output, format="PNG")
    return output.getvalue()


def triton_result(logits=None):
    return {
        "model_name": "resnet18_fp16", "model_version": "1",
        "outputs": [{"name": "logits", "datatype": "FP32", "shape": [1, 7],
                     "data": logits if logits is not None else [0, 1, 4, -1, 2, 3, 0]}],
    }


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self.body = triton_result() if body is None else body

    def json(self):
        return self.body


class FakeTransport:
    def __init__(self, response=None, server_status=200, model_status=200):
        self.response = response or FakeResponse()
        self.server_status = server_status
        self.model_status = model_status
        self.posts = []
        self.gets = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        status = self.model_status if "/models/" in url else self.server_status
        if isinstance(status, Exception):
            raise status
        return FakeResponse(status_code=status)


class TritonClassifierTests(unittest.TestCase):
    def setUp(self):
        self.image_bytes = synthetic_image()
        self.transport = FakeTransport()
        self.classifier = TritonClassifier("http://triton.example", transport=self.transport)

    def test_stage4_transform_shape_dtype_and_parity_without_checkpoint(self):
        # The reference is the original Stage 4 torchvision sequence.
        reference = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with Image.open(BytesIO(self.image_bytes)) as image:
            rgb = image.convert("RGB")
            actual = stage4_eval_preprocess()(rgb)
            expected = reference(rgb)
        self.assertEqual(tuple(actual.shape), (3, 224, 224))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))
        with patch("src.stage9.resnet18_inference.ResNet18Inference", side_effect=AssertionError("checkpoint loaded")):
            self.classifier.classify(self.image_bytes)

    def test_binary_request_softmax_class_order_and_argmax(self):
        result = self.classifier.classify(self.image_bytes)
        self.assertEqual(result["predicted_class_index"], 2)
        self.assertEqual(result["predicted_class"], "bkl")
        self.assertEqual(result["class_order"], list(CLASS_NAMES))
        self.assertEqual(list(result["softmax_scores"]), list(CLASS_NAMES))
        expected = torch.softmax(torch.tensor([0, 1, 4, -1, 2, 3, 0], dtype=torch.float32), dim=0)
        np.testing.assert_allclose(list(result["softmax_scores"].values()), expected.tolist())
        self.assertAlmostEqual(sum(result["softmax_scores"].values()), 1)
        self.assertIn("not clinical probabilities", result["score_interpretation"])
        self.assertIn("not clinical diagnosis", result["intended_use"])
        self.assertEqual(result["classifier_provenance"]["model_version"], "1")

        url, request = self.transport.posts[0]
        self.assertEqual(url, "http://triton.example/v2/models/resnet18_fp16/versions/1/infer")
        self.assertEqual(request["timeout"], (2, 10))
        header_size = int(request["headers"]["Inference-Header-Content-Length"])
        header = json.loads(request["data"][:header_size])
        self.assertEqual(header["inputs"][0]["name"], "normalized_rgb")
        self.assertEqual(header["inputs"][0]["shape"], [1, 3, 224, 224])
        self.assertEqual(header["inputs"][0]["datatype"], "FP32")
        self.assertEqual(header["inputs"][0]["parameters"]["binary_data_size"], 1 * 3 * 224 * 224 * 4)
        sent = np.frombuffer(request["data"][header_size:], dtype="<f4").reshape(1, 3, 224, 224)
        with Image.open(BytesIO(self.image_bytes)) as image:
            expected_tensor = stage4_eval_preprocess()(image.convert("RGB")).unsqueeze(0).numpy()
        np.testing.assert_array_equal(sent, expected_tensor)

    def test_bad_upload_and_seven_logit_contract(self):
        with self.assertRaises(InvalidImageError):
            self.classifier.classify(b"not an image")
        self.assertEqual(self.transport.posts, [])
        for body in (
            triton_result([0] * 6), triton_result([0] * 8),
            triton_result([0, 1, 2, 3, 4, 5, float("nan")]),
            triton_result([0, 1, 2, 3, 4, 5, 10 ** 1000]),
            triton_result([0, 1, 2, 3, 4, 5, "6"]),
            {**triton_result(), "model_version": "2"},
            {**triton_result(), "outputs": [{**triton_result()["outputs"][0], "shape": [7]}]},
        ):
            self.transport.response = FakeResponse(body=body)
            with self.subTest(body=body), self.assertRaises(TritonResponseError):
                self.classifier.classify(self.image_bytes)
        self.transport.response = FakeResponse(body=["invalid"])
        with self.assertRaises(TritonResponseError):
            self.classifier.classify(self.image_bytes)

    def test_upstream_http_and_connection_errors(self):
        for response in (FakeResponse(status_code=503), requests.Timeout("private detail")):
            self.transport.response = response
            with self.assertRaises(TritonUnavailableError):
                self.classifier.classify(self.image_bytes)

    def test_production_configuration_requires_endpoint(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "TRITON_HTTP_URL"):
            TritonClassifier.from_env()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.transport = FakeTransport()
        self.classifier = TritonClassifier("http://triton.example", transport=self.transport)
        self.client = TestClient(create_app(self.classifier))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_ready_and_unavailable_health_without_inference(self):
        ready = self.client.get("/health")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json(), {
            "application": "ready", "triton_server": "ready", "resnet18_fp16": "ready",
        })
        self.transport.model_status = 503
        unavailable = self.client.get("/health")
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.json()["application"], "ready")
        self.assertEqual(unavailable.json()["triton_server"], "ready")
        self.assertEqual(unavailable.json()["resnet18_fp16"], "unavailable")
        self.transport.server_status = requests.ConnectionError("private detail")
        unavailable = self.client.get("/health")
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.json()["triton_server"], "unavailable")
        self.assertEqual(self.transport.posts, [])
        self.assertTrue(all(request["timeout"] == (2, 10) for _, request in self.transport.gets))

    def test_classify_and_invalid_uploads(self):
        response = self.client.post("/classify", files={"image": ("synthetic.png", synthetic_image(), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predicted_class"], "bkl")
        for files in (
            {"image": ("bad.txt", b"garbage", "text/plain")},
            {"other": ("synthetic.png", synthetic_image(), "image/png")},
            [("image", ("one.png", synthetic_image(), "image/png")),
             ("image", ("two.png", synthetic_image(), "image/png"))],
        ):
            with self.subTest(files=files):
                response = self.client.post("/classify", files=files)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.post("/classify", content=b"not multipart").status_code, 415)

    def test_upstream_errors_do_not_expose_private_details(self):
        self.transport.response = FakeResponse(body={"private": "data"})
        response = self.client.post("/classify", files={"image": ("one.png", synthetic_image(), "image/png")})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private", response.text)
        self.transport.response = requests.Timeout("private timeout detail")
        response = self.client.post("/classify", files={"image": ("one.png", synthetic_image(), "image/png")})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)


if __name__ == "__main__":
    unittest.main()
