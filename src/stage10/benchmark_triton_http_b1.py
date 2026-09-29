"""Sequential batch-1 Triton HTTP benchmark on one preprocessed val image."""

import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from PIL import Image
import requests
import torch
import torchvision

from src.stage9.resnet18_inference import ResNet18Inference
from src.stage10.export_resnet18_onnx import sha256
from src.stage10.verify_triton_resnet18_fp16 import (
    ENDPOINT, ENGINE_SHA256, METADATA, OUTPUT_DIR,
)


WARMUP_REQUESTS = 100
MEASURED_REQUESTS = 300
MODEL_REPOSITORY = Path("/workspace/storage/skin-lesion-ai/triton/model_repository")


def main():
    prior = json.loads(METADATA.read_text())
    validation = prior["comparison"]
    image_id = validation["validation_image_id"]
    if image_id != "ISIC_0027419" or validation["reference_diagnosis_used"]:
        raise ValueError("Unexpected validation image provenance")
    image_path = Path(validation["image_path"]).resolve(strict=True)
    if image_path.name != image_id + ".jpg" or sha256(image_path) != validation["image_sha256"]:
        raise ValueError("Validation image does not match the prior check")
    plan = MODEL_REPOSITORY / "resnet18_fp16/1/model.plan"
    engine_hash = sha256(plan)
    if engine_hash != ENGINE_SHA256 or engine_hash != prior["engine_sha256"]:
        raise ValueError("Staged FP16 plan hash mismatch")
    config = MODEL_REPOSITORY / "resnet18_fp16/config.pbtxt"
    if "dynamic_batching" in config.read_text():
        raise ValueError("Triton dynamic batching must remain disabled")

    classifier = ResNet18Inference(device="cpu")
    if classifier._provenance["checkpoint_sha256"] != validation["source_checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    with Image.open(image_path) as image:
        inputs = classifier._preprocess(image.convert("RGB")).unsqueeze(0).contiguous()
    if inputs.shape != (1, 3, 224, 224) or inputs.dtype != torch.float32:
        raise ValueError("Unexpected preprocessed tensor")

    # Match verify_triton_resnet18_fp16: FP32 row-major bytes after a JSON header.
    payload = inputs.numpy().astype("<f4", copy=False).tobytes(order="C")
    header = json.dumps({
        "inputs": [{
            "name": "normalized_rgb", "shape": [1, 3, 224, 224],
            "datatype": "FP32", "parameters": {"binary_data_size": len(payload)},
        }],
        "outputs": [{"name": "logits", "parameters": {"binary_data": False}}],
    }, separators=(",", ":")).encode("utf-8")
    body = header + payload
    headers = {"Content-Type": "application/octet-stream",
               "Inference-Header-Content-Length": str(len(header))}
    if len(payload) != 1 * 3 * 224 * 224 * 4:
        raise ValueError("Unexpected FP32 payload size")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    result_path = OUTPUT_DIR / ("single_client_http_b1_" +
                                timestamp.strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    record = {
        "status": "running", "timestamp_utc": timestamp.isoformat(),
        "endpoint": ENDPOINT, "model": "resnet18_fp16", "version": None,
        "engine_sha256": engine_hash,
        "model_config_sha256": sha256(config),
        "image_id": image_id, "image_sha256": validation["image_sha256"],
        "preprocessing": classifier._provenance["preprocessing"],
        "checkpoint_sha256": classifier._provenance["checkpoint_sha256"],
        "parameters": {
            "batch_size": 1, "sequential_clients": 1,
            "warmup_requests": WARMUP_REQUESTS,
            "measured_requests": MEASURED_REQUESTS,
            "input_shape": [1, 3, 224, 224], "input_dtype": "FP32",
            "input_bytes": len(payload),
            "request_format": "HTTP V2 binary FP32 input; JSON logits output",
            "connection": "one requests.Session; sequential POSTs; no client retries",
            "timing": "perf_counter_ns around each HTTP POST through response-body receipt; excludes image I/O and preprocessing",
            "throughput": "measured requests / wall-clock seconds from before first to after last measured request",
            "dynamic_batching": False,
        },
        "environment": {
            "client_hostname": socket.gethostname(),
            "runai_project": os.environ.get("RUNAI_PROJECT"),
            "runai_job_name": os.environ.get("RUNAI_JOB_NAME"),
            "kubernetes_namespace": Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read_text().strip(),
            "machine": platform.machine(), "python": platform.python_version(),
            "pytorch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_from_pytorch": torch.version.cuda,
            "requests": requests.__version__, "numpy": np.__version__,
        },
        "http_status_counts": {}, "errors": [],
        "warmup_completed": 0, "measured_completed": 0,
    }

    phase, iteration = "warmup", 0
    samples_ns = []
    try:
        with requests.Session() as session:
            for phase, count in (("warmup", WARMUP_REQUESTS),
                                 ("measured", MEASURED_REQUESTS)):
                if phase == "measured":
                    measured_wall_start_ns = perf_counter_ns()
                for iteration in range(count):
                    start_ns = perf_counter_ns()
                    response = session.post(ENDPOINT, data=body, headers=headers, timeout=30)
                    elapsed_ns = perf_counter_ns() - start_ns
                    code = str(response.status_code)
                    record["http_status_counts"][code] = record["http_status_counts"].get(code, 0) + 1
                    response.raise_for_status()
                    result = response.json()
                    outputs = result.get("outputs", [])
                    if (result.get("model_name") != "resnet18_fp16"
                            or result.get("model_version") != "1"
                            or len(outputs) != 1
                            or outputs[0].get("name") != "logits"
                            or outputs[0].get("datatype") != "FP32"
                            or outputs[0].get("shape") != [1, 7]
                            or len(outputs[0].get("data", [])) != 7):
                        raise ValueError("Unexpected Triton response contract")
                    record["version"] = result["model_version"]
                    record[phase + "_completed"] += 1
                    if phase == "measured":
                        samples_ns.append(elapsed_ns)
                if phase == "measured":
                    measured_wall_end_ns = perf_counter_ns()

        latencies_ms = np.asarray(samples_ns, dtype=np.float64) / 1_000_000
        wall_seconds = (measured_wall_end_ns - measured_wall_start_ns) / 1_000_000_000
        if len(latencies_ms) != MEASURED_REQUESTS or wall_seconds <= 0:
            raise RuntimeError("Incomplete benchmark")
        record.update(
            status="passed",
            median_latency_ms=float(np.median(latencies_ms)),
            mean_latency_ms=float(np.mean(latencies_ms)),
            p95_latency_ms=float(np.percentile(latencies_ms, 95)),
            p99_latency_ms=float(np.percentile(latencies_ms, 99)),
            min_latency_ms=float(np.min(latencies_ms)),
            max_latency_ms=float(np.max(latencies_ms)),
            measured_wall_seconds=wall_seconds,
            requests_per_second=MEASURED_REQUESTS / wall_seconds,
            latency_samples_ms=latencies_ms.tolist(),
        )
    except Exception as exc:
        record.update(status="failed", errors=[{
            "phase": phase, "iteration": iteration,
            "error": f"{type(exc).__name__}: {exc}",
        }])
        raise
    finally:
        with result_path.open("x") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
        print(json.dumps({"result_path": str(result_path),
                          **{key: value for key, value in record.items()
                             if key not in {"latency_samples_ms", "preprocessing", "environment"}}},
                         indent=2), flush=True)


if __name__ == "__main__":
    main()
