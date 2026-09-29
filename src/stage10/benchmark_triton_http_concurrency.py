"""Benchmark concurrent batch-1 Triton HTTP requests on one val tensor."""

import argparse
import hashlib
import json
import os
import platform
import socket
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier, Event
from time import perf_counter_ns

import numpy as np
import PIL
from PIL import Image
import requests
import torch
import torchvision

from src.stage9.resnet18_inference import ResNet18Inference
from src.stage10.export_resnet18_onnx import sha256
from src.stage10.verify_triton_resnet18_fp16 import (
    ENDPOINT, ENGINE_SHA256, METADATA, OUTPUT_DIR,
)


CONCURRENCIES = (1, 2, 4, 8, 16, 32)
WARMUP_REQUESTS = 100
MIN_MEASURED_REQUESTS = 300
MIN_MEASURED_PER_WORKER = 20
MODEL_REPOSITORY = Path("/workspace/storage/skin-lesion-ai/triton/model_repository")


def distribute(total, workers):
    return [total // workers + (worker < total % workers) for worker in range(workers)]


def worker(warmups, measurements, body, headers, ready, start, endpoint):
    result = {"warmup_successful": 0, "warmup_failed": 0,
              "measured_successful": 0, "measured_failed": 0,
              "status_counts": Counter(), "errors": [], "latencies_ns": [], "version": None}
    with requests.Session() as session:
        for phase, count in (("warmup", warmups), ("measured", measurements)):
            if phase == "measured":
                ready.wait()
                start.wait()
            for index in range(count):
                response = None
                begin_ns = perf_counter_ns()
                try:
                    response = session.post(endpoint, data=body, headers=headers, timeout=30)
                    elapsed_ns = perf_counter_ns() - begin_ns
                    result["status_counts"][str(response.status_code)] += 1
                    response.raise_for_status()
                    data = response.json()
                    outputs = data.get("outputs", [])
                    if (data.get("model_name") != "resnet18_fp16"
                            or data.get("model_version") != "1"
                            or len(outputs) != 1
                            or outputs[0].get("name") != "logits"
                            or outputs[0].get("datatype") != "FP32"
                            or outputs[0].get("shape") != [1, 7]
                            or len(outputs[0].get("data", [])) != 7):
                        raise ValueError("Unexpected Triton response contract")
                    result["version"] = data["model_version"]
                    result[phase + "_successful"] += 1
                    if phase == "measured":
                        result["latencies_ns"].append(elapsed_ns)
                except (requests.RequestException, ValueError) as exc:
                    if response is None:
                        result["status_counts"]["no_response"] += 1
                    result[phase + "_failed"] += 1
                    if len(result["errors"]) < 10:
                        result["errors"].append({"phase": phase, "index": index,
                                                 "error": f"{type(exc).__name__}: {exc}"})
    return result


def benchmark_level(concurrency, body, headers, endpoint):
    measured = max(MIN_MEASURED_REQUESTS, MIN_MEASURED_PER_WORKER * concurrency)
    warmup_alloc = distribute(WARMUP_REQUESTS, concurrency)
    measured_alloc = distribute(measured, concurrency)
    ready = Barrier(concurrency + 1)
    start = Event()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, warmup_alloc[i], measured_alloc[i],
                               body, headers, ready, start, endpoint) for i in range(concurrency)]
        ready.wait(timeout=120)
        wall_start_ns = perf_counter_ns()
        start.set()
        results = [future.result() for future in futures]
        wall_end_ns = perf_counter_ns()

    successes = sum(item["measured_successful"] for item in results)
    failures = sum(item["measured_failed"] for item in results)
    latencies_ms = np.asarray([ns for item in results for ns in item["latencies_ns"]],
                              dtype=np.float64) / 1_000_000
    wall_seconds = (wall_end_ns - wall_start_ns) / 1_000_000_000
    statuses = Counter()
    for item in results:
        statuses.update(item["status_counts"])
    row = {
        "concurrency": concurrency, "batch_size": 1,
        "warmup_requests": WARMUP_REQUESTS,
        "warmup_successful": sum(item["warmup_successful"] for item in results),
        "warmup_failed": sum(item["warmup_failed"] for item in results),
        "measured_requests": measured,
        "measured_requests_per_worker": measured_alloc,
        "successful_requests": successes, "failed_requests": failures,
        "http_status_counts": dict(statuses),
        "errors": [error for item in results for error in item["errors"]][:20],
        "model_versions_observed": sorted({item["version"] for item in results if item["version"]}),
        "measured_wall_seconds": wall_seconds,
        "aggregate_requests_per_second": successes / wall_seconds,
        "effective_images_per_second": successes / wall_seconds,
        "latency_samples_ms": latencies_ms.tolist(),
    }
    if successes:
        row.update(
            median_latency_ms=float(np.median(latencies_ms)),
            mean_latency_ms=float(np.mean(latencies_ms)),
            p95_latency_ms=float(np.percentile(latencies_ms, 95)),
            p99_latency_ms=float(np.percentile(latencies_ms, 99)),
            min_latency_ms=float(np.min(latencies_ms)),
            max_latency_ms=float(np.max(latencies_ms)),
        )
    if (row["warmup_successful"] + row["warmup_failed"] != WARMUP_REQUESTS
            or successes + failures != measured
            or len(latencies_ms) != successes):
        raise RuntimeError("Request counts do not match the benchmark plan")
    return row


def main(endpoint=ENDPOINT, expect_dynamic=False):
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
    config_text = config.read_text()
    staged_dynamic = "dynamic_batching" in config_text
    if staged_dynamic != expect_dynamic:
        raise ValueError("Staged dynamic-batching mode does not match requested run")
    config_url = endpoint.removesuffix("/infer") + "/config"
    config_response = requests.get(config_url, timeout=10)
    config_response.raise_for_status()
    server_config = config_response.json()
    dynamic_settings = server_config.get("dynamic_batching")
    if (server_config.get("name") != "resnet18_fp16"
            or server_config.get("max_batch_size") != 32
            or (dynamic_settings is not None) != expect_dynamic):
        raise ValueError("Triton model config does not match requested run")
    if expect_dynamic and (dynamic_settings.get("preferred_batch_size") != [8, 16, 32]
                           or dynamic_settings.get("max_queue_delay_microseconds") != 1000):
        raise ValueError("Unexpected Triton dynamic-batching settings")

    classifier = ResNet18Inference(device="cpu")
    if classifier._provenance["checkpoint_sha256"] != validation["source_checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    with Image.open(image_path) as image:
        inputs = classifier._preprocess(image.convert("RGB")).unsqueeze(0).contiguous()
    if inputs.shape != (1, 3, 224, 224) or inputs.dtype != torch.float32:
        raise ValueError("Unexpected preprocessed tensor")
    payload = inputs.numpy().astype("<f4", copy=False).tobytes(order="C")
    if len(payload) != 1 * 3 * 224 * 224 * 4:
        raise ValueError("Unexpected FP32 payload size")
    # Same binary HTTP V2 envelope as verify_triton_resnet18_fp16.py.
    header = json.dumps({
        "inputs": [{"name": "normalized_rgb", "shape": [1, 3, 224, 224],
                    "datatype": "FP32", "parameters": {"binary_data_size": len(payload)}}],
        "outputs": [{"name": "logits", "parameters": {"binary_data": False}}],
    }, separators=(",", ":")).encode("utf-8")
    body = header + payload
    headers = {"Content-Type": "application/octet-stream",
               "Inference-Header-Content-Length": str(len(header))}

    timestamp = datetime.now(timezone.utc)
    record = {
        "timestamp_utc": timestamp.isoformat(), "status": "running",
        "endpoint": endpoint, "model": "resnet18_fp16", "version": "1",
        "engine_sha256": engine_hash, "model_config_sha256": sha256(config),
        "dynamic_batching_absent_in_staged_config": not staged_dynamic,
        "triton_model_config_endpoint": config_url,
        "triton_model_config_dynamic_batching": dynamic_settings,
        "image_id": image_id, "image_sha256": validation["image_sha256"],
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "checkpoint_sha256": classifier._provenance["checkpoint_sha256"],
        "preprocessing": classifier._provenance["preprocessing"],
        "parameters": {
            "concurrency_levels": list(CONCURRENCIES), "batch_size": 1,
            "warmup_requests_per_level": WARMUP_REQUESTS,
            "measured_requests_per_level": "max(300, 20 * concurrency)",
            "client": "one requests.Session per worker in a fixed-size thread pool",
            "request_format": "HTTP V2 binary FP32 input; JSON logits output",
            "timing": "perf_counter_ns around each POST through response-body receipt",
            "aggregate_throughput": "successful measured requests / measured phase wall seconds",
            "input_preparation_excluded": True,
            "dynamic_batching_enabled": expect_dynamic,
        },
        "environment": {
            "client_hostname": socket.gethostname(),
            "runai_project": os.environ.get("RUNAI_PROJECT"),
            "runai_job_name": os.environ.get("RUNAI_JOB_NAME"),
            "kubernetes_namespace": Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read_text().strip(),
            "machine": platform.machine(), "python": platform.python_version(),
            "pytorch": torch.__version__, "torchvision": torchvision.__version__,
            "pillow": PIL.__version__, "cuda_from_pytorch": torch.version.cuda,
            "requests": requests.__version__, "numpy": np.__version__,
        },
        "levels": [],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / ("concurrent_http_b1_" + ("dynamic_" if expect_dynamic else "") +
                         timestamp.strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    try:
        for concurrency in CONCURRENCIES:
            row = benchmark_level(concurrency, body, headers, endpoint)
            record["levels"].append(row)
            print(f"concurrency={concurrency}: {row['successful_requests']}/{row['measured_requests']} "
                  f"successful, {row.get('median_latency_ms', float('nan')):.3f} ms median, "
                  f"{row['aggregate_requests_per_second']:.1f} requests/s", flush=True)
        record["status"] = "passed" if all(
            row["warmup_failed"] == row["failed_requests"] == 0 for row in record["levels"]
        ) else "completed_with_errors"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        with path.open("x") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
        print(f"Results: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--expect-dynamic", action="store_true")
    args = parser.parse_args()
    main(args.endpoint, args.expect_dynamic)
