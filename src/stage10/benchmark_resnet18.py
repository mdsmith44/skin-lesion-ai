"""Benchmark existing Stage 10 ResNet18 runtimes on preloaded validation tensors.

Run from the repository root: python -m src.stage10.benchmark_resnet18
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image
import tensorrt as trt
import torch

from src.stage9.resnet18_inference import ResNet18Inference
from src.stage10.build_resnet18_fp16 import FP32_RUN, OUTPUT_DIR, PROFILE, check_engine
from src.stage10.export_resnet18_onnx import sha256


BATCH_SIZES = (1, 8, 32)
WARMUP_ITERATIONS = 100
MEASURED_ITERATIONS = 300
FP16_RUN = OUTPUT_DIR / "fp16_20260928T020556.315794Z"


def validation_tensors(root, classifier):
    # Load only membership and file identifiers/paths; no diagnosis or test image.
    rows = pd.read_csv(root / "data/processed/metadata_splits.csv",
                       usecols=["split", "image_id", "image_path"])
    selected = rows.loc[rows["split"].eq("val")].drop_duplicates("image_id").head(32)
    if len(selected) != 32:
        raise RuntimeError("Fewer than 32 distinct validation images")
    images, identifiers = [], []
    start = perf_counter()
    for row in selected.itertuples(index=False):
        parts = Path(row.image_path).parts
        offset = next(i for i in range(len(parts) - 2)
                      if parts[i:i + 3] == ("data", "raw", "ham10000"))
        path = root.joinpath(*parts[offset:]).resolve(strict=True)
        if path.name != row.image_id + ".jpg":
            raise RuntimeError("Validation image path does not match image ID")
        with Image.open(path) as image:
            images.append(classifier._preprocess(image.convert("RGB")))
        identifiers.append(row.image_id)
    cpu_batch = torch.stack(images).contiguous()
    gpu_batch = cpu_batch.cuda()
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    return {size: gpu_batch[:size].contiguous() for size in BATCH_SIZES}, identifiers, elapsed


def benchmark(call, stream):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    samples = []
    with torch.inference_mode(), torch.cuda.stream(stream):
        for _ in range(WARMUP_ITERATIONS):
            call()
        stream.synchronize()
        for _ in range(MEASURED_ITERATIONS):
            stream.synchronize()
            start_event.record(stream)
            call()
            end_event.record(stream)
            end_event.synchronize()
            samples.append(start_event.elapsed_time(end_event))
    values = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise RuntimeError("Invalid GPU timing sample")
    return values


def main():
    root = Path(__file__).resolve().parents[2]
    output = OUTPUT_DIR / ("benchmark_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "results.json"
    fp32_meta = json.loads((FP32_RUN / "metadata.json").read_text())
    fp16_meta = json.loads((FP16_RUN / "metadata.json").read_text())
    onnx_meta = json.loads((OUTPUT_DIR / "resnet18_full_finetuned.metadata.json").read_text())
    report = {
        "status": "preparing", "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtimes": ["pytorch_fp32", "tensorrt_fp32", "tensorrt_fp16"],
        "batch_sizes": list(BATCH_SIZES),
        "settings": {
            "warmup_iterations_per_runtime_and_batch": WARMUP_ITERATIONS,
            "measured_iterations_per_runtime_and_batch": MEASURED_ITERATIONS,
            "timing": "CUDA events on one dedicated stream; synchronize after warm-up and each measured inference; GPU execution time only",
            "latency_unit": "ms per batch", "throughput": "batch_size * 1000 / mean measured batch latency in ms",
            "inputs": "First 32 distinct validation images; Stage 4 RGB, Resize(224,224), ToTensor, ImageNet Normalize; preloaded to GPU before timing; same tensors for all runtimes",
            "input_dtype": "float32", "input_shape": "[N,3,224,224]",
            "outputs_used_for_classification": False,
            "pytorch_tf32_enabled": False, "cuda_graph": False,
            "engine_builds_performed": False,
        },
        "peak_gpu_memory": {
            "status": "not_comparably_measured",
            "reason": "PyTorch allocator counters omit TensorRT native allocations; device free-memory snapshots do not capture transient peaks consistently across runtimes.",
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "cuda_runtime_from_pytorch": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "tensorrt_version": trt.__version__,
            "fp32_engine_path": fp32_meta["engine_path"],
            "fp32_engine_sha256": fp32_meta["engine_sha256"],
            "fp16_engine_path": fp16_meta["engine_path"],
            "fp16_engine_sha256": fp16_meta["engine_sha256"],
            "onnx_sha256": onnx_meta["onnx_sha256"],
            "optimization_profile": PROFILE,
        },
        "measurements": [],
    }

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    print(f"Results: {report_path}", flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if fp32_meta["onnx_sha256"] != onnx_meta["onnx_sha256"] or fp16_meta["onnx_sha256"] != onnx_meta["onnx_sha256"]:
            raise RuntimeError("Engine ONNX provenance mismatch")
        for metadata in (fp32_meta, fp16_meta):
            if (sha256(metadata["engine_path"]) != metadata["engine_sha256"]
                    or metadata["optimization_profile"] != PROFILE):
                raise RuntimeError("Engine hash or profile mismatch")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        classifier = ResNet18Inference(device="cuda")
        if classifier._provenance["checkpoint_sha256"] != onnx_meta["source_checkpoint_sha256"]:
            raise RuntimeError("PyTorch checkpoint does not match ONNX source")
        report["environment"]["checkpoint_path"] = classifier._provenance["checkpoint_path"]
        report["environment"]["checkpoint_sha256"] = classifier._provenance["checkpoint_sha256"]
        report["environment"]["preprocessing"] = classifier._provenance["preprocessing"]
        tensors, identifiers, preparation_seconds = validation_tensors(root, classifier)
        report["validation_image_ids"] = identifiers
        report["input_preparation_seconds_excluded_from_timing"] = preparation_seconds
        report["status"] = "measuring"
        save()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engines = {
            "tensorrt_fp32": runtime.deserialize_cuda_engine(Path(fp32_meta["engine_path"]).read_bytes()),
            "tensorrt_fp16": runtime.deserialize_cuda_engine(Path(fp16_meta["engine_path"]).read_bytes()),
        }
        contexts = {name: check_engine(engine, PROFILE) for name, engine in engines.items()}
        stream = torch.cuda.Stream()
        for size in BATCH_SIZES:
            inputs = tensors[size]
            for name in report["runtimes"]:
                if name == "pytorch_fp32":
                    call = lambda x=inputs: classifier._model(x)
                else:
                    context = contexts[name]
                    if not context.set_input_shape("normalized_rgb", tuple(inputs.shape)):
                        raise RuntimeError(f"TensorRT input shape rejected: {name}, batch {size}")
                    result = torch.empty((size, 7), device="cuda", dtype=torch.float32)
                    if (not context.set_tensor_address("normalized_rgb", inputs.data_ptr())
                            or not context.set_tensor_address("logits", result.data_ptr())):
                        raise RuntimeError(f"TensorRT I/O binding failed: {name}")

                    def call(ctx=context, cuda_stream=stream):
                        if not ctx.execute_async_v3(cuda_stream.cuda_stream):
                            raise RuntimeError("TensorRT execution failed")

                values = benchmark(call, stream)
                row = {
                    "runtime": name, "batch_size": size,
                    "warmup_iterations": WARMUP_ITERATIONS,
                    "measured_iterations": MEASURED_ITERATIONS,
                    "median_latency_ms_per_batch": float(np.median(values)),
                    "p95_latency_ms_per_batch": float(np.percentile(values, 95)),
                    "mean_latency_ms_per_batch": float(values.mean()),
                    "throughput_images_per_second": float(size * 1000 / values.mean()),
                    "latency_samples_ms": values.tolist(),
                }
                report["measurements"].append(row)
                save()
                print(f"{name} batch={size}: median={row['median_latency_ms_per_batch']:.3f} ms, "
                      f"p95={row['p95_latency_ms_per_batch']:.3f} ms, "
                      f"throughput={row['throughput_images_per_second']:.1f} images/s", flush=True)
        report["status"] = "passed"
        save()
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
