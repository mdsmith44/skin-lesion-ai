"""Build an FP16 TensorRT engine and compare three runtimes on one val image.

Run from the repository root: python -m src.stage10.build_resnet18_fp16
"""
import json
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image
import tensorrt as trt
import torch

from src.stage10.export_resnet18_onnx import ONNX_PATH, OUTPUT_DIR, sha256
from src.stage9.resnet18_inference import CLASS_NAMES, ResNet18Inference


FP32_RUN = OUTPUT_DIR / "fp32_20260928T015516.161897Z"
FP32_ENGINE = FP32_RUN / "resnet18_full_finetuned.fp32.engine"
PROFILE = {
    "min": [1, 3, 224, 224],
    "opt": [8, 3, 224, 224],
    "max": [32, 3, 224, 224],
}


def check_engine(engine, expected_profile):
    if engine is None or engine.num_io_tensors != 2 or engine.num_optimization_profiles != 1:
        raise RuntimeError("Engine missing or has unexpected tensors/profiles")
    for name, shape, mode in (
        ("normalized_rgb", (-1, 3, 224, 224), trt.TensorIOMode.INPUT),
        ("logits", (-1, 7), trt.TensorIOMode.OUTPUT),
    ):
        if (tuple(engine.get_tensor_shape(name)) != shape
                or engine.get_tensor_dtype(name) != trt.float32
                or engine.get_tensor_mode(name) != mode):
            raise RuntimeError(f"Unexpected TensorRT I/O contract: {name}")
    profile = [list(s) for s in engine.get_tensor_profile_shape("normalized_rgb", 0)]
    if profile != list(expected_profile.values()):
        raise RuntimeError(f"Unexpected optimization profile: {profile}")
    context = engine.create_execution_context()
    for batch in (1, 8, 32):
        if (not context.set_input_shape("normalized_rgb", (batch, 3, 224, 224))
                or tuple(context.get_tensor_shape("logits")) != (batch, 7)):
            raise RuntimeError(f"Unexpected output shape at batch {batch}")
    return context


def infer(context, inputs):
    output = torch.empty((1, 7), device=inputs.device, dtype=torch.float32)
    if (not context.set_input_shape("normalized_rgb", tuple(inputs.shape))
            or not context.set_tensor_address("normalized_rgb", inputs.data_ptr())
            or not context.set_tensor_address("logits", output.data_ptr())
            or not context.execute_async_v3(torch.cuda.current_stream().cuda_stream)):
        raise RuntimeError("TensorRT execution failed")
    torch.cuda.synchronize()
    return output.cpu()


def main():
    root = Path(__file__).resolve().parents[2]
    output = OUTPUT_DIR / ("fp16_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    engine_path = output / "resnet18_full_finetuned.fp16.engine"
    report_path = output / "metadata.json"
    command = [
        shutil.which("trtexec") or "trtexec", f"--onnx={ONNX_PATH}",
        f"--saveEngine={engine_path}",
        "--minShapes=normalized_rgb:1x3x224x224",
        "--optShapes=normalized_rgb:8x3x224x224",
        "--maxShapes=normalized_rgb:32x3x224x224",
        "--fp16", "--noTF32", "--inputIOFormats=fp32:chw",
        "--outputIOFormats=fp32:chw", "--memPoolSize=workspace:1024",
        "--profilingVerbosity=detailed", "--skipInference",
    ]
    report = {
        "status": "building", "tensorrt_version": trt.__version__,
        "onnx_path": str(ONNX_PATH), "onnx_sha256": sha256(ONNX_PATH),
        "engine_path": str(engine_path),
        "precision": "FP16 tactics enabled; FP32 input and output; TF32 disabled",
        "optimization_profile": PROFILE,
        "build_command": shlex.join(command), "build_argv": command,
        "benchmark_performed": False,
        "fp32_engine_path": str(FP32_ENGINE),
        "fp32_engine_sha256": sha256(FP32_ENGINE),
    }

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    print(f"Artifacts: {output}", flush=True)
    try:
        onnx_meta = json.loads((OUTPUT_DIR / "resnet18_full_finetuned.metadata.json").read_text())
        fp32_meta = json.loads((FP32_RUN / "metadata.json").read_text())
        if (report["onnx_sha256"] != onnx_meta["onnx_sha256"]
                or report["onnx_sha256"] != fp32_meta["onnx_sha256"]
                or report["fp32_engine_sha256"] != fp32_meta["engine_sha256"]):
            raise RuntimeError("Source ONNX or FP32 engine hash does not match prior metadata")
        with (output / "build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        report.update(engine_sha256=sha256(engine_path),
                      engine_size_bytes=engine_path.stat().st_size, status="built")
        save()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        fp32_engine = runtime.deserialize_cuda_engine(FP32_ENGINE.read_bytes())
        fp16_engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        fp32_context = check_engine(fp32_engine, PROFILE)
        fp16_context = check_engine(fp16_engine, PROFILE)
        layer_info = fp16_engine.create_engine_inspector().get_engine_information(
            trt.LayerInformationFormat.JSON)
        (output / "fp16_layers.json").write_text(layer_info, encoding="utf-8")
        layers = json.loads(layer_info).get("Layers", [])
        fp16_layers = [layer.get("Name", "") for layer in layers
                       if "Half" in json.dumps(layer) or "FP16" in json.dumps(layer)]
        report["fp16_layer_count"] = len(fp16_layers)
        report["layer_inspection_path"] = str(output / "fp16_layers.json")
        if not fp16_layers:
            raise RuntimeError("No FP16 execution layers found in engine inspector")
        report["io_validation"] = {
            "input": [-1, 3, 224, 224], "output": [-1, 7],
            "dtype": "float32", "shape_checks_batches": [1, 8, 32],
        }
        save()

        # Read membership, ID and path only. No diagnosis column or test image is used.
        rows = pd.read_csv(root / "data/processed/metadata_splits.csv",
                           usecols=["split", "image_id", "image_path"])
        row = rows.loc[rows["split"].eq("val")].iloc[0]
        parts = Path(row["image_path"]).parts
        start = next(i for i in range(len(parts) - 2)
                     if parts[i:i + 3] == ("data", "raw", "ham10000"))
        image_path = root.joinpath(*parts[start:]).resolve(strict=True)
        if image_path.name != row["image_id"] + ".jpg":
            raise RuntimeError("Validation image path does not match image ID")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        classifier = ResNet18Inference(device="cuda")
        if classifier._provenance["checkpoint_sha256"] != onnx_meta["source_checkpoint_sha256"]:
            raise RuntimeError("PyTorch checkpoint does not match ONNX source")
        with Image.open(image_path) as image:
            inputs = classifier._preprocess(image.convert("RGB")).unsqueeze(0).contiguous().cuda()
        with torch.inference_mode():
            torch_logits = classifier._model(inputs).cpu()
        fp32_logits = infer(fp32_context, inputs)
        fp16_logits = infer(fp16_context, inputs)
        if not all(torch.isfinite(x).all() for x in (torch_logits, fp32_logits, fp16_logits)):
            raise RuntimeError("Non-finite logits")

        argmax = {
            "pytorch_fp32": int(torch_logits.argmax(1).item()),
            "tensorrt_fp32": int(fp32_logits.argmax(1).item()),
            "tensorrt_fp16": int(fp16_logits.argmax(1).item()),
        }
        report["comparison"] = {
            "validation_image_id": row["image_id"], "image_path": str(image_path),
            "image_sha256": sha256(image_path), "reference_diagnosis_used": False,
            "preprocessing_applications": 1,
            "preprocessing": classifier._provenance["preprocessing"],
            "source_checkpoint_sha256": classifier._provenance["checkpoint_sha256"],
            "gpu": torch.cuda.get_device_name(), "pytorch_tf32_enabled": False,
            "input_shape": list(inputs.shape), "class_names_in_logit_order": list(CLASS_NAMES),
            "logits": {
                "pytorch_fp32": torch_logits[0].tolist(),
                "tensorrt_fp32": fp32_logits[0].tolist(),
                "tensorrt_fp16": fp16_logits[0].tolist(),
            },
            "max_absolute_logit_difference": {
                "pytorch_vs_tensorrt_fp32": float((torch_logits - fp32_logits).abs().max()),
                "pytorch_vs_tensorrt_fp16": float((torch_logits - fp16_logits).abs().max()),
                "tensorrt_fp32_vs_fp16": float((fp32_logits - fp16_logits).abs().max()),
            },
            "argmax_index": argmax,
            "argmax_class": {name: CLASS_NAMES[index] for name, index in argmax.items()},
            "all_argmax_agree": len(set(argmax.values())) == 1,
        }
        report["status"] = "passed"
        save()
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
