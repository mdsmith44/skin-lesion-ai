"""Compare direct TensorRT and Triton logits for the recorded val image."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import requests
import tensorrt as trt
import torch

from src.stage9.resnet18_inference import CLASS_NAMES, ResNet18Inference
from src.stage10.build_resnet18_fp16 import PROFILE, check_engine, infer
from src.stage10.export_resnet18_onnx import sha256


METADATA = Path(
    "/workspace/storage/skin-lesion-ai/outputs/stage10/resnet18/"
    "fp16_20260928T020556.315794Z/metadata.json"
)
OUTPUT_DIR = Path("/workspace/storage/skin-lesion-ai/outputs/stage10/triton")
ENDPOINT = (
    "http://stage10-resnet18-triton-v2.runai-nccl.svc.cluster.local"
    "/v2/models/resnet18_fp16/infer"
)
ENGINE_SHA256 = "7370bba2e48f106824f7f9a4d4ab6d065a5cf2b512311de6a47c95a5ac844cb3"


def main(endpoint=ENDPOINT):
    metadata = json.loads(METADATA.read_text())
    previous = metadata["comparison"]
    image_id = previous["validation_image_id"]
    if image_id != "ISIC_0027419" or previous["reference_diagnosis_used"]:
        raise ValueError("Unexpected validation image provenance")
    image_path = Path(previous["image_path"]).resolve(strict=True)
    if image_path.name != image_id + ".jpg" or sha256(image_path) != previous["image_sha256"]:
        raise ValueError("Validation image does not match the prior check")
    engine_path = Path(metadata["engine_path"])
    engine_hash = sha256(engine_path)
    if engine_hash != ENGINE_SHA256 or engine_hash != metadata["engine_sha256"]:
        raise ValueError("FP16 engine hash mismatch")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    classifier = ResNet18Inference(device="cuda")
    if classifier._provenance["checkpoint_sha256"] != previous["source_checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    with Image.open(image_path) as image:
        inputs = classifier._preprocess(image.convert("RGB")).unsqueeze(0).contiguous().cuda()
    if inputs.shape != (1, 3, 224, 224) or inputs.dtype != torch.float32:
        raise ValueError("Unexpected preprocessed tensor")

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = check_engine(engine, PROFILE)
    direct_logits = infer(context, inputs).numpy().reshape(1, 7)

    payload = inputs.detach().cpu().numpy().astype("<f4", copy=False).tobytes(order="C")
    header = json.dumps({
        "inputs": [{
            "name": "normalized_rgb", "shape": [1, 3, 224, 224],
            "datatype": "FP32", "parameters": {"binary_data_size": len(payload)},
        }],
        "outputs": [{"name": "logits", "parameters": {"binary_data": False}}],
    }, separators=(",", ":")).encode("utf-8")
    response = requests.post(
        endpoint, data=header + payload,
        headers={"Content-Type": "application/octet-stream",
                 "Inference-Header-Content-Length": str(len(header))},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("model_name") != "resnet18_fp16" or result.get("model_version") != "1":
        raise ValueError("Unexpected Triton model identity/version")
    outputs = result.get("outputs", [])
    if (len(outputs) != 1 or outputs[0].get("name") != "logits"
            or outputs[0].get("datatype") != "FP32"
            or outputs[0].get("shape") != [1, 7]):
        raise ValueError("Unexpected Triton output contract")
    triton_logits = np.asarray(outputs[0].get("data"), dtype=np.float32).reshape(1, 7)
    if not np.isfinite(direct_logits).all() or not np.isfinite(triton_logits).all():
        raise ValueError("Non-finite logits")

    direct_index = int(direct_logits.argmax(axis=1)[0])
    triton_index = int(triton_logits.argmax(axis=1)[0])
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_id": image_id,
        "endpoint": endpoint,
        "model": result["model_name"],
        "version": result["model_version"],
        "engine_sha256": engine_hash,
        "direct_tensorrt_fp16_logits": direct_logits[0].tolist(),
        "triton_logits": triton_logits[0].tolist(),
        "max_absolute_logit_difference": float(np.max(np.abs(direct_logits - triton_logits))),
        "direct_argmax_index": direct_index,
        "direct_argmax_class": CLASS_NAMES[direct_index],
        "triton_argmax_index": triton_index,
        "triton_argmax_class": CLASS_NAMES[triton_index],
        "argmax_agreement": direct_index == triton_index,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = OUTPUT_DIR / f"{image_id}_direct_vs_triton_fp16_{timestamp}.json"
    with path.open("x") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"result_path": str(path), **record}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=ENDPOINT)
    main(parser.parse_args().endpoint)
