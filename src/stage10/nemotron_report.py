"""Stage 10 TensorRT FP16 ResNet evidence with the unchanged Stage 9 reporter.

The reporter receives only Stage 9's ``prompt_evidence`` allowlist. Use
``Stage10NemotronReport().report(image, split='val')`` for one image.
"""
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import tensorrt as trt
import torch
from torchvision import transforms

from src.stage9.reporting import (
    LABELS, PROMPT_VERSION, UNKNOWNS, build_evidence, prompt_evidence,
)
from src.stage10.build_resnet18_fp16 import PROFILE, check_engine


ENGINE_METADATA = Path(
    "/workspace/storage/skin-lesion-ai/outputs/stage10/resnet18/"
    "fp16_20260928T020556.315794Z/metadata.json"
)
SMOKE_ROOT = Path("/workspace/storage/skin-lesion-ai/outputs/stage10/nemotron_report/smoke")


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class TensorRTFP16ResNet18:
    """Existing FP16 engine with the exact Stage 4 RGB preprocessing."""

    def __init__(self, metadata_path=ENGINE_METADATA):
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        engine_path = Path(metadata["engine_path"]).resolve(strict=True)
        if (metadata["status"] != "passed"
                or metadata["optimization_profile"] != PROFILE
                or file_sha256(engine_path) != metadata["engine_sha256"]):
            raise RuntimeError("FP16 TensorRT engine metadata/hash/profile mismatch")
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        self.context = check_engine(self.engine, PROFILE)
        resize = transforms.Resize((224, 224))
        self.preprocess = transforms.Compose([
            resize, transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.provenance = {
            "runtime": "TensorRT", "precision": metadata["precision"],
            "tensorrt_version": trt.__version__, "gpu": torch.cuda.get_device_name(),
            "engine_path": str(engine_path), "engine_sha256": metadata["engine_sha256"],
            "onnx_sha256": metadata["onnx_sha256"],
            "source_checkpoint_sha256": metadata["comparison"]["source_checkpoint_sha256"],
            "optimization_profile": PROFILE,
            "input": {"name": "normalized_rgb", "shape": [1, 3, 224, 224], "dtype": "float32"},
            "output": {"name": "logits", "shape": [1, 7], "dtype": "float32"},
            "class_names_in_logit_order": list(LABELS),
            "preprocessing": {
                "reference": "notebooks/04_model_evaluation.ipynb",
                "steps": ["PIL RGB conversion", "Resize", "ToTensor", "Normalize"],
                "resize_hw": [224, 224], "resize_interpolation": resize.interpolation.value,
                "resize_antialias": resize.antialias,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "augmentation": False,
            },
            "score_interpretation": "Uncalibrated softmax values; not clinical probabilities.",
        }

    def predict(self, image_input):
        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
            input_info = {"kind": "pil_image", "size": list(image.size),
                          "rgb_pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input).expanduser().resolve(strict=True)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            input_info = {"kind": "file", "image_path": str(path),
                          "image_sha256": file_sha256(path)}
        else:
            raise TypeError("image_input must be a file path or PIL image")
        inputs = self.preprocess(image).unsqueeze(0).contiguous().cuda()
        output = torch.empty((1, 7), device="cuda", dtype=torch.float32)
        if (not self.context.set_input_shape("normalized_rgb", tuple(inputs.shape))
                or not self.context.set_tensor_address("normalized_rgb", inputs.data_ptr())
                or not self.context.set_tensor_address("logits", output.data_ptr())
                or not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)):
            raise RuntimeError("TensorRT FP16 ResNet18 inference failed")
        torch.cuda.synchronize()
        if not torch.isfinite(output).all():
            raise RuntimeError("TensorRT FP16 ResNet18 produced non-finite logits")
        scores = torch.softmax(output[0], dim=0).cpu().tolist()
        label = LABELS[int(output.argmax(1).item())]
        return {
            "predicted_class": label,
            "softmax_scores": dict(zip(LABELS, scores)),
            "score_interpretation": "Uncalibrated model scores; not clinical probabilities.",
            "input": input_info,
            "model_provenance": copy.deepcopy(self.provenance),
        }


def stage9_evidence(prediction, *, image_id, lesion_id, split, input_info=None,
                    model_provenance=None):
    """Use Stage 9's builder, including its null-prediction convention."""
    if prediction is not None:
        evidence = build_evidence(prediction, image_id=image_id,
                                  lesion_id=lesion_id, split=split)
    else:
        if split not in {"train", "val"}:
            raise ValueError("Stage 9 development accepts only train or val.")
        evidence = {
            "schema_version": "1.0",
            "observed_metadata": {"image_id": image_id, "lesion_id": lesion_id, "split": split},
            "classifier": {"predicted_class": None, "softmax_scores": None,
                           "top_softmax_score": None, "score_type": "unknown"},
            "unknowns": list(UNKNOWNS),
            "input_provenance": input_info,
            "model_provenance": model_provenance,
        }
    # Stage 9 validates the only object that will reach Nemotron.
    prompt_evidence(evidence)
    return evidence


class Stage10NemotronReport:
    """Coordinate existing FP16 classifier and unchanged Stage 9 Nemotron."""

    def __init__(self, classifier=None, reporter=None):
        self.classifier = classifier if classifier is not None else TensorRTFP16ResNet18()
        if reporter is None:
            from src.stage9.nemotron_reporting import NemotronReporter
            reporter = NemotronReporter()
        self.reporter = reporter

    def report(self, image_input, *, split, image_id=None, lesion_id=None):
        if split not in {"train", "val"}:
            raise ValueError("Stage 9 development accepts only train or val.")
        if image_id is None:
            if not isinstance(image_input, (str, Path)):
                raise ValueError("image_id is required for a PIL image")
            image_id = Path(image_input).stem
        prediction = self.classifier.predict(image_input)
        evidence = stage9_evidence(
            prediction, image_id=image_id, lesion_id=lesion_id, split=split,
            input_info=prediction["input"] if prediction else None,
            model_provenance=prediction["model_provenance"] if prediction else
            getattr(self.classifier, "provenance", None),
        )
        generation = self.reporter.report(evidence)
        status = generation["status"]
        if status not in {"accepted", "rejected"}:
            raise RuntimeError("Unexpected Stage 9 reporter status")
        return {
            "classifier_prediction": prediction,
            "classifier_evidence": evidence,
            "nemotron_raw_generation": generation["raw_response"],
            "validated_report": generation["report"],
            "validation_status": status,
            "rejection_reason": generation.get("validation_error"),
            "classifier_runtime_provenance": getattr(self.classifier, "provenance", None),
            "nemotron_model_provenance": generation["reporter_provenance"],
            "report_contract_version": PROMPT_VERSION,
            "stage9_reporter_result": generation,
        }


def main():
    parser = argparse.ArgumentParser(description="One Stage 10 grounded-report smoke test")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    args = parser.parse_args()
    component = Stage10NemotronReport()
    result = component.report(args.image, split=args.split)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    output = SMOKE_ROOT / ("smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)
    print(json.dumps({"status": result["validation_status"],
                      "rejection_reason": result["rejection_reason"],
                      "report": result["validated_report"]}), flush=True)


if __name__ == "__main__":
    main()
