"""Add cached SmolVLM2-500M outputs beside the frozen seven-image comparison.

Run offline from the repository root:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m src.stage10.extend_smolvlm2_500m_descriptions
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.stage10.compare_smolvlm_descriptions import (
    CLASS_PATTERN, DIAGNOSTIC_PATTERN, GENERATION, OUTPUT_ROOT, PROMPT,
)
from src.stage10.export_resnet18_onnx import sha256


MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
COMPARISON = OUTPUT_ROOT / "comparison_20260928T023015.847968Z"
ORIGINAL = COMPARISON / "results.json"
RESULT = COMPARISON / "smolvlm2_500m_results.json"


def main():
    if RESULT.exists():
        prior = json.loads(RESULT.read_text(encoding="utf-8"))
        if prior.get("status") != "failed" or prior.get("outputs"):
            raise FileExistsError(f"Refusing to overwrite prior 500M outputs: {RESULT}")
    original_sha = sha256(ORIGINAL)
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    if (original["status"] != "passed" or original["prompt"] != PROMPT
            or original["generation_settings"] != GENERATION
            or len(original["images"]) != 7
            or set(original["outputs"]) != {"base", "stage8_lora"}
            or any(len(records) != 7 for records in original["outputs"].values())):
        raise RuntimeError("Frozen original comparison is incomplete or has changed")
    ids = [record["image_id"] for record in original["images"]]
    if len(set(ids)) != 7:
        raise RuntimeError("Original comparison has duplicate image IDs")
    for name in ("base", "stage8_lora"):
        if [record["image_id"] for record in original["outputs"][name]] != ids:
            raise RuntimeError("Original model outputs have mismatched image IDs")

    snapshot = Path(snapshot_download(MODEL_ID, local_files_only=True)).resolve(strict=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    report = {
        "status": "preparing", "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_comparison_path": str(ORIGINAL),
        "original_comparison_sha256": original_sha,
        "model": {"identity": MODEL_ID, "revision": snapshot.name,
                  "lora_active": False, "weights_sha256": sha256(snapshot / "model.safetensors")},
        "prompt": original["prompt"], "generation_settings": dict(original["generation_settings"]),
        "image_ids": ids,
        "image_sha256": {record["image_id"]: record["image_sha256"]
                         for record in original["images"]},
        "environment": {
            "gpu": torch.cuda.get_device_name(), "cuda_from_pytorch": torch.version.cuda,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "pillow": PIL.__version__, "device": "cuda", "model_dtype": "bfloat16",
            "attention_implementation": "sdpa", "cache_snapshot": str(snapshot),
            "network_mode": "local_files_only", "seed": 0,
            "processor_settings": {"do_image_splitting": False, "longest_edge": 512},
            "image_only_processor_guard": "Installed num2words is absent; a local guard raises if video timestamp formatting is reached. No video inputs are used.",
            "input_construction": "RGB image and frozen prompt only, through model's cached processor and chat template",
        },
        "mechanical_flagging": dict(original["mechanical_flagging"]),
        "outputs": [],
    }

    def save():
        RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    save()
    print(f"Results: {RESULT}", flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        # The installed SmolVLM processor requires num2words at construction,
        # although it only calls it when formatting video frame counts. Make
        # that video-only path fail explicitly while retaining its image path.
        from transformers.models.smolvlm import processing_smolvlm

        if processing_smolvlm.num2words is None:
            def reject_video_frame_count(*_args, **_kwargs):
                raise RuntimeError("Video formatting is outside this still-image experiment")

            processing_smolvlm.num2words = reject_video_frame_count
        processor = AutoProcessor.from_pretrained(
            snapshot, local_files_only=True, do_image_splitting=False,
            size={"longest_edge": 512},
        )
        model = AutoModelForImageTextToText.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        report["status"] = "generating"
        save()
        for record in original["images"]:
            image_id = record["image_id"]
            path = Path(record["image_path"]).resolve(strict=True)
            if path.name != image_id + ".jpg" or sha256(path) != record["image_sha256"]:
                raise RuntimeError(f"Image identity/hash mismatch: {image_id}")
            with Image.open(path) as source:
                image = source.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(model.device)
            prefix_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                tokens = model.generate(**inputs, **GENERATION)[0, prefix_length:]
            raw = processor.decode(tokens, skip_special_tokens=True)
            classes = sorted({match.group(0).lower()
                              for match in CLASS_PATTERN.finditer(raw)})
            diagnostic = sorted({match.group(0).lower()
                                 for match in DIAGNOSTIC_PATTERN.finditer(raw)})
            report["outputs"].append({
                "image_id": image_id, "raw_description": raw,
                "generated_token_ids": tokens.tolist(),
                "mechanical_flags": {
                    "contains_class_abbreviation": bool(classes),
                    "matched_class_abbreviations": classes,
                    "contains_diagnostic_term": bool(diagnostic),
                    "matched_diagnostic_terms": diagnostic,
                },
            })
            save()
            print(f"Generated {len(report['outputs'])}/7: {image_id}", flush=True)
        if sha256(ORIGINAL) != original_sha:
            raise RuntimeError("Original comparison changed during the extension")
        report["aggregate_mechanical_checks"] = {
            "outputs_with_class_abbreviation": sum(
                item["mechanical_flags"]["contains_class_abbreviation"]
                for item in report["outputs"]),
            "outputs_with_diagnostic_term": sum(
                item["mechanical_flags"]["contains_diagnostic_term"]
                for item in report["outputs"]),
        }
        report["status"] = "passed"
        save()
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
