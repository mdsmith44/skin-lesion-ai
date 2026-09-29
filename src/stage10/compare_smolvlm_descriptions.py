"""Compare cached base SmolVLM and the Stage 8 LoRA on seven val images.

Run from the repository root with the existing persistent HF cache:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m src.stage10.compare_smolvlm_descriptions
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image
import torch
import transformers
import nemo_automodel
from huggingface_hub import snapshot_download
from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

from src.stage8.inference import MODEL_ID, load_adapted_model, load_processor
from src.stage10.export_resnet18_onnx import sha256
from src.stage10.description_contract import (
    CLASS_PATTERN, DIAGNOSTIC_PATTERN, GENERATION, PROMPT,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path("/workspace/storage/skin-lesion-ai/outputs/stage10/smolvlm")
ADAPTER = Path("/workspace/storage/skin-lesion-ai/outputs/stage8/lora-b64-tempered/LOWEST_VAL")
CLASSES = frozenset(("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"))


def selected_images():
    """Use labels for sampling only; return image IDs and resolved paths."""
    path = ROOT / "data/processed/multimodal/val.csv"
    first = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "val":
                raise RuntimeError("Unexpected split in validation CSV")
            label = row["class_abbreviation"]
            if label in CLASSES and label not in first:
                first[label] = (row["image_id"], row["image_path"])
            if len(first) == len(CLASSES):
                break
    if set(first) != CLASSES:
        raise RuntimeError("Validation split does not include all seven sampling classes")
    selected = []
    for image_id, old_path in first.values():
        parts = Path(old_path).parts
        offsets = [i for i in range(len(parts) - 2)
                   if parts[i:i + 3] == ("data", "raw", "ham10000")]
        if len(offsets) != 1:
            raise RuntimeError(f"Unrecognized image path for {image_id}")
        suffix = parts[offsets[0]:]
        if ".." in suffix or suffix[-1] != image_id + ".jpg":
            raise RuntimeError(f"Invalid image path for {image_id}")
        selected.append((image_id, ROOT.joinpath(*suffix).resolve(strict=True)))
    return selected


def generate(model, processor, prepared):
    outputs = []
    for item in prepared:
        inputs = item["inputs"].to(model.device)
        prefix_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            tokens = model.generate(**inputs, **GENERATION)[0, prefix_length:]
        # Keep decoder output exactly, including whitespace; no strip or rewriting.
        raw = processor.decode(tokens, skip_special_tokens=True)
        classes = sorted(set(match.group(0).lower() for match in CLASS_PATTERN.finditer(raw)))
        diagnostic = sorted(set(match.group(0).lower()
                                for match in DIAGNOSTIC_PATTERN.finditer(raw)))
        outputs.append({
            "image_id": item["image_id"], "raw_description": raw,
            "generated_token_ids": tokens.tolist(),
            "mechanical_flags": {
                "contains_class_abbreviation": bool(classes), "matched_class_abbreviations": classes,
                "contains_diagnostic_term": bool(diagnostic), "matched_diagnostic_terms": diagnostic,
            },
        })
        print(f"Generated {len(outputs)}/7 for {type(model).__name__}", flush=True)
    return outputs


def main():
    output = OUTPUT_ROOT / ("comparison_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "results.json"
    snapshot = Path(snapshot_download(MODEL_ID, local_files_only=True)).resolve(strict=True)
    revision = snapshot.name
    adapter_path = ADAPTER.resolve(strict=True)
    adapter_weights = adapter_path / "model/adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(adapter_weights)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    report = {
        "status": "preparing", "created_utc": datetime.now(timezone.utc).isoformat(),
        "sampling": "First validation CSV row for each of seven reference classes, in first-appearance order; class fields used only for selection and omitted from model inputs/results",
        "validation_csv": str(ROOT / "data/processed/multimodal/val.csv"),
        "prompt": PROMPT, "generation_settings": GENERATION,
        "input_construction": "One RGB image plus fixed user prompt via cached Stage 8 SmolVLM processor and chat template; no target or other model evidence",
        "models": {
            "base": {"identity": MODEL_ID, "revision": revision, "lora_active": False},
            "stage8_lora": {"identity": MODEL_ID, "revision": revision, "lora_active": True,
                            "adapter_path": str(adapter_path),
                            "adapter_weights_sha256": sha256(adapter_weights)},
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(), "cuda_from_pytorch": torch.version.cuda,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "nemo_automodel": getattr(nemo_automodel, "__version__", "unknown"),
            "pillow": PIL.__version__, "model_dtype": "bfloat16 base",
            "attention_implementation": "sdpa", "device": "cuda",
            "hf_cache_snapshot": str(snapshot), "network_mode": "local_files_only",
            "seed": 0,
        },
        "mechanical_flagging": {
            "class_abbreviation_pattern": CLASS_PATTERN.pattern,
            "diagnostic_term_pattern": DIAGNOSTIC_PATTERN.pattern,
            "clinical_quality_score": None,
        },
        "images": [], "outputs": {},
    }

    def save():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    save()
    print(f"Results: {report_path}", flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        processor = load_processor()
        prepared = []
        for image_id, path in selected_images():
            with Image.open(path) as source:
                image = source.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
            prepared.append({"image_id": image_id, "inputs": inputs})
            report["images"].append({"image_id": image_id, "image_path": str(path),
                                     "image_sha256": sha256(path)})
        if len(prepared) != 7 or len({item["image_id"] for item in prepared}) != 7:
            raise RuntimeError("Expected exactly seven distinct validation images")
        save()

        base = NeMoAutoModelForImageTextToText.from_pretrained(
            MODEL_ID, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", use_liger_kernel=False, force_hf=True,
        ).to("cuda").eval()
        if any("lora_A" in name or "lora_B" in name for name, _ in base.named_parameters()):
            raise RuntimeError("Unexpected LoRA parameters in base model")
        report["status"] = "generating_base"
        save()
        report["outputs"]["base"] = generate(base, processor, prepared)
        save()
        del base
        torch.cuda.empty_cache()

        adapted = load_adapted_model(adapter_path)
        lora_parameters = [name for name, _ in adapted.named_parameters()
                           if "lora_A" in name or "lora_B" in name]
        if not lora_parameters:
            raise RuntimeError("Stage 8 LoRA parameters not active")
        report["models"]["stage8_lora"]["lora_parameter_tensor_count"] = len(lora_parameters)
        report["status"] = "generating_lora"
        save()
        report["outputs"]["stage8_lora"] = generate(adapted, processor, prepared)
        report["status"] = "passed"
        save()
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
