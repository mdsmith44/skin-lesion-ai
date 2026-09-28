"""Reusable, offline SmolVLM2 visual-description inference for one image.

Use ``SmolVLMDescriber().describe(image_path_or_pil_image)``. The module CLI
writes one smoke result under persistent Stage 10 storage.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.stage10.description_contract import (
    CONTRACT_VERSION, GENERATION, PROMPT, validate_description,
)


MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
MODEL_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
SMOKE_ROOT = Path("/workspace/storage/skin-lesion-ai/outputs/stage10/smolvlm/describe_smoke")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _image_only_processor_guard():
    # This installed Transformers version requires num2words at processor
    # construction although it uses that package only for video frame counts.
    # This callable makes video formatting fail; it is never used for images.
    from transformers.models.smolvlm import processing_smolvlm

    if processing_smolvlm.num2words is None:
        def reject_video_frame_count(*_args, **_kwargs):
            raise RuntimeError("Video inputs are unsupported by this image-only component")

        processing_smolvlm.num2words = reject_video_frame_count


class SmolVLMDescriber:
    """Load the pinned base model once and describe individual RGB images."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the pinned SmolVLM2 model")
        snapshot = Path(snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        )).resolve(strict=True)
        if snapshot.name != MODEL_REVISION:
            raise RuntimeError("Cached model revision does not match the pinned revision")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        _image_only_processor_guard()
        self.processor = AutoProcessor.from_pretrained(
            snapshot, local_files_only=True, do_image_splitting=False,
            size={"longest_edge": 512},
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        self.snapshot = snapshot

    def describe(self, image_input: str | Path | Image.Image) -> dict:
        """Return raw output and a limited-contract validation decision.

        Only the RGB image and frozen prompt are sent to the model. Acceptance
        means the lexical checks passed, not that the description is verified.
        """
        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
            source = {"kind": "pil_image", "size": list(image.size)}
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input).expanduser().resolve(strict=True)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            source = {"kind": "file", "path": str(path),
                      "image_sha256": _sha256_file(path), "size": list(image.size)}
        else:
            raise TypeError("image_input must be a file path or PIL image")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.model.device)
        prefix_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            token_ids = self.model.generate(**inputs, **GENERATION)[0, prefix_length:]
        raw = self.processor.decode(token_ids, skip_special_tokens=True)
        decision = validate_description(raw)
        return {
            "model_identity": MODEL_ID, "model_revision": MODEL_REVISION,
            "lora_active": False, "raw_generated_text": raw,
            **decision,
            "prompt_contract_version": CONTRACT_VERSION,
            "prompt": PROMPT,
            "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            "generation_settings": dict(GENERATION),
            "generated_token_ids": token_ids.tolist(),
            "input": source,
        }


def main():
    parser = argparse.ArgumentParser(description="Run one SmolVLM2 /describe smoke inference")
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    describer = SmolVLMDescriber()
    result = describer.describe(args.image)
    result["environment"] = {
        "gpu": torch.cuda.get_device_name(), "cuda_from_pytorch": torch.version.cuda,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "pillow": PIL.__version__, "device": "cuda", "model_dtype": "bfloat16",
        "attention_implementation": "sdpa", "cache_snapshot": str(describer.snapshot),
        "network_mode": "local_files_only",
        "processor_settings": {"do_image_splitting": False, "longest_edge": 512},
        "image_only_processor_guard": "The installed num2words package is absent; video frame formatting raises explicitly. No video input is used.",
    }
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    output = SMOKE_ROOT / ("smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    print(json.dumps({"validation_status": result["validation_status"],
                      "rejection_reasons": result["rejection_reasons"],
                      "raw_generated_text": result["raw_generated_text"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
