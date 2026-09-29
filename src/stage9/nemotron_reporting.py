"""Pinned, local-cache-only Nemotron text reporting with fail-closed validation."""
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .reporting import (
    PROMPT_VERSION, SYSTEM_PROMPT, canonical_json, content_hash,
    prompt_evidence, validate_report,
)

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
REVISION = "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
DEFAULT_CACHE = "/workspace/storage/skin-lesion-ai/huggingface/hub"


class NemotronReporter:
    def __init__(self, cache_dir=DEFAULT_CACHE):
        start = time.perf_counter()
        options = dict(revision=REVISION, trust_remote_code=False,
                       local_files_only=True, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, **options,
        ).eval()
        self.model.requires_grad_(False)
        torch.cuda.synchronize()
        self.provenance = {
            "model_id": MODEL_ID, "revision": REVISION,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "model_class": type(self.model).__name__,
            "torch": str(torch.__version__), "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16", "trust_remote_code": False,
            "local_files_only": True, "load_seconds": time.perf_counter() - start,
            "prompt_version": PROMPT_VERSION,
            "system_prompt_sha256": content_hash(SYSTEM_PROMPT),
            "generation": {"do_sample": False, "num_beams": 1,
                           "max_new_tokens": 256, "enable_thinking": False},
        }

    def report(self, evidence):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": canonical_json(prompt_evidence(evidence))},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        ).to("cuda:0")
        if inputs["input_ids"].shape[1] > 2048:
            raise ValueError("Evidence exceeds the reporting prompt limit.")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False, num_beams=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        tokens = output[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
        result = {
            "evidence_sha256": content_hash(evidence),
            "messages": messages, "raw_response": raw,
            "reporter_provenance": dict(self.provenance),
            "generation_seconds": time.perf_counter() - start,
            "generated_tokens": len(tokens),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        try:
            result["report"] = validate_report(raw, evidence)
            result["status"] = "accepted"
        except ValueError as exc:
            result.update(status="rejected", report=None, validation_error=str(exc))
        return result
