"""Single-GPU SmolVLM loading with the installed NeMo AutoModel LoRA API.

Verified against NeMo AutoModel 0.5.0+d02f49cb in image 26.06.00.
This module loads and inspects adapters; it does not run an optimizer.
"""

import torch
from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel.components._peft.lora import (
    PeftConfig,
    apply_lora_to_linear_modules,
)

from .processing import MODEL_ID


def load_lora_model(rank: int = 4, alpha: int = 8):
    """Freeze the FP16 base and add FP32 LoRA weights to text q/v projections.

    FP32 adapter weights support the usual autocast + GradScaler training
    path later. The low rank and restricted targets are initial feasibility
    settings, not hyperparameters selected for classification quality.
    """
    if rank < 1 or alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this loader in the GPU-enabled NeMo container.")
    model = NeMoAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        use_liger_kernel=False,
        use_sdpa_patching=False,
        force_hf=True,
    )
    # This container's loader was observed returning FP32 despite torch_dtype.
    # Convert BEFORE attaching LoRA, so adapter weights can remain FP32.
    model.to(device="cuda", dtype=torch.float16)
    model.config.use_cache = False
    targets = [
        name for name, module in model.named_modules()
        if ".text_model." in name
        and name.endswith((".q_proj", ".v_proj"))
        and isinstance(module, torch.nn.Linear)
    ]
    if not targets:
        raise RuntimeError("No SmolVLM language query/value projections found.")
    config = PeftConfig(
        target_modules=targets,
        dim=rank,
        alpha=alpha,
        dropout=0.0,
        lora_dtype=torch.float32,
        use_memory_efficient_lora=False,
        use_triton=False,
    )
    matched = apply_lora_to_linear_modules(model, config)
    if matched != len(targets):
        raise RuntimeError(f"Expected {len(targets)} LoRA modules, patched {matched}.")
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(
        ".text_model." not in name
        or not any(part in name for part in (".lora_A.", ".lora_B."))
        or parameter.dtype != torch.float32
        for name, parameter in trainable
    ):
        raise RuntimeError("Expected only FP32 text-model LoRA parameters to be trainable.")
    if any(p.dtype != torch.float16 for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen base parameters must remain FP16.")
    total = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for _, p in trainable)
    report = {
        "architecture": type(model).__name__,
        "lora_modules": matched,
        "rank": rank,
        "alpha": alpha,
        "trainable_parameters": trainable_count,
        "total_parameters": total,
        "trainable_percent": 100 * trainable_count / total,
    }
    return model, report
