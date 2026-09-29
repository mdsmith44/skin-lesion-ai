"""Shared inference utilities for Stage 8 adapted SmolVLM models."""

import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from nemo_automodel._transformers import (
    NeMoAutoModelForImageTextToText,
)
from nemo_automodel.components._peft.lora import (
    PeftConfig,
    apply_lora_to_linear_modules,
)
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
)


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def load_adapted_model(checkpoint: str | Path):
    """Load base SmolVLM and restore a NeMo LoRA adapter."""

    from nemo_automodel.components.distributed.init_utils import (
        initialize_distributed,
    )

    checkpoint = Path(checkpoint)

    initialize_distributed(
        backend="nccl",
        timeout_minutes=10,
    )

    model = NeMoAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_liger_kernel=False,
        force_hf=True,
    ).to("cuda")

    model_dir = checkpoint / "model"

    with open(model_dir / "adapter_config.json") as f:
        hf_peft = json.load(f)

    with open(model_dir / "automodel_peft_config.json") as f:
        automodel_peft = json.load(f)

    peft_dict = {
        "dim": hf_peft["r"],
        "alpha": hf_peft["lora_alpha"],
        **automodel_peft,
    }

    peft_config = PeftConfig.from_dict(peft_dict)

    matched = apply_lora_to_linear_modules(
        model,
        peft_config,
    )

    print(f"LoRA modules reconstructed: {matched}")

    checkpoint_config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(checkpoint),
        model_save_format="safetensors",
        model_cache_dir="",
        model_repo_id=MODEL_ID,
        save_consolidated=False,
        is_peft=True,
    )

    checkpointer = Checkpointer(
        config=checkpoint_config,
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
    )

    checkpointer.load_model(
        model,
        str(model_dir),
    )

    model.eval()

    return model


def load_processor():
    """Use exactly the image preprocessing used during training."""

    return AutoProcessor.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        do_image_splitting=False,
        size={"longest_edge": 512},
    )


def generate_prediction(model, processor, sample):
    """Generate from image + instruction only; never provide the target."""

    messages = sample.evaluation_messages()

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    generated_ids = outputs[0, input_length:]

    return processor.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()
