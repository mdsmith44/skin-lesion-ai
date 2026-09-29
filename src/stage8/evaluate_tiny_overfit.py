"""Evaluate the Stage 8 seven-image tiny-overfit diagnostic."""

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

from src.stage8.dataset import HAM10000MultimodalDataset


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

PROJECT_ROOT = Path("/workspace/skin-lesion-ai")

CHECKPOINT = Path(
    "/workspace/storage/skin-lesion-ai/outputs/stage8/"
    "tiny-overfit-v2/epoch_99_step_99"
)

# Exactly the seven training examples used in the tiny-overfit run.
CASES = [
    (6771, "akiec", "ISIC_0029417"),
    (1721, "bcc",   "ISIC_0028155"),
    (0,    "bkl",   "ISIC_0026769"),
    (771,  "df",    "ISIC_0027008"),
    (848,  "mel",   "ISIC_0025964"),
    (46,   "nv",    "ISIC_0024698"),
    (1622, "vasc",  "ISIC_0029486"),
]


def load_adapted_model():
    """Load base SmolVLM and restore the tiny-overfit LoRA adapter."""

    from nemo_automodel.components.distributed.init_utils import (
        initialize_distributed,
    )

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

    model_dir = CHECKPOINT / "model"

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
        checkpoint_dir=str(CHECKPOINT),
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


def main():
    dataset = HAM10000MultimodalDataset(
        project_root=PROJECT_ROOT,
        split="train",
    )

    # Verify that the indices still refer to exactly the examples
    # used to construct the tiny-overfit experiment.
    samples = []

    for index, expected_label, expected_image_id in CASES:
        sample = dataset[index]

        if sample.class_abbreviation != expected_label:
            raise RuntimeError(
                f"Index {index}: expected label {expected_label}, "
                f"found {sample.class_abbreviation}"
            )

        if sample.image_id != expected_image_id:
            raise RuntimeError(
                f"Index {index}: expected image {expected_image_id}, "
                f"found {sample.image_id}"
            )

        samples.append(sample)

    print("Tiny-overfit cases:")
    for sample in samples:
        print(
            f"  {sample.class_abbreviation:5s} "
            f"{sample.image_id}"
        )

    processor = load_processor()
    model = load_adapted_model()

    print()
    print("Free-generation predictions")
    print("=" * 72)

    correct = 0

    for sample in samples:
        prediction = generate_prediction(
            model,
            processor,
            sample,
        )

        expected = sample.class_abbreviation
        normalized = prediction.strip().lower()
        is_correct = normalized == expected

        correct += int(is_correct)

        print(
            f"{sample.image_id}  "
            f"true={expected:5s}  "
            f"generated={prediction!r:12s}  "
            f"{'PASS' if is_correct else 'FAIL'}"
        )

    print("=" * 72)
    print(f"Exact generation: {correct}/{len(samples)}")

    if correct == len(samples):
        print("TINY-OVERFIT CHECK: PASSED")
    else:
        print("TINY-OVERFIT CHECK: FAILED")


if __name__ == "__main__":
    main()
