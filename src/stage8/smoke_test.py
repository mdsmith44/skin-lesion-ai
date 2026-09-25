"""Run one complete NeMo/SmolVLM LoRA update, using training data only.

From the mounted repository root inside the NeMo container:
    python -m src.stage8.smoke_test --config configs/stage8/smolvlm_lora.yaml
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import nemo_automodel
import torch
import transformers
import yaml

from .dataset import HAM10000MultimodalDataset
from .model import load_lora_model
from .processing import MODEL_ID, SmolVLMTrainingCollator, load_processor


def run(config_path: Path, project_root: Path):
    config = yaml.safe_load(config_path.read_text())
    if config["model_id"] != MODEL_ID or config["split"] != "train":
        raise ValueError("This smoke test requires SmolVLM-256M and the train split.")
    if config["batch_size"] != 1 or config["steps"] != 1:
        raise ValueError("This initial smoke test supports batch size 1 and exactly one step.")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    dataset = HAM10000MultimodalDataset(project_root, "train")
    if not 1 <= config["subset_size"] <= len(dataset):
        raise ValueError("Invalid training subset size.")
    # A fixed prefix of the existing training CSV; no new split or label balancing.
    subset_ids = [dataset[i].image_id for i in range(config["subset_size"])]
    sample = dataset[0]
    processor = load_processor()
    batch = SmolVLMTrainingCollator(processor, config["max_sequence_length"])([sample])
    supervised = batch["labels"][0] != -100
    completion = processor.tokenizer.decode(batch["labels"][0, supervised])
    if completion.strip() != sample.response + "<end_of_utterance>":
        raise RuntimeError(f"Unexpected supervised completion: {completion!r}")

    torch.cuda.reset_peak_memory_stats()
    model, adapter_report = load_lora_model(config["lora_rank"], config["lora_alpha"])
    load_peak = torch.cuda.max_memory_allocated()
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    before = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    optimizer = torch.optim.AdamW(parameters.values(), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", init_scale=config["initial_loss_scale"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Drop unused loader allocations before measuring the complete training step.
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    gpu_batch = {key: value.to("cuda") for key, value in batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(**gpu_batch)
        loss = output.loss
    if not torch.isfinite(loss).item():
        raise RuntimeError("Nonfinite training loss.")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in parameters.values()):
        raise RuntimeError("Missing or nonfinite adapter gradients; no update accepted.")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen base unexpectedly received gradients.")
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters.values(), max_norm=1.0)
    if not torch.isfinite(gradient_norm).item() or gradient_norm.item() <= 0:
        raise RuntimeError("Expected a finite, nonzero adapter gradient norm.")
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if scaler.get_scale() < scale_before:
        raise RuntimeError("GradScaler skipped the optimizer update.")
    state = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    changed = [name for name in state if not torch.equal(state[name], before[name])]
    if not changed or any(not torch.isfinite(p).all().item() for p in state.values()):
        raise RuntimeError("No finite LoRA parameter update was verified.")
    if not optimizer.state:
        raise RuntimeError("Optimizer state was not initialized.")

    report = {
        "status": "passed",
        "purpose": "One-step training feasibility only; not a quality evaluation",
        "config": config,
        "versions": {"nemo_automodel": nemo_automodel.__version__,
                     "torch": torch.__version__, "transformers": transformers.__version__,
                     "cuda": torch.version.cuda},
        "model_revision": getattr(model.config, "_commit_hash", None),
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "subset_image_ids": subset_ids,
        "trained_image_id": sample.image_id,
        "supervised_completion": completion,
        "input_shapes": {key: list(value.shape) for key, value in batch.items()},
        "image_processing": {"do_image_splitting": False, "longest_edge": 512},
        "lora": adapter_report,
        "loss": loss.item(),
        "gradient_norm_before_clipping": gradient_norm.item(),
        "changed_adapter_tensors": len(changed),
        "adapter_tensors": len(state),
        "optimizer_state_entries": len(optimizer.state),
        "step_seconds": elapsed,
        "load_peak_allocated_bytes": load_peak,
        "step_peak_allocated_bytes": peak_allocated,
        "step_peak_reserved_bytes": peak_reserved,
        "memory_note": "PyTorch allocator only; excludes some driver/library and other-process memory",
    }
    output_dir = project_root / config["output_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    # Native NeMo adapter tensors, not a Hugging Face PEFT export or resumable trainer checkpoint.
    checkpoint_path = output_dir / "adapter.pt"
    torch.save({"adapter_state_dict": state, "model_id": MODEL_ID,
                "model_revision": report["model_revision"], "lora": adapter_report}, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if set(reloaded["adapter_state_dict"]) != set(state) or any(
        not torch.equal(state[name], reloaded["adapter_state_dict"][name]) for name in state
    ):
        raise RuntimeError("Saved adapter tensor verification failed.")
    report["adapter_save_verified"] = True
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Artifacts: {output_dir}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path("configs/stage8/smolvlm_lora.yaml"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root)


if __name__ == "__main__":
    main()
