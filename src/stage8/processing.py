"""SmolVLM processor and batch-size-one, assistant-only loss preparation."""

import torch
from transformers import AutoProcessor

from .dataset import MultimodalSample

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def load_processor():
    """Use the persistent HF cache and one 512-pixel image view."""
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 512}
    return processor


def prepare_evaluation_inputs(sample: MultimodalSample, processor):
    """Return CPU tensors containing no target answer or loss labels."""
    prompt = processor.apply_chat_template(
        sample.evaluation_messages(), tokenize=False, add_generation_prompt=True,
    )
    return processor(text=[prompt], images=[[sample.image]], return_tensors="pt")


class SmolVLMTrainingCollator:
    """Prepare a single sample; fail instead of silently truncating image tokens.

    Labels equal input IDs only for the assistant completion (including its
    end-of-turn formatting). -100 tells the causal-language loss to ignore
    image tokens, the user instruction, and the assistant role prefix.
    The model shifts labels internally; do not shift them a second time here.
    """

    def __init__(self, processor, max_sequence_length: int = 512):
        self.processor = processor
        self.max_sequence_length = max_sequence_length

    def __call__(self, samples: list[MultimodalSample]):
        if len(samples) != 1:
            raise ValueError("The initial 4 GB smoke-test collator requires batch size 1.")
        sample = samples[0]
        prompt = prepare_evaluation_inputs(sample, self.processor)
        full_text = self.processor.apply_chat_template(
            sample.training_messages(), tokenize=False, add_generation_prompt=False,
        )
        batch = self.processor(
            text=[full_text], images=[[sample.image]], return_tensors="pt",
        )
        prefix_length = prompt["input_ids"].shape[1]
        ids = batch["input_ids"]
        if ids.shape[1] > self.max_sequence_length:
            raise ValueError("Sequence exceeds the smoke-test limit; refusing to truncate.")
        if not torch.equal(prompt["input_ids"], ids[:, :prefix_length]):
            raise ValueError("Chat-template token prefix mismatch; answer masking is unsafe.")
        labels = ids.clone()
        labels[:, :prefix_length] = -100
        labels[batch["attention_mask"] == 0] = -100
        if not (labels != -100).any():
            raise ValueError("No assistant tokens remain for supervision.")
        batch["labels"] = labels
        return batch
