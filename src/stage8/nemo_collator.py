"""NeMo collator for SmolVLM/Idefics3.

Masks the entire user prompt and supervises only the actual assistant
response. This avoids NeMo AutoModel's generic BPE pattern-match fallback.
"""

import torch


def smolvlm_training_collate_fn(
    examples,
    processor,
    max_length=512,
):
    if not examples:
        raise ValueError("Cannot collate an empty batch.")

    full_conversations = []
    prompt_conversations = []

    for item in examples:
        conversation = item["conversation"]

        if len(conversation) < 2:
            raise ValueError(
                "Expected a user turn followed by an assistant turn."
            )

        if conversation[-1]["role"] != "assistant":
            raise ValueError(
                "Final conversation turn must be the assistant response."
            )

        full_conversations.append(conversation)
        prompt_conversations.append(conversation[:-1])

    prompt_lengths = []

    # Establish the exact assistant boundary independently for each sample.
    for full_conversation, prompt_conversation in zip(
        full_conversations,
        prompt_conversations,
    ):
        prompt = processor.apply_chat_template(
            prompt_conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        full = processor.apply_chat_template(
            full_conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )

        prompt_ids = prompt["input_ids"][0]
        full_ids = full["input_ids"][0]

        if len(full_ids) > max_length:
            raise ValueError(
                f"Sequence length {len(full_ids)} exceeds "
                f"max_length={max_length}."
            )

        if len(prompt_ids) >= len(full_ids):
            raise ValueError(
                "No assistant response remains after the prompt."
            )

        if not torch.equal(
            prompt_ids,
            full_ids[: len(prompt_ids)],
        ):
            raise ValueError(
                "Generation prompt is not an exact prefix of "
                "the complete training conversation."
            )

        prompt_lengths.append(len(prompt_ids))

    # Batch the complete conversations.
    encoded = processor.apply_chat_template(
        full_conversations,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
        processor_kwargs={
            "padding": True,
            "truncation": False,
        },
    )

    labels = encoded["input_ids"].clone()

    # Mask everything before the actual assistant response.
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100

    # Mask padding.
    if "attention_mask" in encoded:
        labels[encoded["attention_mask"] == 0] = -100

    # NeMo's VLM loss path expects the causal shift to be performed here:
    #
    # input[t] predicts label[t], where label[t] originally came from
    # token t+1.
    labels = labels[:, 1:]

    for key in (
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
    ):
        if key in encoded:
            encoded[key] = encoded[key][:, :-1]

    encoded["labels"] = labels

    if "pixel_values" in encoded:
        encoded["pixel_values"] = encoded["pixel_values"].to(
            dtype=torch.bfloat16
        )

    return encoded
