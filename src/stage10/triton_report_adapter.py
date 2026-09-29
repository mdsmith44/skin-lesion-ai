"""Build split-free Stage 9 prompt evidence from trusted Triton classification."""

import hashlib
import math
import secrets

from src.stage9.reporting import LABELS, prompt_evidence
from src.stage10.triton_classifier import EXPECTED_ENGINE_SHA256, MODEL_NAME, MODEL_VERSION


class InvalidClassifierEvidence(ValueError):
    """The trusted classifier result does not meet the reporting input contract."""


def report_prompt_evidence(classification: dict, image_bytes: bytes) -> dict:
    """Validate the classifier result, then use Stage 9's exact prompt projection."""
    try:
        scores = classification["softmax_scores"]
        order = classification["class_order"]
        index = classification["predicted_class_index"]
        label = classification["predicted_class"]
        provenance = classification["classifier_provenance"]
        input_info = classification["input"]
        if classification.get("score_type") != "uncalibrated_softmax":
            raise ValueError("Unexpected score type")
        if (not isinstance(scores, dict) or list(scores) != list(LABELS)
                or order != list(LABELS)):
            raise ValueError("Unexpected seven-class order")
        values = list(scores.values())
        if (any(type(value) not in (int, float) or not math.isfinite(value)
                or not 0 <= value <= 1 for value in values)
                or not math.isclose(sum(values), 1.0, abs_tol=1e-6)):
            raise ValueError("Invalid softmax scores")
        if (type(index) is not int or not 0 <= index < len(LABELS)
                or label != LABELS[index]
                or index != max(range(len(LABELS)), key=values.__getitem__)):
            raise ValueError("Class does not match score argmax")
        if (not isinstance(provenance, dict)
                or provenance.get("runtime") != "Triton HTTP V2"
                or provenance.get("model_name") != MODEL_NAME
                or provenance.get("model_version") != MODEL_VERSION
                or provenance.get("expected_engine_sha256") != EXPECTED_ENGINE_SHA256
                or not isinstance(input_info, dict)):
            raise ValueError("Unexpected classifier provenance")
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        if input_info.get("image_sha256") != image_sha256:
            raise ValueError("Classifier input does not match uploaded image")
        evidence = {
            "observed_metadata": {"image_id": f"img_{secrets.token_hex(8)}"},
            "classifier": {
                "predicted_class": label,
                "top_softmax_score": format(values[index], ".6f"),
            },
        }
        return prompt_evidence(evidence)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InvalidClassifierEvidence("Classifier result is invalid for reporting.") from exc
