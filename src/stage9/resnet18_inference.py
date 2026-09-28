"""Reload the Stage 3/4 classifier without training or pretrained downloads.

Scores are uncalibrated model softmax scores, not clinical probabilities.
The API accepts an image path; dataset selection belongs to the caller.
"""

import copy
import hashlib
from pathlib import Path

import PIL
from PIL import Image
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import resnet18


DEFAULT_CHECKPOINT = Path(
    "/workspace/storage/skin-lesion-ai/models/resnet18_full_finetuned.pt"
)
CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}


def stage4_eval_preprocess():
    """Return the exact Stage 4 evaluation transform without loading a model."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class ResNet18Inference:
    """Load once and call ``predict(image_path)`` for JSON-compatible evidence.

    Architecture and exact label indices are checked before loading weights.
    Full state loading is strict, including the seven-class head and BN buffers.
    """

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
        path = Path(checkpoint_path).expanduser().resolve(strict=True)
        self.device = torch.device(
            device if device is not None else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            handle.seek(0)
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)

        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a Stage 3 checkpoint dictionary.")
        if checkpoint.get("architecture") != "resnet18":
            raise ValueError("Checkpoint architecture must be 'resnet18'.")
        mapping = checkpoint.get("class_to_idx")
        if (not isinstance(mapping, dict) or mapping != CLASS_TO_IDX
                or any(type(value) is not int for value in mapping.values())):
            raise ValueError(f"Checkpoint class_to_idx must equal {CLASS_TO_IDX}.")
        if not isinstance(checkpoint.get("model_state_dict"), dict):
            raise ValueError("Checkpoint is missing model_state_dict.")

        self._model = resnet18(weights=None)
        self._model.fc = nn.Linear(self._model.fc.in_features, len(CLASS_NAMES))
        self._model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self._model.to(self.device).eval()
        self._model.requires_grad_(False)

        # Exact Stage 4 sequence; retain torchvision Resize defaults.
        self._preprocess = stage4_eval_preprocess()
        resize = self._preprocess.transforms[0]
        self._provenance = {
            "architecture": "resnet18",
            "checkpoint_path": str(path),
            "checkpoint_sha256": digest,
            "class_to_idx": dict(mapping),
            "training_strategy": checkpoint.get("training_strategy"),
            "device": str(self.device),
            "dtype": str(next(self._model.parameters()).dtype),
            "software": {
                "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
                "pillow": PIL.__version__,
            },
            "preprocessing": {
                "reference": "notebooks/04_model_evaluation.ipynb",
                "steps": ["PIL RGB conversion", "Resize", "ToTensor", "Normalize"],
                "resize_hw": [224, 224],
                "resize_interpolation": resize.interpolation.value,
                "resize_antialias": resize.antialias,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "augmentation": False,
            },
            "decision_rule": "argmax over seven class softmax scores",
        }

    def predict(self, image_path):
        """Return prediction, all seven scores, and model/input provenance.

        No reference diagnosis or metadata is used to compute the prediction.
        Dataset membership is not inferred from the image filename.
        """
        path = Path(image_path).expanduser().resolve(strict=True)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            handle.seek(0)
            with Image.open(handle) as image:
                inputs = self._preprocess(image.convert("RGB"))
        self._model.eval()
        with torch.inference_mode():
            logits = self._model(inputs.unsqueeze(0).to(self.device))
            scores = torch.softmax(logits, dim=1)[0].cpu()
        if scores.shape != (7,) or not torch.isfinite(scores).all():
            raise RuntimeError("Classifier produced invalid softmax scores.")
        return {
            "predicted_class": CLASS_NAMES[int(scores.argmax().item())],
            "softmax_scores": dict(zip(CLASS_NAMES, scores.tolist())),
            "score_interpretation": "Uncalibrated model scores; not clinical probabilities.",
            "intended_use": "Research and education only; not clinical diagnosis.",
            "input": {"image_path": str(path), "image_sha256": digest},
            "model_provenance": copy.deepcopy(self._provenance),
        }
