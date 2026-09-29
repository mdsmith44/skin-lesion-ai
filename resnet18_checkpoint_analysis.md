# Historical Stage 3/4 ResNet18 checkpoint lineage

The notebook code establishes that **`resnet18_full_finetuned.pt` contains a
standard torchvision ResNet18 with a seven-class head, initialized from the
unweighted partial-fine-tuning checkpoint and then trained for five additional
epochs with the entire network unfrozen.** The training lineage below comes
from notebook source and saved outputs. Stage 10 subsequently verified and
used the persistent checkpoint; the [Stage 10 deployment record](docs/stage10_deployment.md)
is canonical for serving provenance.

## How the checkpoint was created

The training lineage in [Stage 3](notebooks/03_transfer_learning.ipynb) is:

| Phase | Starting weights | Trainable parameters | Optimizer | Epochs |
|---|---|---|---|---:|
| Classifier training | ImageNet `ResNet18_Weights.DEFAULT`; new classification head | `fc` | Adam, `lr=1e-3` | 5 |
| Partial fine-tuning | Previous phase | `layer4` and `fc` | New Adam, `lr=1e-4` | 5 |
| Full fine-tuning | Saved `resnet18_finetuned.pt` | Entire network | New Adam, `lr=1e-5` | 5 |

All three phases use **unweighted `nn.CrossEntropyLoss()`**. The class-weighted experiment is a separate branch of the notebook and does not supply the full model’s starting weights.

Training uses the existing training partition, batch size 32, shuffled batches, and:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

Images are opened with PIL and converted to RGB.

The full network contains **11,180,103 parameters**, all trainable during the final phase. The saved checkpoint is the model **after the fifth full-fine-tuning epoch**, not an automatically restored best epoch. The notebook has no early-stopping or best-checkpoint selection logic. Its final saved validation output is accuracy 0.796, balanced accuracy 0.601, and macro F1 0.602.

[Stage 4](notebooks/04_model_evaluation.ipynb) subsequently compares the
three saved models using validation macro F1 and selects `"Full fine-tune"`.

## What is stored in the file

The save statement in [Stage 3](notebooks/03_transfer_learning.ipynb) writes
this dictionary:

```python
{
    "model_state_dict": full_ft_model.state_dict(),
    "class_to_idx": class_to_idx,
    "architecture": "resnet18",
    "training_strategy": "full network fine-tuning",
    "val_accuracy": 0.796,
    "val_balanced_accuracy": 0.601,
    "val_macro_f1": 0.602,
}
```

This includes model parameters and buffers, including BatchNorm running statistics. It does **not** include the instantiated model, preprocessing, optimizer state, RNG state, or training history. The validation metadata values are hard-coded rounded numbers.

## Architecture and class mapping required for inference

Reconstruct:

```python
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 7)
```

The final layer is `Linear(512, 7)` with bias. Loading the saved state restores the entire network; no ImageNet download or earlier checkpoint is needed.

Both notebooks explicitly use:

| Output index | Label |
|---:|---|
| 0 | `akiec` |
| 1 | `bcc` |
| 2 | `bkl` |
| 3 | `df` |
| 4 | `mel` |
| 5 | `nv` |
| 6 | `vasc` |

For a reusable loader, read `class_to_idx` from the checkpoint and verify it matches this mapping. Stage 4 instead reconstructs the mapping independently and does not check the saved one.

## Exact evaluation preprocessing

[Stage 4’s transform](notebooks/04_model_evaluation.ipynb) is:

1. Open the image with PIL and convert to **RGB**.
2. Resize directly to **224 × 224**.
3. `ToTensor()` converts the usual 8-bit image to a floating-point `[3, 224, 224]` tensor scaled to `[0, 1]`.
4. Normalize channels with ImageNet mean and standard deviation.
5. Add a batch dimension for single-image inference: `[1, 3, 224, 224]`.

There is no center crop, aspect-ratio-preserving resize, or evaluation augmentation. Do not substitute the pretrained weights’ default transform pipeline. The notebook leaves interpolation and antialias settings at torchvision defaults.

## Minimal inference code

This illustrative code mirrors the Stage 4 loader and prediction logic with
explicit checkpoint-mapping verification. Supply the path to the external
checkpoint through `RESNET18_CHECKPOINT`; the checkpoint is not stored in Git.
The example was not executed as part of this document cleanup.

```python
from pathlib import Path
import os

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

checkpoint_path = Path(os.environ["RESNET18_CHECKPOINT"]).expanduser().resolve(strict=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

expected_mapping = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}
class_to_idx = checkpoint["class_to_idx"]
assert checkpoint["architecture"] == "resnet18"
assert class_to_idx == expected_mapping
idx_to_class = {idx: label for label, idx in class_to_idx.items()}

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model = model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def predict(image_path):
    with Image.open(image_path) as image:
        inputs = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())
    return {
        "predicted_class": idx_to_class[predicted_index],
        "softmax_scores": {
            idx_to_class[i]: float(value)
            for i, value in enumerate(probabilities.cpu().tolist())
        },
    }
```

Stage 4 uses seven-class **argmax**, not a special melanoma threshold. Its
threshold experiments do not establish a final alternative decision rule.
Softmax scores are **uncalibrated model scores, not clinical probabilities**.

## Later artifact verification

Stage 10 used the persistent checkpoint at
`/workspace/storage/skin-lesion-ai/models/resnet18_full_finetuned.pt`, with
SHA-256 `fd08a89ff6e459e3321b3b1ba5cdd0d3650289e28f4b78739d96b775dc6df73f`.
The tracked [Stage 10 ONNX exporter](src/stage10/export_resnet18_onnx.py)
loaded that checkpoint through the strict ResNet18 inference loader and
successfully exported the ONNX model used to build the TensorRT deployment
engine. See the [deployment record](docs/stage10_deployment.md) for the derived
artifact hashes and runtime path. This verifies the checkpoint used for Stage
10 inference; it does not reproduce the original Stage 3 training run.

## Important discrepancies and limits

- **Misleading comment:** the full-fine-tuning initialization cell says “Reload the class-weighted model,” but the [actual load path](notebooks/03_transfer_learning.ipynb) is `resnet18_finetuned.pt`, the **unweighted** checkpoint.
- **BatchNorm behavior:** the earlier phases freeze parameters through `requires_grad=False`, but `train_one_epoch()` calls `model.train()` globally. Consequently, BatchNorm running statistics can still update in nominally frozen layers. Reloading the complete state dictionary preserves those buffers.
- **Reproducibility:** Stage 3 contains no explicit training RNG seed. The fixed split seed does not make training bit-for-bit reproducible.
- **Path assumptions:** both notebooks use `Path.cwd().parent`, assuming execution from `notebooks/`, and open saved `image_path` values directly. A root-level inference script should use explicit paths and resolve any historical image paths to current storage.
- The historical recorded environment lists torch `2.14.0`, torchvision `0.29.0`, and Pillow `12.3.0`; Stage 3’s saved output reports torch `2.14.0+cu130`. Stage 10 later loaded the checkpoint in its NeMo environment, but that does not establish a bit-for-bit recreation of the Stage 3 training environment.
