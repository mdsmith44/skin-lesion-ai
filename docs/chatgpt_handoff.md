# Skin Lesion AI — ChatGPT Project Handoff

## Purpose

This document provides context for a new ChatGPT conversation so that development of the Skin Lesion AI project can continue without reconstructing the earlier stages from scratch.

The project is an educational/research project exploring computer vision, vision-language models, and NVIDIA AI tooling using the HAM10000 dermatoscopic image dataset.

**This project is not intended for clinical diagnosis, medical decision-making, or patient care.**

---

# How ChatGPT Should Help

Act as a technical mentor and pair programmer for this project.

The user is learning computer vision, multimodal AI, and NVIDIA tooling while building the project.

Use an incremental approach:

1. Explain the concept being introduced.
2. Explain why it matters to this project.
3. Provide a small amount of code or a concrete step.
4. Let the user run it.
5. Ask for or inspect the result.
6. Continue to the next step.

Do not dump an entire implementation at once unless explicitly requested.

When introducing NVIDIA tools, explain how they relate to the equivalent PyTorch / Hugging Face concepts already used in the project.

Favor understanding and reproducibility over simply making the code run.

---

# Project Goal

The project investigates the progression from conventional computer vision to multimodal AI:

```text
HAM10000
    |
    v
Dataset exploration
    |
    v
Baseline CNN
    |
    v
ResNet-18 transfer learning
    |
    v
Medical-ML evaluation
    |
    v
Grad-CAM explainability
    |
    v
Zero-shot VLM experiments
    |
    v
Multimodal instruction dataset
    |
    v
NVIDIA NeMo + PEFT / LoRA
    |
    v
Nemotron / structured reporting
    |
    v
NVIDIA deployment tooling
```

Stages 1–7 are complete.

The next stage is:

**Stage 8 — NVIDIA NeMo + parameter-efficient VLM adaptation**

---

# Development Environment

The project uses a Conda environment named:

```text
skin-lesion-ai
```

Python:

```text
3.11
```

Primary framework:

```text
PyTorch
```

The original development machine used:

```text
NVIDIA GeForce RTX 3050 Laptop GPU
4 GB VRAM
```

PyTorch successfully detected CUDA.

A collaborator's hardware may differ, so GPU capability and memory should be checked before choosing a training strategy.

Do not assume that a model or fine-tuning configuration that works on a large GPU will fit on a 4 GB GPU.

---

# Repository Structure

Important files include:

```text
skin-lesion-ai/
├── data/
│   ├── raw/
│   │   └── ham10000/
│   └── processed/
│       ├── metadata_splits.csv
│       ├── multimodal_dataset.csv
│       └── multimodal/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
├── models/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_gradcam_explainability.ipynb
│   ├── 06_vision_language_model.ipynb
│   └── 07_multimodal_dataset.ipynb
├── results/
├── environment.yml
├── environment-lock.yml
└── README.md
```

Raw HAM10000 images and trained PyTorch checkpoints are excluded from Git.

---

# Dataset

Dataset:

**HAM10000**

Total images:

```text
10,015
```

Unique lesions:

```text
7,470
```

Classes:

```text
akiec
bcc
bkl
df
mel
nv
vasc
```

Class counts:

| Class | Images |
|---|---:|
| nv | 6705 |
| mel | 1113 |
| bkl | 1099 |
| bcc | 514 |
| akiec | 327 |
| vasc | 142 |
| df | 115 |

The dataset is severely imbalanced.

---

# Data Splitting Decision

HAM10000 can contain multiple images of the same lesion.

Therefore, images were **not independently randomly split**.

Instead, splitting was performed at the lesion level so that the same lesion cannot appear in multiple partitions.

The split uses a fixed random seed:

```text
42
```

Result:

| Split | Images |
|---|---:|
| Train | 7002 |
| Validation | 1532 |
| Test | 1481 |

There is no lesion overlap between train, validation, and test.

HAM10000 does not provide a usable patient identifier in the metadata used here.

Therefore:

**lesion-level independence is enforced, but patient-level independence cannot be guaranteed.**

Do not replace this split with a new random image-level split.

The existing split is stored in:

```text
data/processed/metadata_splits.csv
```

---

# Stage 2 — Baseline CNN

A small CNN was trained as the initial baseline.

Approximate validation performance:

```text
Accuracy:          72.7%
Balanced accuracy: 34.3%
Macro F1:          36.9%
```

The large gap between accuracy and balanced accuracy demonstrated strong majority-class effects.

---

# Stage 3 — ResNet-18 Transfer Learning

ResNet-18 was selected as the transfer-learning architecture.

Experiments included:

1. frozen ResNet feature extractor
2. layer4 + classifier fine-tuning
3. class-weighted layer4 + classifier fine-tuning
4. full-network fine-tuning

Important validation results:

```text
Partial fine-tuning
Accuracy:          78.5%
Balanced accuracy: 61.1%
Macro F1:          59.0%

Weighted partial fine-tuning
Accuracy:          72.3%
Balanced accuracy: 64.7%
Macro F1:          59.1%

Full fine-tuning
Accuracy:          79.6%
Balanced accuracy: 60.1%
Macro F1:          60.2%
```

The full fine-tuned model was selected using validation macro F1 before evaluating the test set.

---

# Stage 4 — Final Test Evaluation

Final held-out test performance:

```text
Accuracy:          82.85%
Balanced accuracy: 64.30%
Macro precision:   71.84%
Macro recall:      64.30%
Macro F1:          66.58%
Melanoma AUC:      0.905
```

Melanoma test performance was approximately:

```text
Precision: 0.626
Recall:    0.588
F1:        0.606
```

Do not tune future models or prompts against this test set.

Use validation data for model-development decisions.

---

# Stage 5 — Grad-CAM

Grad-CAM was applied to the final ResNet-18.

Target layer:

```python
model.layer4[-1]
```

Representative cases included:

```text
melanoma true positive
melanoma false negative
melanoma false positive
correctly classified nevus
```

The comparison figure is stored at:

```text
results/gradcam_case_comparison.png
```

Important interpretation:

Grad-CAM shows image regions that influenced model behavior.

It does **not** establish that those regions are clinically meaningful or prove that the model is reasoning like a dermatologist.

---

# Stage 6 — Vision-Language Model Baseline

Two small general-purpose VLMs were evaluated.

Models:

```text
HuggingFaceTB/SmolVLM-256M-Instruct

HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

The experiments investigated:

1. natural-language image description
2. zero-shot HAM10000 classification

A fixed balanced evaluation sample contained:

```text
35 images
5 images per class
```

The same frozen classification prompt was used when comparing models.

Do not repeatedly modify the prompt based on performance on this evaluation sample.

---

# Stage 6 Results

### SmolVLM 256M

```text
Strict accuracy:        14.3%
Semantic accuracy:      17.1%
Invalid outputs:        5
Average inference time: 0.62 sec/image
```

### SmolVLM2 500M

```text
Strict accuracy:        11.4%
Invalid outputs:        0
Average inference time: 0.68 sec/image
```

The 500M model predicted:

```text
akiec: 27 / 35
bkl:    8 / 35
all other classes: 0
```

This demonstrated severe class collapse.

Important conclusion:

**A larger general-purpose VLM did not automatically provide better performance on this specialized medical-image classification task.**

The VLMs also produced plausible-sounding but unsupported medical language during description experiments.

Therefore:

**Fluent language generation should not be interpreted as grounded clinical reasoning.**

---

# Stage 7 — Multimodal Dataset

HAM10000 was transformed into an instruction-tuning representation.

Original representation:

```text
(image, class)
```

New representation:

```text
(image, instruction, response)
```

Example:

```text
Image:
ISIC_0026993.jpg

Instruction:
What lesion category is shown in this dermatoscopic image?

Response:
melanoma
```

Four equivalent instruction templates are used.

One instruction is randomly assigned per image using:

```text
seed = 42
```

There remains exactly one training record per image.

Do not duplicate every image four times simply because four instruction templates exist.

---

# Stage 7 Files

Master dataset:

```text
data/processed/multimodal_dataset.csv
```

Split files:

```text
data/processed/multimodal/train.csv
data/processed/multimodal/val.csv
data/processed/multimodal/test.csv
```

Expected sizes:

```text
train: 7002
val:   1532
test:  1481
total: 10015
```

The original lesion-aware split is preserved.

Do not create a new train/validation/test split for VLM training.

---

# Important Training / Evaluation Distinction

The multimodal CSVs contain target responses for all partitions.

During **training**, the model receives information needed to calculate loss against the target response.

During **validation/test inference**, the target response must not be included in the model prompt.

Evaluation should conceptually be:

```text
image + instruction
        |
        v
      model
        |
        v
generated response
        |
        v
compare with stored target
```

Never expose the ground-truth response to the model during evaluation.

---

# PEFT and LoRA

PEFT means:

**Parameter-Efficient Fine-Tuning**

LoRA means:

**Low-Rank Adaptation**

LoRA is one PEFT technique.

Instead of updating the full pretrained weight matrix:

```text
W
```

LoRA learns a small low-rank update:

```text
W' = W + BA
```

where `B` and `A` contain far fewer trainable parameters than `W`.

This is important because full VLM fine-tuning may require substantially more GPU memory for:

- model weights
- activations
- gradients
- optimizer states

LoRA reduces the number of trainable parameters and optimizer state.

QLoRA additionally uses a quantized base model with LoRA adapters.

---


# Stage 8 — NeMo AutoModel + Docker

Stage 8 deliberately uses Docker to gain practical experience with containerized NVIDIA ML workflows and to isolate the NeMo environment from the existing `skin-lesion-ai` Conda environment.

### Docker environment

- Host development environment: WSL2
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4 GB
- Docker: Docker Desktop with WSL2 integration
- Docker GPU passthrough: verified
- NeMo AutoModel image: `nvcr.io/nvidia/nemo-automodel:26.06.00`
- NeMo AutoModel version inside container: `0.5.0+d02f49cb`
- Container Python: 3.12.3
- Container PyTorch: NVIDIA build with CUDA support
- GPU is visible to PyTorch inside the container.

The host Conda environment remains the working environment for the earlier project stages. Do not install NeMo into that environment unless the project direction explicitly changes.

### Docker mounts

The NeMo container is launched with the project directory and Hugging Face cache bind-mounted:

- Host project → `/workspace/skin-lesion-ai`
- Host Hugging Face cache → `/root/.cache/huggingface`

This allows the container to access HAM10000 directly and allows downloaded Hugging Face models to persist when disposable `--rm` containers are removed.

Current launch pattern:

```bash
docker run --rm -it \
    --gpus all \
    -v "$(pwd):/workspace/skin-lesion-ai" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -w /workspace/skin-lesion-ai \
    nvcr.io/nvidia/nemo-automodel:26.06.00 \
    bash
```

### Verified NeMo/SmolVLM status

NeMo AutoModel successfully loads:

`HuggingFaceTB/SmolVLM-256M-Instruct`

through:

`NeMoAutoModelForImageTextToText.from_pretrained(...)`

The resulting Hugging Face architecture is:

`Idefics3ForConditionalGeneration`

A dedicated `SmolVLMForConditionalGeneration` class is NOT available under `nemo_automodel.components.models` in this AutoModel build. Do not assume one is required; the generic Hugging Face-compatible NeMo loader works.

GPU memory measurements in FP16:

- Model loaded on GPU: approximately 0.49 GB allocated
- Single-image multimodal inference peak: approximately 0.83 GB

A Stage 8 pre-fine-tuning inference test was successfully run on `ISIC_0026993`.

- Ground truth: `mel`
- SmolVLM response: `Bkl.`
- Semantically normalized prediction: `bkl`

This reproduces the Stage 6 zero-shot failure on the same image and provides a useful before-fine-tuning comparison.

### Immediate next objective

Transition from environment exploration to repository-based Stage 8 development.

Create a small, reproducible NeMo training scaffold consisting approximately of:

- `configs/stage8/smolvlm_lora.yaml`
- `src/stage8/dataset.py`
- `src/stage8/inference.py`
- `notebooks/08_nemo_finetuning.ipynb`
- Stage 8 checkpoint/output location

First implement a HAM10000 multimodal dataset adapter using the existing Stage 7 data.

Preserve the existing lesion-aware train/validation/test splits. Do not create a new split and do not use the test set for model development.

Then create the smallest practical LoRA smoke test:

1. SmolVLM-256M
2. Very small training subset
3. Local batch size 1
4. Minimal number of training steps
5. PEFT/LoRA
6. Mixed precision where appropriate
7. Measure peak GPU memory
8. Determine whether one complete forward → loss → backward → optimizer step fits within the 4 GB RTX 3050

The initial objective is not model quality. The objective is to establish that NeMo + SmolVLM + HAM10000 + LoRA training works end-to-end on the available hardware.

If the 4 GB GPU cannot support the training smoke test, preserve the same Docker/NeMo experiment design and move training to larger GPU hardware rather than abandoning the NeMo workflow.



# Stage 9 — Nemotron

Do not use Nemotron merely to say that the project used another NVIDIA product.

First identify a legitimate role for a language model downstream of the vision system.

A possible architecture is:

```text
dermatoscopic image
        |
        v
specialized vision / multimodal model
        |
        v
structured prediction
        |
        v
Nemotron
        |
        v
structured natural-language explanation/report
```

The language model should not invent unsupported clinical findings.

Potential work includes:

- structured reporting
- explanation formatting
- constrained summarization
- generation from structured model outputs

Any medical language should remain clearly framed as educational/research output rather than clinical advice.

---

# Stage 10 — NVIDIA Deployment

After the modeling pipeline is stable, investigate NVIDIA deployment tooling.

Potential technologies include:

```text
NVIDIA NIM
NVIDIA Triton Inference Server
```

The goal is to understand the transition from:

```text
research notebook
```

to:

```text
reproducible inference service
```

Possible topics:

- model packaging
- inference APIs
- GPU serving
- batching
- latency
- throughput
- containerization
- reproducible deployment

Do not begin deployment work until the model and evaluation pipeline are stable.

---

# Collaboration / Git Workflow

The repository has a checkpoint tag:

```text
v0.1-pre-nemo
```

This represents completion of Stages 1–7 before introducing NVIDIA NeMo.

For Stage 8, use a feature branch rather than developing directly on `main`.

Suggested branch:

```text
feature/nemo-setup
```

Create it with:

```bash
git checkout main
git pull
git checkout -b feature/nemo-setup
```

Commit small, understandable milestones.

Examples:

```text
Add NeMo environment setup

Add multimodal NeMo inference baseline

Add LoRA training configuration

Add VLM validation metrics
```

Avoid committing:

- raw HAM10000 images
- model caches
- credentials
- API tokens
- large checkpoints

---

# Guiding Principles

Throughout the remaining project:

1. Preserve the lesion-aware split.
2. Keep test data isolated from model-development decisions.
3. Do not fabricate medical annotations that HAM10000 does not provide.
4. Distinguish classification labels from generated explanations.
5. Treat VLM language as untrusted unless grounded in available data.
6. Report balanced metrics because the dataset is highly imbalanced.
7. Compare new models against existing baselines.
8. Prefer small reproducible experiments before expensive training.
9. Explain important NVIDIA concepts rather than treating the tools as black boxes.
10. Keep the project reproducible for both collaborators.

---

# Immediate Next Instruction

The user is ready to begin **Stage 8**.

Start by helping them create a `feature/nemo-setup` branch.

Then inspect their hardware/software environment and current NVIDIA NeMo documentation before installing or changing anything.

Proceed incrementally and wait for results between major steps.