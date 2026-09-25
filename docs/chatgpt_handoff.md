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

Latest pacing preference: the user asked to move through the smoke test quickly rather than pause for teaching at every step. Continue with concise explanations and practical milestones; do not repeat the completed setup or smoke-test preparation.

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

The current stage is:

**Stage 8 — NVIDIA NeMo + parameter-efficient VLM adaptation**

The repository-based one-step LoRA training smoke test has passed on the 4 GB RTX 3050. Longer training and model-quality evaluation remain future work.

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

The saved CSVs were inspected during Stage 8 and contain **one fixed classification instruction**, listing the seven categories and asking for only the category abbreviation. Responses are abbreviations (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).

For example, training image `ISIC_0026769.jpg` has target response `bkl`.

An earlier version of this handoff described four randomly assigned instructions and full category names. That does not describe the saved artifacts used for the successful smoke test. Preserve the existing instructions and responses exactly; do not regenerate or duplicate records.

There remains exactly one record per image. The Stage 7 notebook currently references `INSTRUCTIONS` after its definition was commented out; rerunning it is unnecessary for Stage 8 and that reproducibility issue has not been repaired.

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

This reproduces the Stage 6 zero-shot failure on the same image. **`ISIC_0026993` belongs to the test split.** Keep this result as historical context; do not reuse the image for smoke tests, tuning, or development comparisons. The successful training smoke test below uses only a training image.

### Repository scaffold now implemented

- `configs/stage8/smolvlm_lora.yaml`: project configuration for the one-step runner, not a NeMo distributed-training recipe.
- `src/stage8/__init__.py`: package entry point.
- `src/stage8/dataset.py`: preserves saved records and splits, resolves old absolute host image paths beneath the current project root, and loads RGB images lazily. Development access is limited to train/validation.
- `src/stage8/processing.py`: cached SmolVLM processor, evaluation inputs without targets, and batch-size-one collator with assistant-only loss labels.
- `src/stage8/model.py`: generic NeMo model loader and native NeMo LoRA attachment to language-model query/value projections only.
- `src/stage8/smoke_test.py`: complete forward → loss → backward → optimizer step, integrity checks, memory measurements, and local artifact serialization.
- `notebooks/08_nemo_finetuning.ipynb`: dataset checks, processor/LoRA inspection, forward-only check, and full smoke-test execution.
- `.gitignore`: excludes `/outputs/stage8/` from Git.

A standalone `src/stage8/inference.py` has not been implemented. Evaluation-input preparation exists, but a model-quality evaluation pipeline is still future work.

### Passed training smoke test — exact recorded results

Run ID: `20260925T054227.031064Z` (2026-09-25 UTC; 2026-09-24 America/Los_Angeles).

**Status: passed. One complete LoRA training update fits on the NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM under these settings.**

Configuration saved with the run:

```yaml
model_id: HuggingFaceTB/SmolVLM-256M-Instruct
seed: 42
split: train
subset_size: 4
batch_size: 1
steps: 1
learning_rate: 0.0001
lora_rank: 4
lora_alpha: 8
max_sequence_length: 512
initial_loss_scale: 128.0
output_root: outputs/stage8
```

The fixed candidate subset consists of the first four existing training rows:

```text
ISIC_0026769
ISIC_0025661
ISIC_0031633
ISIC_0027850
```

Only the first image, `ISIC_0026769`, was used for the single optimizer step. The four-image subset does not mean four images were trained on. No new split or class balancing was introduced.

Processor and precision settings:

- Image splitting disabled; longest edge 512 pixels, with one image view.
- Frozen base weights explicitly converted to FP16; LoRA parameters stored in FP32; CUDA autocast uses FP16.
- Native NeMo LoRA rank 4, alpha 8, dropout 0; 60 text-model query/value projection modules adapted.
- Vision encoder, multimodal connector, and all original parameters remain frozen.
- PyTorch SDPA attention; optional Liger, SDPA patching, memory-efficient LoRA, and Triton paths disabled. No quantization/QLoRA.
- AdamW learning rate 0.0001; gradient norm clipped to 1.0; initial GradScaler loss scale 128.0.
- Model cache used with `local_files_only=True`.
- Prompt/image/role-prefix labels masked with `-100`; supervision covers only the answer and end-of-turn formatting. No manual label shift or silent truncation.

Exact supervised completion, represented as a JSON string:

```json
" bkl<end_of_utterance>\n"
```

| Result | Exact recorded value |
|---|---:|
| Training loss | 6.935909271240234 |
| Gradient norm before clipping | 17.58233070373535 |
| Changed adapter tensors | 120 |
| Total adapter tensors | 120 |
| Optimizer state entries | 120 |
| Training step seconds | 2.3690221450015088 |
| Model-load peak allocated bytes | 1052350976 |
| Full-step peak allocated bytes | 737883648 |
| Full-step peak reserved bytes | 1115684864 |
| Reported GPU total bytes | 4294508544 |
| Trainable parameters | 230400 |
| Total parameters including adapters | 256715328 |
| Trainable percentage | 0.08974921824691356 |

The full-step peaks are approximately **0.69 GiB allocated / 1.04 GiB reserved**. These are PyTorch allocator measurements, excluding some CUDA driver/library allocations and other-process memory. Step timing excludes model loading and artifact saving.

Tensor shapes:

```json
{
  "pixel_values": [
    1,
    1,
    3,
    512,
    512
  ],
  "pixel_attention_mask": [
    1,
    1,
    512,
    512
  ],
  "input_ids": [
    1,
    166
  ],
  "attention_mask": [
    1,
    166
  ],
  "labels": [
    1,
    166
  ]
}
```

Recorded software versions:

- nemo_automodel: `0.5.0+d02f49cb`
- torch: `2.12.0a0+0291f960b6.nv26.04.48445190`
- transformers: `5.8.1`
- cuda: `13.2`
- Model revision: `7e3e67edbbed1bf9888184d9df282b700a323964`
- Architecture: `Idefics3ForConditionalGeneration`
- Container: `nvcr.io/nvidia/nemo-automodel:26.06.00`

### Issues found and resolved

1. The installed NeMo loader returned FP32 base parameters despite the FP16 dtype request. The repository loader explicitly converts the base to FP16 **before** attaching FP32 LoRA parameters and verifies both dtypes.
2. The first probe used the default FP16 GradScaler scale of 65536. It produced nonfinite gradients and the optimizer skipped its update; zero adapter tensors changed. That probe was not a passing training test. Lowering the initial scale to 128 produced the passing run above.

The successful runner verifies finite loss and adapter gradients, no gradients on frozen parameters, a finite nonzero gradient norm, no skipped optimizer step, changed finite adapter tensors, and initialized optimizer state.

Additional checks passed: notebook schema, Python/notebook syntax, saved split/record integrity, image loading in Docker, train/validation lesion separation, exact answer masks for all seven classes, rejection of invalid batch sizes and excessive sequence lengths, and Git exclusion of output artifacts. Original train/validation/test split sizes and zero lesion overlap were also verified during the initial repository audit; no test inference or test-based development was performed in this smoke test.

### Artifacts and reproduction

Local artifacts, relative to the repository root:

```text
outputs/stage8/20260925T054227.031064Z/metrics.json
outputs/stage8/20260925T054227.031064Z/adapter.pt
```

`metrics.json` is the source of the exact results above. `adapter.pt` contains native NeMo adapter tensors and model/LoRA metadata. Reloading the saved tensors and comparing them with the in-memory tensors passed (`adapter_save_verified: true`). This does not yet verify end-to-end adapter restoration into a fresh model. It is not a Hugging Face PEFT export or a resumable trainer checkpoint; optimizer/scaler state is not saved.

Run from the repository root **inside the existing GPU-enabled NeMo container**, with the project and Hugging Face cache mounted as described above:

```bash
python -m src.stage8.smoke_test --config configs/stage8/smolvlm_lora.yaml
```

Each successful execution creates a new timestamped directory under `outputs/stage8/`. These artifacts are local and ignored by Git, so they will not appear in a fresh clone.

### Current boundary and next objective

The one-step feasibility milestone is complete. It does not establish improved classification, clinical validity, stability over a longer run, or memory requirements for other image/sequence settings. No new validation/test quality metrics were computed.

Continue from this working scaffold rather than repeating environment setup. The next development milestone can verify fresh-model adapter restoration and extend to a short multi-step training run with validation-only evaluation. Preserve lesion-aware splits, test isolation, the 4 GB constraint, existing targets, and the prohibition on fabricated medical annotations. If larger experiments exceed available VRAM, preserve the NeMo/Docker workflow when moving to larger hardware.

---

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

Current Stage 8 branch:

```text
feature/nemo-setup
```

This branch already exists and contains the smoke-test scaffold. Do not recreate it or switch back to `main` to resume work. Inspect `git status` before editing or committing so existing work is preserved.

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

The repository-based **Stage 8 one-step NeMo + SmolVLM-256M + LoRA smoke test has passed** on the 4 GB RTX 3050. The user requested faster progress through the smoke test; it is no longer necessary to pause after every teaching step.

Read the implemented files and exact results above, inspect current Git state on `feature/nemo-setup`, and continue from the completed feasibility milestone. Do not recreate the branch, regenerate Stage 7 data, reinstall NeMo in Conda, or treat the historical test image as a development example.

The next proposed milestone is adapter restoration and a short multi-step training experiment, followed by validation-only evaluation. This next milestone has not yet been implemented or run.
