# Skin Lesion AI

An educational computer-vision and multimodal-AI project using the
HAM10000 dermatoscopic skin-lesion dataset.

The project explores the progression from conventional image
classification to transfer learning, model evaluation, explainability,
vision-language models (VLMs), multimodal instruction tuning, and
eventually NVIDIA NeMo and Nemotron tooling.

> **Research and educational use only.**
> This project is not intended for clinical diagnosis, medical
> decision-making, or patient care.

---

## Project Question

How do specialized computer-vision models and general-purpose
vision-language models compare on a domain-specific medical imaging
task, and can multimodal domain adaptation improve VLM performance?

The project follows this progression:

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

---

## Dataset

The project uses the HAM10000 ("Human Against Machine with 10000
training images") dermatoscopic image dataset.

The dataset contains 10,015 images representing seven lesion
categories:

| Label | Category |
|---|---|
| `akiec` | Actinic keratosis / intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesion |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevus |
| `vasc` | Vascular lesion |

The dataset is highly imbalanced, with melanocytic nevi (`nv`)
representing approximately 67% of the images.

Because multiple images can correspond to the same lesion, the
train/validation/test split is performed at the **lesion level** to
prevent images of the same lesion from appearing in multiple splits.

HAM10000 does not provide a usable patient identifier in the metadata
used here, so lesion-level separation does not guarantee patient-level
independence.

### Split

| Split | Images |
|---|---:|
| Train | 7,002 |
| Validation | 1,532 |
| Test | 1,481 |

The raw dataset is intentionally excluded from this repository.

---

## Stage 1 — Dataset Exploration

Notebook:

`notebooks/01_dataset_exploration.ipynb`

Work includes:

- metadata exploration
- class-distribution analysis
- image inspection
- lesion/image relationship analysis
- lesion-aware stratified train/validation/test splitting
- leakage checks

The resulting split metadata is stored under `data/processed/`.

---

## Stage 2 — Baseline CNN

Notebook:

`notebooks/02_baseline_cnn.ipynb`

A small convolutional neural network establishes an initial baseline.

Approximate validation results:

| Metric | Result |
|---|---:|
| Accuracy | 72.7% |
| Balanced accuracy | 34.3% |
| Macro F1 | 36.9% |

The large gap between accuracy and balanced accuracy demonstrates the
effect of HAM10000's class imbalance.

---

## Stage 3 — Transfer Learning

Notebook:

`notebooks/03_transfer_learning.ipynb`

A pretrained ResNet-18 is adapted to HAM10000.

Experiments include:

1. frozen feature extractor
2. partial fine-tuning
3. class-weighted partial fine-tuning
4. full fine-tuning

The experiments demonstrate the tradeoff between overall accuracy and
minority-class performance.

Model checkpoints are excluded from normal Git history.

---

## Stage 4 — Model Evaluation

Notebook:

`notebooks/04_model_evaluation.ipynb`

The final model was selected using validation performance before
evaluating the held-out test set.

Final ResNet-18 test results:

| Metric | Result |
|---|---:|
| Accuracy | 82.8% |
| Balanced accuracy | 64.3% |
| Macro precision | 71.8% |
| Macro recall | 64.3% |
| Macro F1 | 66.6% |
| Melanoma AUC | 0.905 |

The evaluation emphasizes that accuracy alone is insufficient for an
imbalanced medical-imaging dataset.

---

## Stage 5 — Explainability

Notebook:

`notebooks/05_gradcam_explainability.ipynb`

Grad-CAM is used to visualize image regions associated with ResNet-18
predictions.

Cases examined include:

- melanoma true positive
- melanoma false negative
- melanoma false positive
- correctly classified melanocytic nevus

Example visualizations are stored in `results/`.

Grad-CAM is treated as a model-inspection technique rather than evidence
of clinical reasoning.

### Grad-CAM Example

The figure below compares Grad-CAM visualizations for representative
classification outcomes, including a melanoma true positive, melanoma
false negative, melanoma false positive, and correctly classified
melanocytic nevus.

![Grad-CAM case comparison](results/gradcam_case_comparison.png)

The visualizations help inspect which image regions influenced the
model's predictions. They should not be interpreted as evidence that
the model is identifying clinically meaningful structures.

Grad-CAM is used here as a model-inspection and explainability technique,
not as evidence of clinical reasoning or diagnostic validity.

---

## Stage 6 — Vision-Language Models

Notebook:

`notebooks/06_vision_language_model.ipynb`

Zero-shot experiments were performed with small SmolVLM models.

The experiments explored two capabilities:

- natural-language image description
- zero-shot lesion classification

A fixed balanced sample of 35 test images (5 per class) was used for a
small controlled classification experiment.

### Zero-Shot Results

| Metric | SmolVLM 256M | SmolVLM2 500M |
|---|---:|---:|
| Images | 35 | 35 |
| Strict accuracy | 14.3% | 11.4% |
| Invalid-format outputs | 5 | 0 |
| Average inference time | 0.62 s/image | 0.68 s/image |

The 500M model produced valid output formatting for all 35 examples but
showed severe class bias: 27 of 35 images were classified as `akiec`
and the remaining 8 as `bkl`.

The experiment illustrates an important distinction:

**General multimodal capability does not imply domain-specific
classification expertise.**

The VLM also generated plausible-sounding but unsupported medical
language during image-description experiments, reinforcing the need to
separate fluent language generation from grounded or clinically valid
interpretation.

Example VLM artifacts are stored in `results/`.

---

## Stage 7 — Multimodal Dataset Construction

Notebook:

`notebooks/07_multimodal_dataset.ipynb`

HAM10000 is transformed from a conventional classification dataset:

    (image, class)

into an instruction-tuning representation:

    (image, instruction, response)

For example:

    USER:
    [dermatoscopic image]

    What lesion category is shown in this dermatoscopic image?

    ASSISTANT:
    melanoma

Multiple equivalent instruction formulations are assigned
reproducibly while retaining one record per image.

The original lesion-aware train/validation/test assignments are
preserved.

Processed multimodal files are stored under:

`data/processed/multimodal/`

---

## Next — NVIDIA NeMo

The next stage will investigate parameter-efficient adaptation of a VLM
using the NVIDIA ecosystem.

Planned topics include:

- NVIDIA NeMo / NeMo AutoModel
- parameter-efficient fine-tuning (PEFT)
- Low-Rank Adaptation (LoRA)
- multimodal model adaptation
- comparison with the zero-shot VLM baseline
- GPU-memory and training-efficiency considerations

The project will then explore where Nemotron models can contribute to
structured explanation or reporting and, where appropriate, NVIDIA
deployment tooling such as NIM or Triton.

---

## Project Structure

    skin-lesion-ai/
    ├── data/
    │   ├── raw/                 # HAM10000 (not tracked by Git)
    │   └── processed/           # Splits and multimodal metadata
    ├── models/                  # Local model checkpoints (not tracked)
    ├── notebooks/
    │   ├── 01_dataset_exploration.ipynb
    │   ├── 02_baseline_cnn.ipynb
    │   ├── 03_transfer_learning.ipynb
    │   ├── 04_model_evaluation.ipynb
    │   ├── 05_gradcam_explainability.ipynb
    │   ├── 06_vision_language_model.ipynb
    │   └── 07_multimodal_dataset.ipynb
    ├── results/                 # Figures, metrics, and example outputs
    ├── src/                     # Reusable source code
    ├── environment.yml
    ├── environment-lock.yml
    └── README.md

---

## Environment

The project was developed using Python 3.11 with PyTorch and CUDA.

Create the Conda environment with:

    conda env create -f environment.yml
    conda activate skin-lesion-ai

`environment.yml` contains the primary project dependencies.

`environment-lock.yml` captures the more detailed environment used
during development and is retained for reproducibility and debugging.

GPU acceleration is recommended but is not required for dataset
exploration.

---

## Data Setup

HAM10000 is not included in this repository because of its size.

After obtaining the dataset, place the raw files under:

    data/raw/ham10000/

The expected structure includes the HAM10000 metadata CSV and image
directories.

The notebooks document the subsequent preprocessing and split
construction.

---

## Collaboration

Development is organized so new experiments can be implemented on
feature branches and reviewed before merging into `main`.

The upcoming NVIDIA work is intended to provide a clean transition from
the current PyTorch/Transformers baseline into NeMo-based multimodal
adaptation.

---

## Current Status

Stages 1–7 establish the pre-NeMo baseline:

- HAM10000 exploration and leakage-aware splitting
- CNN baseline
- ResNet-18 transfer learning
- held-out model evaluation
- Grad-CAM explainability
- zero-shot SmolVLM experiments
- multimodal instruction-dataset construction

The next major milestone is **Stage 8: NVIDIA NeMo and
parameter-efficient VLM adaptation**.