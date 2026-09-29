# Skin Lesion AI

Skin Lesion AI is a completed Stage 1–10 research project that follows a domain-specific machine-learning system from dataset design through an internal, multi-model NVIDIA deployment. It began with HAM10000 exploration and a specialized ResNet18 classifier, tested zero-shot and adapted vision-language models, and then assigned perception, description, and reporting to separate models based on the experimental evidence.

The later lifecycle ran in an NVIDIA GB300 and Run:ai environment: NeMo AutoModel and PEFT/LoRA experimentation, pinned Nemotron inference, ONNX and TensorRT optimization, Triton serving, persistent model storage, and containerized FastAPI services deployed through Run:ai/Kubernetes.

> **Research and education only.** This repository is not a clinical diagnostic system and must not be used for medical decision-making or patient care.

For the decisions, results, caveats, and project history, start with the **[Stages 1–10 Technical Retrospective](docs/stages1_10_retrospective.md)**.

## Architecture at a Glance

The final system exposes three image routes through a CPU FastAPI gateway. Each route has a distinct responsibility:

```text
                                      ┌─ /classify
                                      │  ResNet18 → ONNX → TensorRT FP16
                                      │  → Triton HTTP V2 → class + 7 scores
uploaded image → CPU FastAPI gateway ─┼─ /describe
                                      │  pinned SmolVLM2 → description validator
                                      └─ /report
                                         ResNet18/Triton
                                         → trusted structured evidence
                                         → pinned Nemotron
                                         → strict Stage 9 JSON validator
```

| Route | Model responsibility | Contract |
|---|---|---|
| `POST /classify` | Specialized image classification | Returns the predicted class/index and seven **uncalibrated softmax scores** |
| `POST /describe` | Visual-language description | Preserves raw SmolVLM2 text and returns an accepted description or explicit rejection |
| `POST /report` | Constrained evidence-to-JSON formatting | Internally classifies the uploaded bytes, constructs allowlisted evidence, then accepts or rejects Nemotron output |

The `/report` caller supplies only image bytes. Caller-provided classes, scores, prompts, diagnoses, split metadata, and report fields are not trusted. The service verifies classifier provenance, score order, argmax consistency, and the uploaded-image hash before constructing report evidence.

**SmolVLM descriptions are never supplied to Nemotron as evidence.** Nemotron receives only a server-generated image identifier, the ResNet18 predicted class, and the displayed top uncalibrated score. It does not receive the image, reference diagnosis, patient history, Grad-CAM, or visual findings.

The deployed model identities are pinned:

- Description: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Reporting: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, revision `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`.

See the [Stage 10 deployment record](docs/stage10_deployment.md) for the route, validation, failure, and service contracts.

## NVIDIA GB300 Platform

The GB300 environment supported the transition from model experiments to an operational research deployment. It was part of the later project lifecycle, not a claim that every earlier notebook or workload required this platform. Early CV work ran in a conventional PyTorch environment, and the first complete Stage 8 LoRA update was verified on an RTX 3050 laptop GPU with 4 GB VRAM.

On the documented Run:ai workspace, one NVIDIA GB300 and persistent project storage provided a common environment for later experimentation and artifact management:

| Lifecycle area | Documented use of the GB300 environment |
|---|---|
| Multimodal adaptation | Later NeMo AutoModel experiments used native LoRA on SmolVLM language attention while the general-purpose vision encoder remained frozen |
| Grounded generation | The pinned 30B Nemotron model ran in BF16 on one GB300 from a persistent, local-only Hugging Face cache |
| Inference optimization | The ResNet18 checkpoint was exported to ONNX, built as TensorRT FP32 and FP16 engines, numerically checked, and benchmarked on an NVIDIA GB300 |
| Model serving | Triton served the FP16 TensorRT engine through its HTTP V2 API; preprocessing and score interpretation remained in the application |
| Orchestration | Run:ai/Kubernetes provided internal service routing; live SmolVLM and Nemotron services were each documented using one GB300 and mounted persistent storage |
| Containers and delivery | Separate CPU gateway, SmolVLM, and Nemotron Dockerfiles were published through GitHub Actions to GHCR with full-commit-SHA tags and mutable `latest` rules |

The two language-service images build from `nvcr.io/nvidia/nemo-automodel:26.06.00` and use offline model caches. The CPU gateway has no GPU or persistent-volume requirement. Runtime service URLs are injected by the deployment rather than baked into the images.

The tracked repository does not retain every exact deployed image digest or a sanitized copy of every historical Run:ai workload specification. The deployment document separates source provenance, observed live state, and retained artifacts accordingly.

## Key Results

### Model and reporting results

| Experiment | Accuracy | Balanced accuracy | Macro F1 | Interpretation |
|---|---:|---:|---:|---|
| Selected fully fine-tuned ResNet18, 1,481-image test split | 82.85% | 64.30% | 66.58% | Strongest tested classifier; macro precision 71.84%, melanoma AUC 0.905 |
| Selected Stage 8C SmolVLM + language-side LoRA, same test split | 66.24% | 27.47% | 26.86% | Zero invalid labels, but weak performance across minority classes |
| Stage 9 grounded reporting, nine fixed development cases | — | — | — | 9/9 accepted Nemotron reports exactly matched the deterministic template |

The comparison does not establish a general limitation of VLMs. Stage 8 adapted approximately 230,400 language-side LoRA parameters while leaving the general-purpose vision encoder frozen; ResNet18 adapted its visual features directly to HAM10000.

The nine Stage 9 cases comprised one training image, seven validation images, and one synthetic missing-prediction case. For this fixed six-field transformation, the deterministic template produced the same accepted output. The 30B model therefore showed no demonstrated benefit for that narrow mapping; it is retained as an explicit constrained-generation experiment, not a necessary component for fixed templating.

### Stage 10 ResNet18 execution benchmark

The authoritative benchmark used preloaded validation tensors, CUDA events, 100 warm-ups, and 300 measured iterations for each runtime/batch combination on an NVIDIA GB300.

| Runtime | Batch | Median GPU execution per batch | Derived throughput |
|---|---:|---:|---:|
| PyTorch FP32 | 1 | 1.922 ms | ≈519 images/s |
| TensorRT FP32 | 1 | 1.107 ms | ≈903 images/s |
| TensorRT FP16 | 1 | 0.156 ms | ≈6,394 images/s |
| TensorRT FP16 | 8 | 0.193 ms | ≈41,344 images/s |
| TensorRT FP16 | 32 | 0.370 ms | ≈86,386 images/s |

Throughput is derived from mean batch time, not the rounded medians displayed above. At batch 1, the median TensorRT FP16 execution was approximately 12.3× faster than PyTorch FP32 in this local measurement.

These are **model-execution measurements with preprocessing excluded**. They are not HTTP or end-to-end API latency. A separate single-client Triton HTTP test measured approximately 6.14 ms median batch-1 round trips. Neither result measures the complete `/describe` or `/report` path.

The active Triton configuration requests preferred dynamic batches of 8, 16, and 32 with a 1,000 μs queue delay. Retained concurrency measurements do not show which batch sizes Triton actually executed, and the two concurrency runs do not establish that dynamic batching caused their throughput difference.

## Dataset and Evaluation Discipline

HAM10000 contains 10,015 dermatoscopic images, 7,470 unique lesions, and seven classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, and `vasc`. It is severely imbalanced: `nv` accounts for 6,705 images, about 67% of the dataset.

Because a lesion can have multiple images, the project avoided random image-level splitting. A fixed-seed lesion-level split was created and carried through the multimodal dataset:

| Partition | Images | Unique lesions |
|---|---:|---:|
| Training | 7,002 | 5,229 |
| Validation | 1,532 | 1,120 |
| Test | 1,481 | 1,121 |

No lesion IDs overlap across partitions. The metadata used here has no usable patient identifier, so **patient-level independence is not established**. Balanced accuracy and macro F1 accompany accuracy because majority-class performance can otherwise dominate the result.

Model selection used validation metrics before each final locked evaluation. The historical record still matters: Stage 6 used a small balanced sample from the test partition, and an early Stage 8 probe inspected one test image. Stage 8C was selected and committed before its full test evaluation, but the test set was not globally pristine throughout the entire project and must not support further tuning.

## What We Learned

- **Different models earned different responsibilities.** ResNet18 remained the strongest tested classifier; SmolVLM2 was isolated as a description service; Nemotron was limited to a closed evidence-to-JSON contract.
- **Accuracy alone was misleading.** The baseline CNN reached 72.7% validation accuracy but only 34.3% balanced accuracy and 36.9% macro F1.
- **Training-objective details changed the experiment.** A response-only collator was required after a fallback label-matching path could supervise class names already present in the instruction.
- **Sampling changed precision/recall tradeoffs.** Natural, equal-class, and tempered Stage 8 sampling produced different majority/minority behavior; no strategy dominated every validation metric.
- **Evidence boundaries reduced unsupported generation paths.** Reporting inputs are allowlisted, unsupported content is rejected, and rejected output is retained rather than silently repaired.
- **Deterministic code was the better baseline for a fixed transformation.** All nine accepted Stage 9 outputs matched the template exactly.
- **Performance claims require measurement boundaries.** GPU execution, Triton HTTP round trips, model load time, and full application latency are distinct.

## Stage 1–10 Evolution

| Stage | Decision, experiment, or engineering result |
|---:|---|
| 1 | Explored HAM10000 imbalance and created a lesion-aware split with overlap checks |
| 2 | Trained a small CNN baseline; balanced metrics exposed majority-class effects |
| 3 | Compared frozen, partial, weighted-partial, and full ResNet18 fine-tuning |
| 4 | Selected the saved full-fine-tune model by validation macro F1 and ran the final test evaluation |
| 5 | Added Grad-CAM for model inspection, without treating attribution as clinical reasoning |
| 6 | Tested two small zero-shot SmolVLM variants for description and 35-image classification |
| 7 | Built image/instruction/response records while retaining the original lesion-aware split |
| 8 | Implemented NeMo AutoModel LoRA, corrected response masking, verified tiny overfit, and compared natural, equal, and tempered sampling |
| 9 | Separated ResNet18 perception from pinned Nemotron formatting through trusted evidence and a strict six-field validator |
| 10 | Exported ResNet18 through ONNX/TensorRT, served it with Triton, containerized three services, and deployed the internal system with Run:ai/Kubernetes |

## Technology Stack

| Area | Technologies |
|---|---|
| Data and modeling | Python, pandas, Pillow, PyTorch, torchvision, scikit-learn |
| Multimodal adaptation | Hugging Face Transformers, NVIDIA NeMo AutoModel, PEFT/LoRA |
| Language generation | SmolVLM/SmolVLM2, NVIDIA Nemotron |
| Optimization and serving | ONNX, TensorRT 10.16.1.11, NVIDIA Triton Inference Server, FastAPI |
| Platform and delivery | NVIDIA GB300, CUDA 13.2, Docker, GHCR, GitHub Actions, Run:ai/Kubernetes |
| Engineering controls | Model/artifact hashes, pinned revisions, strict schemas, mocked service tests, synthetic contract tests |

## Repository Guide

```text
notebooks/                 Stages 1–8 exploration, training, and evaluation
configs/stage8/            NeMo/LoRA smoke, diagnostic, and full-run configs
src/stage8/                Dataset, response-only collator, training, inference
src/stage9/                ResNet18 loading, evidence contract, Nemotron runner
src/stage10/               Export/build/benchmark scripts and inference services
tests/                     Synthetic contract and mocked service tests
docs/                      Technical records and deployment details
Dockerfile.stage10-*       CPU gateway, SmolVLM, and Nemotron service images
.github/workflows/         ARM64 container build and GHCR publication workflows
```

Detailed records:

- **[Stages 1–10 Technical Retrospective](docs/stages1_10_retrospective.md)** — decisions, results, lessons, limitations, and engineering evolution.
- [Stage 8: NeMo VLM Adaptation](docs/stage8_nemo_vlm.md) — objective correction, LoRA, sampling experiments, and locked test evaluation.
- [Stage 9: Grounded Nemotron Reporting](docs/stage9_grounded_reporting.md) — evidence boundary, six-field contract, model pin, and development results.
- [Stage 10: Deployed Research Services](docs/stage10_deployment.md) — service architecture, live smoke scope, containers, Run:ai wiring, and benchmarks.
- [Stage 10: Triton ResNet18](docs/stage10_triton_resnet18.md) — model repository, tensor contract, batching configuration, and serving verification.
- [ResNet18 Checkpoint Analysis](resnet18_checkpoint_analysis.md) — training lineage, saved-state semantics, preprocessing, and reproducibility caveats.

## Reproduction Guide

Raw HAM10000 images, trained checkpoints, TensorRT plans, model caches, and large run outputs are excluded from Git. Place HAM10000 under `data/raw/ham10000/`; the tracked split is `data/processed/metadata_splits.csv`.

For Stages 1–7, create the Python 3.11 environment and run the numbered notebooks in order:

```bash
conda env create -f environment.yml
conda activate skin-lesion-ai
jupyter notebook
```

`environment-lock.yml` preserves the detailed earlier environment. The Stage 3 training code does not set an explicit training RNG seed, so the fixed split does not make training bit-for-bit reproducible. The saved final ResNet18 is the fifth full-fine-tuning epoch, not an automatically restored best epoch; see the checkpoint analysis before reproducing or interpreting that lineage.

Stage 8 uses the tracked modules and YAML configurations under `src/stage8/` and `configs/stage8/`. The selected configuration is `configs/stage8/smolvlm_nemo_train_tempered.yaml`; its persistent selected checkpoint was `lora-b64-tempered/epoch_3_step_439`.

Stage 9 and Stage 10 synthetic/mocked tests require neither live model inference nor HAM10000:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  tests.test_stage9_reporting \
  tests.test_stage10_api \
  tests.test_stage10_gateway_describe \
  tests.test_stage10_gateway_report \
  tests.test_stage10_description_contract \
  tests.test_stage10_smolvlm_api \
  tests.test_stage10_nemotron_api -v
```

Tracked Stage 10 scripts cover checkpoint-to-ONNX export, FP32 and FP16 engine builds, direct Triton verification, local runtime benchmarking, and sequential and concurrent HTTP benchmarks. Reproducing live services also requires the documented persistent artifacts and pinned local model snapshots.

Provenance anchors:

- ResNet18 checkpoint SHA-256: `fd08a89ff6e459e3321b3b1ba5cdd0d3650289e28f4b78739d96b775dc6df73f`
- TensorRT FP16 engine SHA-256: `7370bba2e48f106824f7f9a4d4ab6d065a5cf2b512311de6a47c95a5ac844cb3`

## Interpretation Limits

HAM10000 classification is a research benchmark, not clinical validation. Softmax values are uncalibrated model scores. Grad-CAM indicates image regions that influenced a prediction but does not establish clinically meaningful reasoning. Description or report validator acceptance establishes compliance with narrow structural and lexical contracts, not factual accuracy or safety. A perfectly grounded report can faithfully repeat an incorrect classifier prediction.

The Stage 8 comparison covers one small VLM, a frozen vision encoder, language-side LoRA, and a limited sampling search. It does not answer how vision-side adaptation, other VLMs, or full multimodal fine-tuning would perform. The Stage 10 one-image smoke and component benchmarks establish integration and scoped execution behavior, not clinical generalization or end-to-end service performance.
