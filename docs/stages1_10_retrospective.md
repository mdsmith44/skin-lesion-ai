# Skin Lesion AI — Stages 1–10 Technical Retrospective

## Executive Summary

Skin-lesion-ai evolved from HAM10000 dataset exploration into an internal research deployment combining specialized computer vision, multimodal experimentation, constrained language generation, and NVIDIA inference tooling. The progression was HAM10000 exploration → a small CNN baseline → ResNet18 transfer learning → held-out evaluation and Grad-CAM → zero-shot VLM experiments → NeMo AutoModel with LoRA → grounded Nemotron reporting → TensorRT, Triton, and FastAPI services.

The central architectural finding was that these models earned different responsibilities. A fully fine-tuned ResNet18 remained the strongest tested classifier. Language-side adaptation of a small VLM demonstrated a working multimodal training pipeline but substantially weaker classification. Nemotron subsequently received tightly restricted classifier evidence for report formatting, while a separate SmolVLM service explored visual description. Generated descriptions never became reporting evidence.

This retrospective follows those decisions and their evidence, including unsuccessful experiments and reproducibility gaps. It describes a **research and education project, not a clinical diagnostic system**. Deployment, fast inference, and valid JSON do not establish suitability for patient care.

## 1. Problem and Experimental Discipline

Stage 1 established the dataset and evaluation constraints before model development. HAM10000 contains 10,015 dermatoscopic images representing 7,470 unique lesions. Its seven labels are `akiec` (actinic keratoses and intraepithelial carcinoma), `bcc` (basal cell carcinoma), `bkl` (benign keratosis-like lesions), `df` (dermatofibroma), `mel` (melanoma), `nv` (melanocytic nevus), and `vasc` (vascular lesions).

Class imbalance dominates the problem: `nv` accounts for 6,705 images, compared with 1,113 `mel`, 1,099 `bkl`, 514 `bcc`, 327 `akiec`, 142 `vasc`, and 115 `df`. Accuracy therefore gives substantial weight to the majority class. Balanced accuracy averages class recall; macro F1 averages class-specific F1, exposing precision/recall weaknesses that aggregate accuracy can obscure.

Multiple images can depict the same lesion. Randomly splitting individual images would allow related views into different partitions. The project instead split at lesion level with seed 42 and preserved those assignments throughout subsequent dataset transformations.

| Partition | Images | Unique lesions |
|---|---:|---:|
| Training | 7,002 | 5,229 |
| Validation | 1,532 | 1,120 |
| Test | 1,481 | 1,121 |

The [exploration notebook](../notebooks/01_dataset_exploration.ipynb) records zero lesion overlap between partitions. The available metadata has no usable patient identifier, so this establishes lesion separation, not verified patient independence.

Training fitted parameters; validation supported model and checkpoint selection; locked evaluations measured selected models on the test partition. That discipline requires a historical qualification: Stage 6 inspected test images, and an early Stage 8 probe also used a test image. Stage 8C's later lock-before-evaluation sequence is documented, but the test partition was not globally unseen throughout the project. It is already accessed and must not become a source of further tuning.

## 2. Building the Specialized Computer-Vision Baseline — Stages 1–5

The initial CNN turned dataset imbalance into an observable modeling problem. Stage 2 reached approximately 72.7% validation accuracy, but only 34.3% balanced accuracy and 36.9% macro F1. Those results motivated evaluation across classes rather than treating a rising accuracy curve as sufficient progress.

Stage 3 introduced ImageNet-initialized torchvision ResNet18 and progressively increased adaptation. Training began with a new classifier head, continued with `layer4` plus the head, and then unfroze the entire network. A separate class-weighted partial-fine-tuning branch tested whether the loss could improve minority-class performance.

The validation tradeoff was concrete. Unweighted partial fine-tuning reached approximately 78.5% accuracy, 61.1% balanced accuracy, and 59.0% macro F1. The weighted branch increased balanced accuracy to 64.7%, while accuracy fell to 72.3% and macro F1 was 59.1%. Full fine-tuning reached 79.6% accuracy, 60.1% balanced accuracy, and 60.2% macro F1. Stage 4 selected among the saved models using validation macro F1, choosing the full fine-tune before its final test evaluation.

The [Stage 4 notebook](../notebooks/04_model_evaluation.ipynb) and [saved metrics](../results/resnet18_final_test_metrics.csv) record:

| Final ResNet18 held-out metric | Result |
|---|---:|
| Accuracy | 82.85% |
| Balanced accuracy | 64.30% |
| Macro precision | 71.84% |
| Macro recall | 64.30% |
| Macro F1 | 66.58% |
| Melanoma ROC AUC | 0.905 |

The final classification rule was seven-class argmax. Melanoma AUC describes ranking across thresholds; it does not establish a selected clinical operating threshold. Classifier softmax outputs remain uncalibrated scores.

Checkpoint lineage matters to interpreting this result. The [checkpoint analysis](../resnet18_checkpoint_analysis.md) establishes that the final artifact descends from the **unweighted** partial checkpoint. Head training, partial fine-tuning, and full fine-tuning each ran for five epochs, with Adam learning rates of `1e-3`, `1e-4`, and `1e-5`, respectively, using unweighted cross-entropy. The historical comment “Reload the class-weighted model” contradicts the actual load path; the weighted branch did not supply the final model's starting weights.

The saved full model contains the fifth epoch's state, not an automatically restored best epoch. Earlier full-training epochs had higher validation accuracy, and the notebook has no best-checkpoint or early-stopping mechanism. Selecting the strongest saved model by validation macro F1 is therefore distinct from selecting the best epoch. Checkpoint validation metadata is also hard-coded and rounded. Stage 3 supplies no explicit training RNG seed or saved RNG state; the split seed cannot make training bit-for-bit reproducible. Nominally frozen phases also call `model.train()` globally, allowing BatchNorm running statistics to change even where parameters are frozen.

Stage 5 applied Grad-CAM to `model.layer4[-1]`, comparing a melanoma true positive, false negative, false positive, and correctly classified nevus. The [comparison figure](../results/gradcam_case_comparison.png) adds spatial context to error analysis. These attribution maps do not prove clinically meaningful reasoning or show that the model reasons like a dermatologist.

## 3. Testing the Vision-Language Hypothesis — Stages 6–8

Stage 6 asked whether general-purpose multimodal models could describe these images or classify them without task-specific training. It compared `HuggingFaceTB/SmolVLM-256M-Instruct` and `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` with a frozen classification prompt on 35 images, five per class. The notebook constructs this sample from the test partition; its results are a small historical probe, not a separate validation benchmark.

The [handoff](chatgpt_handoff.md) records 14.3% strict accuracy for the 256M model, 17.1% after semantic normalization, and five invalid outputs. The 500M model achieved 11.4% strict accuracy with no invalid labels, but predicted only `akiec` or `bkl`. Larger size did not improve this particular comparison. Description experiments also produced unsupported medical language despite visual-only instructions, motivating a clearer separation between fluent text and verified evidence.

Stage 7 converted image/class records into image/instruction/response records while preserving the lesion-aware partitions and one record per image. The saved artifacts used in Stage 8 contain one fixed classification instruction and abbreviation targets. This differs from an earlier handoff description of multiple instructions and full category names. The notebook still references `INSTRUCTIONS` after its definition was commented out; reproducing the saved artifacts is not equivalent to rerunning that notebook unchanged.

Training responses are available for loss calculation. During validation and test inference, only the image and instruction reach the model; stored targets are used afterward for scoring. This distinction became especially important in Stage 8's training-objective correction.

[Stage 8](stage8_nemo_vlm.md) used NVIDIA NeMo AutoModel with rank-4, alpha-8 LoRA on language-model query/value projections of SmolVLM-256M-Instruct, adapting approximately 230,400 parameters. The vision encoder and original model parameters remained frozen. Early pipeline work included a complete single-update smoke test on the original 4 GB RTX 3050 laptop GPU; later experiments used the documented Run:ai environment.

The default NeMo collator could not derive turn markers and fell back to token-pattern matching. Because the instruction already listed all possible labels, this could supervise matching class tokens inside the prompt. Suspiciously low loss was therefore not accepted as evidence of learning. The initial objective's results were excluded.

The replacement [response-only collator](../src/stage8/nemo_collator.py) verifies that the tokenized prompt is an exact prefix of the complete conversation, then masks prompt and padding labels with `-100`. A tiny-overfit diagnostic trained one training image per class for 100 steps, reaching approximately 0.0628 loss. Reloaded adapters generated all seven labels correctly without target input. That established pipeline function, not generalization.

Three corrected experiments then tested sampling. Stage 8A retained natural frequencies. An attempted Stage 8B `WeightedRandomSampler` worked in isolation but was bypassed by the NeMo recipe's own `DistributedSampler`; that intended balanced run was discarded. The correction constructed a deterministic logical dataset with 1,000 samples per class. Stage 8C instead used a 7,000-sample logical dataset with class proportions proportional to `n_c^0.5`, reducing imbalance less aggressively.

Validation used the original natural distribution, with all 1,532 images and target-free generation:

| Experiment | Accuracy | Balanced accuracy | Macro precision | Macro F1 |
|---|---:|---:|---:|---:|
| 8A: natural | 0.6880 | 0.2273 | 0.2616 | 0.2264 |
| 8B: equal-class | 0.3734 | 0.3787 | 0.2608 | 0.2492 |
| 8C: tempered | 0.6423 | 0.2880 | 0.3198 | 0.2868 |

All three produced zero invalid labels. Natural sampling favored `nv`; equal-class sampling increased balanced recall but introduced many minority-class false positives. Tempering delivered the highest macro F1 among these runs. Relative to 8A, melanoma recall rose from 0.1412 to 0.3824 while precision fell from 0.3934 to 0.3439: sampling changed the error tradeoff rather than solving it uniformly.

Stage 8C's `epoch_3_step_439` checkpoint had the lowest validation loss, 0.2747, versus 0.3070 at the final epoch. It was selected and the configuration locked before the full Stage 8C test evaluation; no further alpha values were tested. This procedure does not erase the earlier test-image exposure described above.

The locked evaluation covered all 1,481 test images with zero invalid generations:

| Selected model, held-out test | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Specialized ResNet18 | 0.8285 | 0.6430 | 0.6658 |
| Stage 8C SmolVLM + LoRA | 0.6624 | 0.2747 | 0.2686 |

Stage 8C's macro precision was 0.2645 and macro recall 0.2747. Valid output syntax clearly did not imply reliable classification across classes. The result supports retaining ResNet18 for this project's perception task. It does **not** establish a general limitation of VLMs: ResNet18 adapted its visual features directly, whereas Stage 8 only adapted language-side LoRA over a frozen general-purpose vision encoder.

## 4. Separating Perception from Language — Stage 9

Stage 9 changed the role of language generation:

```text
image → specialized ResNet18 → structured trusted evidence
      → pinned Nemotron → strict JSON/report validator
```

“Trusted” identifies the evidence's controlled origin and integrity, not clinical truth. The evidence object records classifier scores, image and checkpoint hashes, class mapping, preprocessing, and runtime provenance. An allowlist projects only the image identifier, predicted class, and displayed top score into the language prompt. Nemotron receives no image, reference diagnosis, patient history, Grad-CAM, or Stage 7 instruction. It cannot supply visual findings or explain the classifier's reasoning from those inputs.

The pinned model is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, revision `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`. It runs in BF16 on one NVIDIA GB300 using native Transformers, cached files, and disabled remote model code. Generation is greedy and deterministic in configuration: no sampling, one beam, thinking disabled, and at most 256 new tokens. No model is trained in this stage.

The frozen `stage9-grounded-v1` contract has exactly six fields: `image_id`, `predicted_class`, `top_softmax_score`, `summary`, `unknowns`, and `limitation`. The score is an exact six-decimal string; missing predictions have fixed unknown behavior. Summary wording is prescribed, unavailable visual/clinical information remains explicit, and the limitation states research-only use and uncalibrated scores.

The [validator](../src/stage9/reporting.py) rejects malformed JSON, duplicate keys, extra or missing fields, changed facts, and unsupported wording. Failures remain visible with null reports and retained raw output; there is no repair, prompt retry, or silent template substitution.

The [development record](stage9_grounded_reporting.md) reports eight passing synthetic contract tests and nine fixed generation cases: one training image, seven validation images selected by reference class, and one synthetic missing-prediction case. All nine generated reports were accepted and exactly matched the deterministic template. Reference labels selected cases but never entered reporting evidence. The runner restricts dataset access to train/validation; the underlying path-based classifier alone does not enforce split membership.

For this fixed structured transformation, the template accomplished the same output without an LLM. Nemotron demonstrated integration and constrained formatting, with no demonstrated added information or classification benefit. That is an engineering finding about this experiment's contract, not a conclusion about broader language tasks.

## 5. Production-Style NVIDIA Deployment — Stage 10

Stage 10 translated these responsibilities into independently deployed inference services behind a CPU FastAPI gateway:

| Route | Service responsibility |
|---|---|
| `/classify` | Gateway preprocessing → Triton HTTP V2 → TensorRT FP16 ResNet18; application returns class and seven uncalibrated scores |
| `/describe` | Dedicated SmolVLM service generates visual text and applies a conservative description validator |
| `/report` | Nemotron service classifies the uploaded image through Triton, constructs restricted evidence, generates, and validates the report |

The `/report` trust boundary begins with image bytes. Uploads accept exactly one `image` field; extra fields and duplicate files are rejected. Callers cannot supply class, score, prompt, diagnosis, split, or reporting evidence. The Nemotron service performs classification internally. Its [adapter](../src/stage10/triton_report_adapter.py) checks score order, finiteness, normalization, argmax consistency, expected classifier provenance, and the uploaded-image hash before constructing an opaque server-generated identifier and split-free prompt evidence. **SmolVLM text is never supplied to Nemotron.**

The description service uses base SmolVLM2-500M-Video-Instruct, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, without Stage 8 LoRA. Its frozen prompt and lexical/sentence checks constrain output but cannot establish visual accuracy. Reporting reuses the pinned Stage 9 model and unchanged six-field validator, returning raw generation, validation status, and the deterministic baseline separately. A validator rejection is a completed inference with structured output, not a silently substituted success.

The classifier conversion path is PyTorch checkpoint → ONNX → TensorRT. The tracked FP32 builder implements a same-input comparison against PyTorch; the FP16 builder records three-runtime logit differences and argmax agreement and checks for FP16 execution layers. These are narrow verification procedures, not a new held-out accuracy study. The Triton note separately records exact agreement of all seven logits between live Triton and direct FP16 inference on one normalized validation tensor.

The deployed engine uses FP16 tactics with FP32 input/output and TF32 disabled. Triton's `resnet18_fp16` model repository contains `config.pbtxt` and `1/model.plan`, with minimum/optimum/maximum batch sizes of 1/8/32. Stage 4 RGB conversion, direct 224 × 224 resize, and ImageNet normalization remain outside Triton, as do softmax and label conversion.

Separate gateway, SmolVLM, and Nemotron Dockerfiles and GitHub Actions workflows build ARM64 images for GHCR with commit-SHA tags. Run:ai/Kubernetes services provide internal routing. Persistent storage holds model artifacts and the Hugging Face cache; language services use cached models offline. The deployment record reports one GB300 for each language service, while the CPU gateway needs no persistent volume.

The documented one-image live smoke established service connectivity: classification returned `bkl`, description was rejected for exceeding the sentence limit, and reporting matched the template. The exact deployed image digests, sanitized workload manifests, timestamp, and complete final raw response were not retained in reviewed repository artifacts. The [deployment record](stage10_deployment.md) explicitly distinguishes these reported observations from an older saved smoke result.

## 6. Performance Engineering

The repository separates model execution from transport and orchestration. Its local benchmark compares PyTorch FP32, TensorRT FP32, and TensorRT FP16 using preloaded validation tensors and CUDA events, with 100 warm-ups and 300 measured iterations per runtime/batch. The authoritative saved result reports `status: passed` and records:

| Runtime | Batch | Median GPU execution per batch | Recorded derived throughput |
|---|---:|---:|---:|
| PyTorch FP32 | 1 | 1.922 ms | ≈519 images/s |
| TensorRT FP32 | 1 | 1.107 ms | ≈903 images/s |
| TensorRT FP16 | 1 | 0.156 ms | ≈6,394 images/s |
| TensorRT FP16 | 8 | 0.193 ms | ≈41,344 images/s |
| TensorRT FP16 | 32 | 0.370 ms | ≈86,386 images/s |

Throughput is derived from mean batch time in the benchmark implementation, not from the rounded medians shown here. Input preparation is excluded. At batch 1, the verified median times make TensorRT FP16 approximately 12.3× faster than PyTorch FP32 for this local GPU-execution measurement. That ratio does not include preprocessing or transport.

A separate single-client Triton HTTP benchmark measured client-observed batch-1 round trips: approximately 6.14 ms median after 100 warm-ups across 300 requests. Concurrent HTTP tests reached approximately 898 requests/s without dynamic batching and 952 requests/s with its active configuration at 32 clients, with 640 measured requests per corresponding concurrency level.

These two runs do not isolate a causal batching improvement. Preferred batch sizes of 8, 16, and 32 with a 1,000-microsecond queue delay are configuration, not proof of executed batch sizes; no executed-batch histogram was established by those results.

Neither local GPU timings nor Triton HTTP tests measure end-to-end gateway/model-service latency. They exclude the full description/reporting paths. Cold model loading is another distinct scope. Cross-runtime peak-memory comparisons were also avoided because allocator counters were not comparable. The performance contribution is a reproducible measurement structure with explicit boundaries, alongside the recorded component results.

## 7. Engineering Evolution

Early notebooks made data distributions, errors, and modeling choices inspectable. Stage 8 moved recurring logic into reusable dataset, collator, model, and evaluation modules under `src/stage8/`, backed by explicit YAML experiment configurations. That transition made errors in objectives and sampling behavior easier to isolate.

Stage 9 introduced reusable checkpoint loading, evidence construction, provenance, and strict output contracts. Synthetic tests in `tests/test_stage9_reporting.py` cover malformed outputs and evidence restrictions without GPU or dataset dependencies. Stage 10 added optimized artifacts, HTTP clients, container boundaries, and mocked gateway/service tests, including `tests/test_stage10_gateway_report.py`.

The result is an evolution from exploratory execution toward independently deployable services with explicit inputs, outputs, and failure behavior. Model training, model conversion, contract validation, and deployment smoke checks remain different forms of evidence rather than interchangeable definitions of correctness.

## 8. Key Lessons

- **Protect the split and document exposure.** Lesion-aware partitioning prevented repeated-lesion leakage; preserving its history also required acknowledging earlier test-image inspection.
- **Read metrics together.** Both the CNN and Stage 8A showed how majority-class success can coexist with weak balanced accuracy and macro F1.
- **Verify the objective and actual data path.** Response-only masking and the overridden sampler mattered more than a reassuring loss curve or intended configuration.
- **Treat sampling as an error tradeoff.** Equal and tempered sampling changed minority recall and false positives; neither dominated every metric.
- **Assign models evidence-backed roles.** ResNet18 supplied the strongest tested perception, while language models explored separate description and formatting responsibilities.
- **Compare against deterministic code.** The nine-case template match made the LLM's lack of demonstrated benefit visible for this fixed reporting contract.
- **Preserve measurement and provenance boundaries.** Hashes, pinned revisions, preprocessing, runtime environments, and precisely scoped timings make results interpretable without claiming exact training reproducibility.

## 9. Limitations

HAM10000 performance is a benchmark result, not clinical diagnosis. Lesion separation does not verify patient independence; severe imbalance and small rare-class supports constrain interpretation. No calibration claim is supported, and Grad-CAM does not establish clinically meaningful reasoning.

Stage 8 explored a small VLM, frozen visual features, language-side LoRA, and three sampling strategies. It did not evaluate the wider space of visual adaptation or multimodal architectures. Selected models have already been evaluated on test data, with historical exposure predating the Stage 8C lock; that partition cannot support fresh iterative development claims.

The original ResNet18 training lacks explicit RNG reproducibility. Deployment artifacts improve provenance but do not reconstruct those earlier runs. Serving conversion checks are narrow, and a one-image smoke is not a broad service-quality evaluation. Contract acceptance establishes permitted structure and copied evidence, not clinical or factual correctness: a faithful report can reproduce an incorrect classifier prediction.

## 10. Technologies and Skills Demonstrated

The repository evidences Python, PyTorch/torchvision, scikit-learn evaluation, Hugging Face Transformers, NVIDIA NeMo AutoModel, parameter-efficient fine-tuning with native NeMo LoRA, and NVIDIA Nemotron integration. Deployment work adds ONNX export, TensorRT optimization, NVIDIA Triton Inference Server, FastAPI, Docker, Git/GitHub, GitHub Actions, GHCR, Run:ai/Kubernetes, and NVIDIA GB300 execution.

The demonstrated skills span experimental design, leakage-aware datasets, imbalance analysis, transfer learning, multimodal objective debugging, generative evaluation, provenance checking, strict contracts, service integration, and scoped benchmarking. PEFT here denotes the adaptation method; the successful Stage 8 workflow used native NeMo LoRA.

## 11. Reproducibility and Supporting Documentation

Detailed evidence and reproduction procedures remain in:

- [Stage 8 NeMo VLM](stage8_nemo_vlm.md): corrected objective, sampling experiments, selected checkpoint, and evaluation artifacts.
- [Stage 9 grounded reporting](stage9_grounded_reporting.md): frozen contract, model pin, synthetic tests, and fixed development results.
- [Stage 10 deployment](stage10_deployment.md): service architecture, reported smoke, runtime provenance, and measurement limits.
- [Stage 10 Triton ResNet18](stage10_triton_resnet18.md): model repository, tensor contract, engine identity, and direct serving verification.
- [ResNet18 checkpoint analysis](../resnet18_checkpoint_analysis.md): training lineage, saved-state semantics, preprocessing, and reproducibility caveats.

Two useful provenance anchors are the source ResNet18 checkpoint SHA-256 `fd08a89ff6e459e3321b3b1ba5cdd0d3650289e28f4b78739d96b775dc6df73f` and deployed FP16 engine SHA-256 `7370bba2e48f106824f7f9a4d4ab6d065a5cf2b512311de6a47c95a5ac844cb3`. The model revisions above identify the reporting and description snapshots. Stage 8C's selected adapter is `lora-b64-tempered/epoch_3_step_439` in persistent outputs.

Data, checkpoints, engines, caches, and large outputs live outside Git. The publication workflows produce immutable full-commit-SHA container tags and also apply mutable `latest` tags under documented rules. Reproduction still requires the external artifacts and documented environments because the reviewed record does not retain every exact deployed image digest and runtime artifact needed to reconstruct each historical deployment state. This retrospective preserves those limits rather than treating implementation, historical observations, and reproducible artifacts as equivalent evidence.
