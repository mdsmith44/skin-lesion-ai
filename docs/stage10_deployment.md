# Stage 10 — deployed research inference services

Stage 10 is operational as an internal Run:ai deployment. This document records
the implemented contracts and the reported one-image live smoke test. The
deployment is for research and education only, **not clinical diagnosis or
patient care**. [Stage 9](stage9_grounded_reporting.md) records the earlier
grounded-report development experiment; [Stage 8](stage8_nemo_vlm.md) records
the separate VLM adaptation experiments.

## Architecture and evidence boundary

The CPU FastAPI gateway has three image-inference routes, plus `GET /health`:

| Gateway route | Internal path | Result |
|---|---|---|
| `POST /classify` | Gateway → Triton HTTP V2 → TensorRT FP16 ResNet18 | Predicted class/index and seven uncalibrated softmax scores |
| `POST /describe` | Gateway → SmolVLM2 500M service → visual-description validator | Raw generated text and an accepted description or an explicit rejection |
| `POST /report` | Gateway → Nemotron service → Triton/ResNet18 → trusted split-free evidence → pinned Nemotron → unchanged Stage 9 validator | Raw generation, accepted six-field report or rejection, and a separate deterministic baseline |

Each POST accepts exactly one multipart file field named `image` (maximum 10
MiB). Extra fields and duplicate image files are rejected. `/report` forwards
only the original image bytes: callers cannot supply a class, score, split,
diagnosis, prompt, report field, or other evidence. The gateway does not
reinterpret the internal report. SmolVLM output is deliberately **never** used
as Nemotron evidence. There is no combined inference route.

The gateway's `GET /health` checks dependency readiness without inference or
text generation. It reports `application`, `triton_server`, `resnet18_fp16`,
`smolvlm2_service`, and `nemotron_report_service` separately. All five must be
`ready` for HTTP 200; otherwise it returns 503. Unreadable images and malformed
uploads return 4xx, dependency failures return sanitized 5xx, and malformed
successful upstream responses return 502. A model's **validator rejection is
still a completed inference**: it is returned as structured HTTP 200 data,
not replaced by another model result.

The route implementations are [the gateway](../src/stage10/api.py),
[Triton classifier](../src/stage10/triton_classifier.py),
[SmolVLM service](../src/stage10/smolvlm_api.py), and
[Nemotron service](../src/stage10/nemotron_api.py). Their mocked tests are in
`tests/test_stage10_api.py`, `tests/test_stage10_gateway_describe.py`,
`tests/test_stage10_gateway_report.py`, `tests/test_stage10_smolvlm_api.py`,
and `tests/test_stage10_nemotron_api.py`.

## Verified live smoke and its scope

The reported end-to-end smoke used the validation image
`ISIC_0027419.jpg`, whose uploaded bytes had SHA-256
`b1476cf07c4b4040aeb0146b16d747f1aaaf7fff12418b735b3a6c7ed9ffb37f`.
Its reference diagnosis was not supplied to the services or used for this
check. The gateway returned HTTP 200 from `/health` with all five fields
`ready`.

- `/classify` returned class `bkl`, index `2`, and top uncalibrated softmax
  `0.6147692203521729`.
- `/describe` completed generation, but the frozen validator returned
  `validation_status: rejected`, `validated_description: null`, and
  `more_than_two_sentences` in `rejection_reasons`. The raw generated text is
  retained in the response. This is a contract rejection, not a service
  failure; no claim about visual correctness follows from it.
- `/report` classified the image as `bkl` again. Nemotron received only an
  opaque server-generated `img_` identifier, the trusted predicted class, and
  the trusted top score formatted as a six-decimal string. No train/validation
  split, lesion ID, SmolVLM text, diagnosis, or caller-supplied evidence entered
  the prompt. The strict Stage 9 validator accepted the generated report. Its
  six fields exactly matched the **separately returned** deterministic
  `template_report()` baseline for this one case.

The baseline match shows no observable benefit from the 30B model for this
fixed transformation in this smoke case. It does not establish the same result
for other inputs or a broader reporting task. These final five-dependency
smoke observations were supplied from the live deployment; the older saved
FastAPI smoke JSON under persistent storage records an earlier three-field
health state and must not be mistaken for this final check. No timestamp,
complete raw response, or deployed image digest is asserted here because those
were not retained in the reviewed repository artifacts.

## Classifier and Triton provenance

The output order is exactly `akiec, bcc, bkl, df, mel, nv, vasc`.
The gateway applies the Stage 4 evaluation transform: PIL image conversion to
RGB, direct resize to 224 × 224, `ToTensor()` FP32 conversion, then ImageNet
normalization with mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`. There is no center crop or evaluation augmentation.
The same normalized `[1,3,224,224]` FP32 tensor contract is used by Triton.
Softmax, argmax, and label conversion occur in the application, outside the
engine. Scores are **uncalibrated model scores, not clinical probabilities**.

Triton serves `resnet18_fp16`, version `1`, from
`/workspace/storage/skin-lesion-ai/triton/model_repository/resnet18_fp16/`.
Its `tensorrt_plan` input is `normalized_rgb` FP32 `[N,3,224,224]`; output
`logits` is FP32 `[N,7]`. The TensorRT 10.16.1.11 plan uses FP16 tactics with
FP32 I/O, TF32 disabled, and one optimization profile with minimum/optimum/
maximum batches **1/8/32**. The source ONNX SHA-256 is
`d1ec2128bca0cc466e544d29d150378c25a1744e90bc8e196ef8534d93418447`;
the copied engine/`1/model.plan` SHA-256 is
`7370bba2e48f106824f7f9a4d4ab6d065a5cf2b512311de6a47c95a5ac844cb3`.
The FP16 build metadata at
`/workspace/storage/skin-lesion-ai/outputs/stage10/resnet18/fp16_20260928T020556.315794Z/metadata.json`
records the full build command and engine size (22,725,012 bytes). This is a
persistent artifact, not a file in Git.

The active persistent `config.pbtxt` has `max_batch_size: 32` and:

```protobuf
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 1000
}
```

These are **configured preferences**, not evidence that executions of sizes
8, 16, or 32 occurred. The persistent dynamic-batching concurrency result at
`/workspace/storage/skin-lesion-ai/outputs/stage10/triton/concurrent_http_b1_dynamic_20260928T061454.185868Z.json`
records the model-config API response and active config SHA-256
`aec83caeba580abb81e04db135dac2bbb429eb447e47e7234e314a7ce9c0bd22`;
it does not by itself establish an executed-batch histogram. The archived
no-batching config remains under persistent `outputs/stage10/triton/configs/`.
The [Triton repository note](stage10_triton_resnet18.md) gives the compact
model-repository contract.

## Description and report contracts

`/describe` uses base
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at pinned revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`, without Stage 8 LoRA. The
image and [frozen visual-only prompt](../src/stage10/description_contract.py)
are the only model inputs. Generation sets `do_sample=False`, `num_beams=1`,
and `max_new_tokens=96`. The response preserves `raw_generated_text` even
when `validated_description` is null. The conservative validator checks
sentence count and explicit lexical patterns for class/diagnostic terms,
treatment, clinical interpretation, health-status claims, and nonvisual
patient claims. `accepted` means only that these limited checks passed; it
does not mean factual accuracy, clinical safety, or freedom from hallucination.

`/report` uses `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, pinned and
resolved to revision `bf77c3174f68ad409e1c2aa60daeb46e32d1c606` in the
reported deployment. It loads BF16 on one NVIDIA GB300 from the persistent
local cache with `local_files_only=True` and remote model code disabled. The
observed cold model load was **63.514064090006286 seconds**; that is not
request latency. Generation is deterministic (`do_sample=False`, one beam,
at most 256 new tokens, thinking disabled) and serialized per service process.

The [Stage 10 adapter](../src/stage10/triton_report_adapter.py) checks the
seven finite ordered softmax scores, argmax/class consistency, expected Triton
model/version/engine hash, and the uploaded image hash. It then constructs a
split-free evidence object and passes it through unchanged Stage 9
`prompt_evidence()`. Only `observed_metadata.image_id` and
`classifier.{predicted_class,top_softmax_score}` reach Nemotron. The frozen
`stage9-grounded-v1` system prompt requires exactly six fields:
`image_id`, `predicted_class`, `top_softmax_score`, `summary`, `unknowns`, and
`limitation`. [Stage 9 validation](../src/stage9/reporting.py) rejects
malformed, changed, extra, or unsupported content; it does not repair a
rejection or silently substitute the deterministic template. `/report` returns
`classifier_evidence`, exact `prompt_evidence`, a distinct `nemotron` object
(`raw_generated_text`, `validated_report` or null, status/reason, provenance),
and `deterministic_template_baseline` separately.

## Deployment wiring and reproducibility

| Internal service | DNS name in `runai-nccl` | Runtime configuration |
|---|---|---|
| Triton | `stage10-resnet18-triton-dynamic.runai-nccl.svc.cluster.local` | HTTP V2 on port 8000; mounted model repository |
| SmolVLM | `stage10-smolvlm.runai-nccl.svc.cluster.local` | HTTP port 8000; cached pinned model |
| Nemotron | `stage10-nemotron.runai-nccl.svc.cluster.local` | HTTP port 8000; `TRITON_HTTP_URL` and cached pinned model |
| CPU gateway | `stage10-skin-lesion-api-v2.runai-nccl.svc.cluster.local` | HTTP port 8000; required `TRITON_HTTP_URL`, `SMOLVLM_HTTP_URL`, `NEMOTRON_HTTP_URL` |

The gateway sets `TRITON_HTTP_URL`, `SMOLVLM_HTTP_URL`, and
`NEMOTRON_HTTP_URL` to `http://` plus the corresponding internal service DNS
name above (HTTP port 8000). The Nemotron service separately sets its own
`TRITON_HTTP_URL` to the Triton service. These URLs are deployment runtime
configuration, not values baked into the containers.

The gateway image is CPU-only and needs no PVC. The two language-model images
use `nvcr.io/nvidia/nemo-automodel:26.06.00`; their Dockerfiles set
`HF_HOME=/workspace/storage/skin-lesion-ai/huggingface`, `HF_HUB_OFFLINE=1`,
and `TRANSFORMERS_OFFLINE=1`. The Nemotron deployment uses one GB300 and PVC
`nccl-20g-project-w5g6t` mounted at `/workspace/storage`. The live SmolVLM
Run:ai deployment was also verified during the deployment session as using one
NVIDIA GB300 and the same PVC mounted at `/workspace/storage`, with
`HF_HOME=/workspace/storage/skin-lesion-ai/huggingface`. These are observed
live deployment facts supplied from that session; the exact Run:ai workload
manifest is not currently preserved in tracked repository artifacts.

The [CPU gateway](../Dockerfile.stage10-api),
[SmolVLM](../Dockerfile.stage10-smolvlm), and
[Nemotron](../Dockerfile.stage10-nemotron) Dockerfiles have separate GitHub
Actions workflows that build native ARM64 images in these GHCR repositories:

- `ghcr.io/mdsmith44/skin-lesion-ai-stage10-api`
- `ghcr.io/mdsmith44/skin-lesion-ai-stage10-smolvlm`
- `ghcr.io/mdsmith44/skin-lesion-ai-stage10-nemotron`

The workflows publish full-commit-SHA tags and have `latest` rules; `latest`
is mutable. Relevant source commits are
`e084bc52b31d04164cf94a9dce460d15b1b49f36` (grounded service),
`15104b5251c1ac2e125a15f30bcbfbc5c028b175` (Nemotron container), and
`383eb5cbbaf1681a577e1254e77626f0e0d77076` (gateway `/report`). These
are source provenance, **not verified deployed image tags or digests**; the
exact deployed digests and sanitized Run:ai workload/Service specifications
remain to be recorded.

Tracked reproduction scripts cover [ONNX export](../src/stage10/export_resnet18_onnx.py),
[FP32](../src/stage10/build_resnet18_fp32.py) and
[FP16](../src/stage10/build_resnet18_fp16.py) builds,
[direct Triton verification](../src/stage10/verify_triton_resnet18_fp16.py),
[local runtime](../src/stage10/benchmark_resnet18.py),
[sequential HTTP](../src/stage10/benchmark_triton_http_b1.py) and
[concurrent HTTP](../src/stage10/benchmark_triton_http_concurrency.py)
benchmarks, and the [256M base/LoRA](../src/stage10/compare_smolvlm_descriptions.py)
plus [500M extension](../src/stage10/extend_smolvlm2_500m_descriptions.py)
description comparisons. `src/stage10/nemotron_report.py` is an **earlier
direct-TensorRT, train/val-bound experiment**, not the deployed `/report`
implementation. Reproducing these experiments requires their separate
documented development environment and persistent artifacts. Model binaries,
checkpoints, the TensorRT plan, HAM10000 data, and Hugging Face cache do not
belong in Git.

The focused gateway and contract tests use synthetic inputs and mocked
dependencies, with no live model inference:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  tests.test_stage9_reporting \
  tests.test_stage10_api tests.test_stage10_gateway_describe \
  tests.test_stage10_gateway_report tests.test_stage10_description_contract \
  tests.test_stage10_smolvlm_api tests.test_stage10_nemotron_api -v
```

## Measured performance and limits

The saved local runtime benchmark at
`/workspace/storage/skin-lesion-ai/outputs/stage10/resnet18/benchmark_20260928T022123.204260Z/results.json`
preloaded validation tensors and measured GPU execution with CUDA events,
100 warm-ups and 300 measured iterations per runtime/batch. For TensorRT FP16:

| Batch | Median GPU execution per batch | Derived throughput |
|---:|---:|---:|
| 1 | 0.156 ms | about 6,394 images/s |
| 8 | 0.193 ms | about 41,344 images/s |
| 32 | 0.370 ms | about 86,386 images/s |

The separate single-client Triton HTTP test at
`/workspace/storage/skin-lesion-ai/outputs/stage10/triton/single_client_http_b1_20260928T053136.671673Z.json`
measured **client-observed round trips** for batch-1 requests after 100
warm-ups and across 300 requests; its median was about **6.14 ms**. In the
separate batch-1 HTTP concurrency tests, 32 clients reached about **898
requests/s** without dynamic batching and **952 requests/s** with the active
dynamic-batching configuration, with 640 measured requests in each 32-client
level. These are two runs, not evidence that batching caused the difference.
GPU-event time, Triton HTTP round-trip time, and model cold-load time have
different scopes. **None is an end-to-end gateway latency or a Nemotron/SmolVLM
serving benchmark.** The local GPU-memory counters were not comparable across
PyTorch and TensorRT allocators, so no cross-runtime peak-memory claim is made.

HAM10000 classification metrics, a single deployment image, uncalibrated
softmax values, VLM text, strict-contract acceptance, and throughput results
do not establish clinical validity, calibration, generalization, factual
visual accuracy, or suitability for patient care. A classifier can be wrong
while Nemotron perfectly reproduces its prediction. No reference diagnosis
was used in the Stage 10 smoke or reporting prompt.
