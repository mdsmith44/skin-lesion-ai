# Stage 9 — Grounded Nemotron reporting

Stage 9 connects the existing specialized ResNet18 classifier to text-only
Nemotron reporting. No model is trained or adapted in this stage. No HAM10000
test data is used. Stage 8C remains locked and unchanged.

The scope is a reproducible, locally executed reporting prototype, not a
clinical system or a deployment service. Stage 10 serving is separate work.

## Architecture and evidence boundary

```text
train/validation image
    → existing ResNet18 checkpoint
    → structured evidence with seven uncalibrated softmax scores
    → allowlisted text evidence
    → pinned Nemotron, reasoning off
    → strict JSON and grounding validator
    → accepted report OR explicit rejection
```

ResNet18 performs image classification. Nemotron sees no image, reference
diagnosis, Stage 7 instruction, patient history, or Grad-CAM. It cannot derive
visual findings or explain why the classifier made its prediction.

The evidence schema (`schema_version: 1.0`) contains:

| Field | Source |
|---|---|
| `observed_metadata` | Image/lesion identifiers and existing train/val membership |
| `classifier.predicted_class` | Seven-class ResNet18 argmax |
| `classifier.softmax_scores` | All seven full precision model scores |
| `classifier.top_softmax_score` | Winning score rendered to six decimal places |
| `unknowns` | Visual findings, clinical diagnosis, patient history |
| `input_provenance` | Resolved image path and SHA-256 |
| `model_provenance` | Checkpoint path/hash, mapping, preprocessing, runtime versions |

Only the image identifier, predicted class and displayed top score are sent to
Nemotron. Reference labels are used only to select one validation example per
class for the small development check; they are not reporting evidence.

## Closed report contract

`src/stage9/reporting.py` defines the frozen `stage9-grounded-v1` prompt,
evidence builder, deterministic template baseline and validator. The report has
exactly six fields:

- `image_id`: copied identifier.
- `predicted_class`: copied classifier output or `unknown` when absent.
- `top_softmax_score`: exact six-decimal string, or `unknown` when absent.
- `summary`: constrained sentence identifying this as a ResNet18 prediction
  with an uncalibrated softmax score; a fixed unknown sentence when absent.
- `unknowns`: the three unavailable clinical/visual fields.
- `limitation`: research-only and uncalibrated-score statement.

The validator rejects malformed JSON, duplicate keys, extra/missing fields,
changed scores or labels, and any summary beyond the prescribed wording. It
does not repair responses, retry prompts or silently replace failures with the
template. Rejected responses remain in audit artifacts with `report: null`.

This narrow contract makes grounding mechanically verifiable. It does **not**
demonstrate general hallucination detection or safe arbitrary medical prose.
The deterministic template generates the same required report without an LLM;
it is the appropriate baseline and cheaper choice for this fixed schema.
Nemotron's demonstrated role here is constrained text formatting and integration,
not improved classification, clinical reasoning, or added factual knowledge.

## Models and environment

ResNet18 checkpoint:

```text
/workspace/storage/skin-lesion-ai/models/resnet18_full_finetuned.pt
SHA-256: fd08a89ff6e459e3321b3b1ba5cdd0d3650289e28f4b78739d96b775dc6df73f
```

The loader checks `architecture` and exact `class_to_idx` before strictly
loading the complete state dictionary. Mapping: `akiec=0, bcc=1, bkl=2, df=3,
mel=4, nv=5, vasc=6`. Stage 4 preprocessing is preserved: PIL RGB, direct
224×224 resize, ToTensor, ImageNet normalization, no augmentation. Checkpoint
weights and BatchNorm state are unchanged.

Nemotron:

```text
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
revision: bf77c3174f68ad409e1c2aa60daeb46e32d1c606
cache: /workspace/storage/skin-lesion-ai/huggingface/hub
```

Native Transformers `NemotronHForCausalLM`, BF16 on one NVIDIA GB300, with
`trust_remote_code=False`. The reporting loader uses cached files only.
Generation is greedy, one beam, reasoning disabled, at most 256 new tokens.
Prompts exceeding 2,048 tokens are rejected, not truncated.

Verified container: Python 3.12.3, Transformers 5.8.1, NVIDIA PyTorch
2.12.0a0+0291f960b6.nv26.04.48445190, CUDA 13.2. No packages were installed or
upgraded. Initial standalone compatibility testing returned the requested JSON
using 60.1 GiB peak allocated GPU memory; that first generation's 28.6 seconds
included first-use effects and is not a throughput benchmark.

The initial model download was performed separately. On a fresh workspace,
the pinned snapshot must be provisioned in the cache before using this runner.
The runner deliberately does not download missing models automatically.

## Reproduction

Run from `/workspace/skin-lesion-ai` in the existing NeMo container.

Unit tests use synthetic data only and need neither GPU nor checkpoints:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_stage9*.py' -v
```

One training example through the complete pipeline:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python -m src.stage9.run_reports --split train
```

Seven validation examples (first saved row per reference class) and a synthetic
missing-prediction case:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python -m src.stage9.run_reports --split val --per-class --missing-case
```

This is a fixed development check, not a new split, classification benchmark,
or estimate of patient-level generalization. The CLI accepts only train/val.
The raw ResNet inference API accepts a path and does not infer split membership;
dataset isolation is enforced by the development runner.

Each invocation creates a fresh timestamped directory under:

```text
/workspace/storage/skin-lesion-ai/outputs/stage9/reports/
```

Artifacts include a manifest with selected identifiers, source/CSV hashes, one
JSON per case with full evidence, actual prompts, raw generation, validation
status, accepted report, template baseline, model provenance, timing and GPU
memory, plus an aggregate summary. Outputs are outside Git by default.

## Validation scope

Completed on 2026-09-27 using one frozen prompt and no retries or prompt tuning:

| Check | Result |
|---|---|
| Synthetic contract/rejection unit tests | 8 passed |
| First training image, full pipeline | 1/1 accepted |
| First validation image per reference class | 7/7 accepted |
| Synthetic absent classifier prediction | 1/1 accepted, prediction/score unknown |
| Exact match to deterministic report baseline | 9/9 generated reports |

Training run: `20260927T064410.169644Z`.
Validation plus missing-data run: `20260927T064602.861401Z`.
Both are under the persistent reports directory above.

| Validation image | ResNet18 prediction | Displayed softmax score |
|---|---|---:|
| ISIC_0027419 | bkl | 0.617010 |
| ISIC_0029967 | bcc | 0.973890 |
| ISIC_0031023 | mel | 0.788148 |
| ISIC_0029404 | bkl | 0.711729 |
| ISIC_0034093 | bcc | 0.944172 |
| ISIC_0032115 | nv | 0.999951 |
| ISIC_0029915 | df | 0.401396 |

These are model predictions, not reference diagnoses. Sampling one example per
reference class does not ensure one prediction per class. No classification
accuracy estimate is claimed from this development sample.

The first training report took 39.0 seconds; the first validation report took
17.6 seconds. Subsequent real-image validation reports took 5.39–5.60 seconds;
the synthetic missing case took 4.62 seconds. These single-process observations
include first-use effects and are not a controlled serving benchmark.

Example accepted report:

```json
{
  "image_id": "ISIC_0026769",
  "predicted_class": "bkl",
  "top_softmax_score": "0.692280",
  "summary": "ResNet18 predicted bkl with an uncalibrated softmax score of 0.692280.",
  "unknowns": ["visual_findings", "clinical_diagnosis", "patient_history"],
  "limitation": "Research and education only; not clinical diagnosis. Scores are uncalibrated."
}
```

Unit tests check exact facts, unknown handling, exclusion of reference labels
and arbitrary instruction fields, invalid score rejection, malformed/duplicate
JSON, extra/missing fields, unsupported medical wording, and test-split rejection
before file access. Identifier and displayed-score fields reject arbitrary text
before prompt construction. The runner exits with a nonzero status if any report
is rejected, after preserving the rejection artifacts. Accepted outputs must exactly match the deterministic
baseline for every required field. A model can fail to follow this contract;
that failure must remain visible as a rejection.

Clinical validity, calibration, visual observations, patient independence and
general report-writing reliability are not established by Stage 9. The classifier
can be wrong even when its report is perfectly faithful. No generated report
should be used for diagnosis, treatment or patient care.

## Completion and next boundary

Stage 9's constrained reporting prototype is complete: reusable classifier
inference, evidence construction, pinned text generation, strict rejection,
orchestration, a deterministic baseline, small development checks and persistent
audit artifacts. No change to model weights, Stage 8 code/configuration, saved
dataset partitions or dependencies was required.

Future work can evaluate richer reporting requirements or Stage 10 serving.
Neither arbitrary medical prose nor a clinical inference service is authorized
by these results. For the current fixed report, a deterministic template is
sufficient; keep Nemotron as an explicitly identified experimental formatter.
