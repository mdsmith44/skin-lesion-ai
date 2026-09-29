# Stage 8: VLM Adaptation with NVIDIA NeMo AutoModel and LoRA

## 1. Goal

Stage 8 investigates whether a general-purpose vision-language model (VLM) can be
adapted to classify skin-lesion images from the HAM10000 dataset.

The experiment uses:

- **Base model:** `HuggingFaceTB/SmolVLM-256M-Instruct`
- **Training framework:** NVIDIA NeMo AutoModel
- **Adaptation method:** Parameter-Efficient Fine-Tuning (PEFT) using LoRA
- **Task:** Seven-class HAM10000 lesion classification
- **Output:** One of the seven HAM10000 diagnostic abbreviations:
  `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, or `vasc`

The purpose of this stage is not to develop a clinical diagnostic system.
Instead, it explores how a general-purpose multimodal model behaves when adapted
to a specialized and highly imbalanced medical-image classification task.

The existing lesion-level train/validation/test split from earlier project stages
is retained so that images belonging to the same lesion do not cross dataset
splits. The held-out test set remains isolated until model development and
checkpoint selection are complete.


## 2. Model and Adaptation Strategy

### 2.1 Why a VLM?

Earlier stages used a specialized ResNet18 image classifier. Stage 8 introduces
a different architecture: a vision-language model that processes an image and a
text instruction and generates a textual response.

Conceptually:

    skin-lesion image
            +
      text instruction
            |
            v
        SmolVLM
            |
            v
    generated class label

Each training example contains an image, an instruction defining the seven valid
HAM10000 labels, and the correct class abbreviation as the assistant response.

During evaluation, the correct response is never supplied to the model. The
model receives only the image and instruction and must generate the class label.


### 2.2 LoRA Adaptation

Rather than fully fine-tuning SmolVLM, Stage 8 uses Low-Rank Adaptation (LoRA),
a parameter-efficient fine-tuning technique.

The pretrained model weights remain frozen while small trainable low-rank
adapters are inserted into selected model layers. The effective adapted weight
can be viewed conceptually as:

    W_adapted = W_base + Delta_W

where `W_base` remains frozen and `Delta_W` is represented by the much smaller
LoRA parameters.

For this experiment, LoRA is applied to the query and value projections in the
language model's attention layers:

    *.text_model.*.q_proj
    *.text_model.*.v_proj

The LoRA configuration uses:

- Rank: 4
- Alpha: 8
- Trainable LoRA parameters: approximately 230,400

The vision encoder remains frozen throughout Stage 8. Therefore, this experiment
does not teach the SmolVLM vision encoder new dermatology-specific visual
features. Instead, it tests how effectively a small language-side adaptation can
use the visual representations already produced by the pretrained model.

This distinction is important when interpreting the final results. The earlier
ResNet18 experiments allowed visual feature extraction to adapt directly to
HAM10000, whereas the Stage 8 VLM relies on a frozen general-purpose vision
encoder.


## 3. Correcting the Training Objective

### 3.1 Initial NeMo Collator Behavior

An important issue was discovered during the initial NeMo AutoModel training
experiments.

The default NeMo VLM collator attempted to determine which tokens belonged to
the assistant response so that only those tokens contributed to the
language-model training loss. NeMo emitted the warning:

    could not derive turn markers ... Falling back to BPE pattern-match labels

This fallback behavior was problematic for this dataset because the user
instruction itself contains the seven valid HAM10000 class abbreviations.

For example, the prompt contains labels such as:

    akiec, bcc, bkl, df, mel, nv, vasc

while the assistant response might simply be:

    mel

A token-pattern matching approach could therefore identify occurrences of the
target class inside the instruction rather than exclusively identifying the
assistant's answer.


### 3.2 Why This Matters

For supervised VLM adaptation, the model should learn:

    image + instruction -> assistant response

The prompt provides context but should not itself be treated as the prediction
target.

If prompt tokens containing the class names are incorrectly included in the
training labels, a very low training or validation loss can be misleading. The
model may be rewarded for predicting tokens that already appear in the
instruction instead of learning the intended image-to-class mapping.

Consequently, the results from the original training objective were treated as
invalid and were not used as Stage 8 results.


### 3.3 Custom Response-Only Collator

A custom NeMo-compatible collator was implemented in
`src/stage8/nemo_collator.py`.

For each example, it constructs both:

1. The prompt containing the image and user instruction, with the generation
   prompt appended.
2. The complete training conversation containing the prompt plus the assistant
   response.

The collator verifies that the tokenized prompt is an exact prefix of the full
conversation. It then masks the entire prompt portion of the labels using
`-100`, which causes those tokens to be ignored by the language-model
cross-entropy loss.

Conceptually:

    Tokens:
    [ image/prompt tokens ][ assistant answer tokens ]

    Labels:
    [ -100 -100 ... -100 ][ mel ... ]

                             ^^^^^^^
                         loss computed here

Padding tokens are also masked.

This guarantees that the optimization objective is based on predicting the
assistant response rather than reproducing class-label tokens appearing in the
instruction.


### 3.4 Verification

The corrected collator was tested across all seven HAM10000 classes.

For a melanoma example, the prompt contained 162 tokens and the complete
conversation contained 165 tokens. The prompt was verified as an exact prefix,
and the remaining tokens corresponded to the assistant response:

    mel<end_of_utterance>

A corrected NeMo smoke test subsequently produced a nontrivial training loss
rather than the suspicious near-zero behavior from the original objective.

This correction became the foundation for all subsequent Stage 8 experiments.


## 4. Tiny-Overfit Verification

Before performing another full training run, the corrected training pipeline was
tested using a deliberately tiny dataset.

### 4.1 Purpose

The goal of this experiment was not to measure generalization. Instead, it was
a diagnostic test of the training machinery.

A sufficiently expressive model should be able to memorize a very small training
set. Failure to do so would suggest that the training objective, gradient flow,
LoRA configuration, data pipeline, or inference procedure was still incorrect.

One training image was selected from each of the seven HAM10000 classes:

- `akiec`
- `bcc`
- `bkl`
- `df`
- `mel`
- `nv`
- `vasc`

This produced a seven-example training dataset.


### 4.2 Training

The same corrected response-only collator and language-side LoRA configuration
used for the later Stage 8 experiments were applied to the seven-example
dataset.

The tiny-overfit run was trained for 100 optimization steps.

Training loss decreased to approximately:

    0.0628

This demonstrated that the corrected optimization objective could successfully
fit the small training set.


### 4.3 Free-Generation Test

Memorizing the training loss alone was not considered sufficient evidence that
the complete pipeline worked.

The final checkpoint was therefore reloaded and evaluated using the same
target-free generation procedure intended for real evaluation:

    image + instruction -> generated response

The correct assistant response was not supplied during generation.

The model generated the correct class abbreviation for all seven training
examples:

    7 / 7 correct

This test verified the complete path:

    image
      |
      v
    processor
      |
      v
    SmolVLM + LoRA
      |
      v
    free generation
      |
      v
    HAM10000 class label


### 4.4 Interpretation

The tiny-overfit result does not demonstrate generalization and is not reported
as a model-performance result.

Instead, it serves as a pipeline validation test. It provides evidence that:

- the corrected labels represent the assistant response,
- gradients reach the intended LoRA parameters,
- the LoRA parameters can learn the task,
- saved adapters can be reloaded correctly, and
- free-generation inference can recover the learned labels.

After this diagnostic passed, full Stage 8 training resumed using the original
lesion-aware training and validation splits.

## 5. Stage 8A: Natural Class Distribution

After validating the corrected training pipeline, the first full experiment
used the natural HAM10000 training distribution.

The training set is highly imbalanced. The largest class, `nv`, contains 4,679
training images, while several minority classes contain fewer than 250.

Training used:

- Natural HAM10000 training distribution
- Batch size: 64
- Maximum optimization steps: 550
- Five epochs
- Learning rate: `1e-4`
- AdamW optimizer
- Rank-4 LoRA on language-model `q_proj` and `v_proj`
- Frozen vision encoder
- Corrected response-only collator

Validation loss improved throughout training, reaching its lowest value at the
final checkpoint:

    epoch_4_step_549
    validation loss = 0.2392


### 5.1 Free-Generation Validation Results

The selected checkpoint was evaluated on all 1,532 validation images using
target-free generation.

Results:

| Metric | Stage 8A |
|---|---:|
| Accuracy | 0.6880 |
| Balanced accuracy | 0.2273 |
| Macro precision | 0.2616 |
| Macro recall | 0.2273 |
| Macro F1 | 0.2264 |
| Invalid generations | 0 |

Prediction counts revealed strong majority-class behavior:

| Predicted class | Count |
|---|---:|
| nv | 1235 |
| bkl | 194 |
| mel | 61 |
| bcc | 23 |
| akiec | 19 |
| df | 0 |
| vasc | 0 |

Although overall accuracy was relatively high, the model predicted `nv` for
approximately 81% of validation images.

This illustrates why accuracy alone is insufficient for this dataset. Balanced
accuracy and macro F1 reveal that performance across the seven classes was much
weaker than the overall accuracy suggests.


## 6. Stage 8B: Equal-Class Sampling

Stage 8B investigated whether aggressively balancing the training distribution
would reduce the majority-class behavior observed in Stage 8A.

### 6.1 NeMo Sampling Constraint

An initial attempt used PyTorch's `WeightedRandomSampler`. The sampler behaved
correctly in an isolated PyTorch test, but inspection of the NeMo AutoModel
training recipe showed that the recipe constructs its own
`DistributedSampler` and passes it directly to the DataLoader.

As a result, the configured weighted sampler was not used during NeMo training.

The first intended "balanced" training run was therefore effectively identical
to Stage 8A and was discarded as an invalid comparison.


### 6.2 Balanced Logical Dataset

Rather than modifying NVIDIA's training code, balancing was implemented at the
dataset level.

A deterministic logical training dataset was constructed containing 1,000
examples from each class:

| Class | Natural training images | Logical samples |
|---|---:|---:|
| akiec | 230 | 1000 |
| bcc | 366 | 1000 |
| bkl | 774 | 1000 |
| df | 76 | 1000 |
| mel | 778 | 1000 |
| nv | 4679 | 1000 |
| vasc | 99 | 1000 |

Total logical training samples:

    7000

Minority classes were repeated as evenly as possible, while `nv` was
downsampled without replacement.

NeMo's normal `DistributedSampler` could then shuffle the logical dataset
without overriding the desired class distribution.


### 6.3 Validation Results

The lowest validation loss occurred at:

    epoch_3_step_439
    validation loss = 0.4068

Free-generation validation results were:

| Metric | Stage 8B |
|---|---:|
| Accuracy | 0.3734 |
| Balanced accuracy | 0.3787 |
| Macro precision | 0.2608 |
| Macro recall | 0.3787 |
| Macro F1 | 0.2492 |
| Invalid generations | 0 |

Equal-class sampling substantially changed model behavior. The model no longer
collapsed primarily to `nv`, and balanced accuracy increased from 0.2273 to
0.3787.

However, the model now produced many minority-class false positives. Overall
accuracy fell from 0.6880 to 0.3734.

The experiment demonstrated that aggressive balancing corrected one problem
while introducing another.


## 7. Stage 8C: Tempered Class Sampling

Stage 8C was designed as a principled compromise between the natural
distribution of Stage 8A and the equal-class distribution of Stage 8B.

Rather than making every class equally likely, the logical class distribution
was defined using:

    p_c proportional to n_c^alpha

where:

- `n_c` is the natural number of training examples for class `c`
- `alpha = 0.5`

At the extremes:

    alpha = 1.0  -> natural class proportions
    alpha = 0.0  -> equal class proportions

Therefore, `alpha = 0.5` retains information about the natural distribution
while reducing its extreme imbalance.


### 7.1 Tempered Training Distribution

The 7,000-example logical training dataset contained:

| Class | Natural training images | Tempered logical samples |
|---|---:|---:|
| akiec | 230 | 599 |
| bcc | 366 | 756 |
| bkl | 774 | 1100 |
| df | 76 | 345 |
| mel | 778 | 1103 |
| nv | 4679 | 2704 |
| vasc | 99 | 393 |
| **Total** | **7002** | **7000** |

The logical dataset contained 5,027 unique training images, compared with 3,323
unique images in the equal-class Stage 8B dataset.

All other major training settings were held constant so that the sampling
distribution was the primary experimental change.


### 7.2 Checkpoint Selection

Validation loss was:

| Checkpoint | Validation loss |
|---|---:|
| epoch_0_step_109 | 0.3516 |
| epoch_1_step_219 | 0.3000 |
| epoch_2_step_329 | 0.2974 |
| epoch_3_step_439 | **0.2747** |
| epoch_4_step_549 | 0.3070 |

The final epoch was worse than the preceding checkpoint. Therefore, the
validation-selected Stage 8C checkpoint was:

    epoch_3_step_439

This checkpoint was selected before accessing the held-out test set.


### 7.3 Validation Results

Free-generation evaluation on the natural validation split produced:

| Metric | Stage 8C |
|---|---:|
| Accuracy | 0.6423 |
| Balanced accuracy | 0.2880 |
| Macro precision | 0.3198 |
| Macro recall | 0.2880 |
| Macro F1 | **0.2868** |
| Invalid generations | 0 |

Stage 8C produced the highest macro F1 of the three sampling strategies while
retaining substantially more overall accuracy than equal-class sampling.

For example, melanoma performance changed substantially relative to Stage 8A:

| Melanoma metric | Stage 8A | Stage 8C |
|---|---:|---:|
| Precision | 0.3934 | 0.3439 |
| Recall | 0.1412 | 0.3824 |
| F1 | 0.2078 | 0.3621 |

The increase in melanoma recall came with some loss of precision, illustrating
the tradeoff introduced by stronger minority-class representation.


### 7.4 Comparison of Sampling Strategies

| Metric | 8A Natural | 8B Equal | 8C Tempered |
|---|---:|---:|---:|
| Accuracy | **0.6880** | 0.3734 | 0.6423 |
| Balanced accuracy | 0.2273 | **0.3787** | 0.2880 |
| Macro precision | 0.2616 | 0.2608 | **0.3198** |
| Macro recall | 0.2273 | **0.3787** | 0.2880 |
| Macro F1 | 0.2264 | 0.2492 | **0.2868** |
| Invalid generations | 0 | 0 | 0 |

No single sampling strategy was best on every metric.

Natural sampling favored overall accuracy but strongly favored the majority
class. Equal-class sampling produced the highest balanced recall but generated
many minority-class false positives. Tempered sampling provided a middle ground
and produced the highest validation macro F1.

Stage 8C was therefore locked as the final Stage 8 model configuration.

No additional values of `alpha` were tested after this selection. This avoided
continuing to tune the sampling distribution against the validation set after a
clear experimental comparison had been obtained.

## 8. Locked Held-Out Test Evaluation

After Stage 8C was selected using validation results, the model configuration
and checkpoint were locked before the held-out test split was accessed.

The locked checkpoint was:

    epoch_3_step_439

The Git history also preserves this sequence:

    8d147ce  Add tempered sampling experiment for Stage 8 VLM
    cf87bb4  Support locked Stage 8 test evaluation

The tempered-sampling experiment was therefore committed before the code was
modified to permit Stage 8 evaluation on the test split.

No model selection, hyperparameter tuning, or sampling changes were made using
test-set results.


### 8.1 Test Procedure

The final model was evaluated once on all 1,481 images in the held-out test
split.

Evaluation used free generation:

    image + instruction -> generated class abbreviation

The target response was not supplied to the model.

All 1,481 generations produced valid HAM10000 class labels.

Results:

| Metric | Validation | Held-out test |
|---|---:|---:|
| Accuracy | 0.6423 | **0.6624** |
| Balanced accuracy | 0.2880 | **0.2747** |
| Macro precision | 0.3198 | **0.2645** |
| Macro recall | 0.2880 | **0.2747** |
| Macro F1 | 0.2868 | **0.2686** |
| Invalid generations | 0 | **0** |

Overall accuracy increased slightly on the test split, while the
imbalance-sensitive metrics decreased modestly. Macro F1 changed from 0.2868 on
validation to 0.2686 on the held-out test set.


### 8.2 Test Performance by Class

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| bcc | 0.1915 | 0.2535 | 0.2182 | 71 |
| vasc | 0.0000 | 0.0000 | 0.0000 | 19 |
| mel | 0.3542 | 0.3091 | 0.3301 | 165 |
| akiec | 0.1304 | 0.1304 | 0.1304 | 46 |
| nv | 0.8386 | 0.8488 | 0.8437 | 992 |
| df | 0.0000 | 0.0000 | 0.0000 | 20 |
| bkl | 0.3368 | 0.3810 | 0.3575 | 168 |

The model remained strongest on the majority `nv` class.

It demonstrated some discrimination for `bkl`, `mel`, and `bcc`, but performance
on the rarest classes remained poor. No `df` examples were correctly
classified, and none of the three `vasc` predictions were correct.

The prediction distribution was:

| Predicted class | Count |
|---|---:|
| nv | 1004 |
| bkl | 190 |
| mel | 144 |
| bcc | 94 |
| akiec | 46 |
| vasc | 3 |
| df | 0 |

The difference between 66.24% overall accuracy and 27.47% balanced accuracy is
particularly important. Overall accuracy is heavily influenced by the large
number of `nv` examples and substantially overstates performance across the
seven diagnostic classes.


## 9. Comparison with the Specialized ResNet18

Earlier project stages trained a ResNet18 specifically for HAM10000 image
classification.

The selected ResNet18 model was fully fine-tuned on the visual classification
task and achieved the following held-out test performance:

| Metric | ResNet18 | Stage 8 VLM |
|---|---:|---:|
| Accuracy | 0.8285 | 0.6624 |
| Balanced accuracy | 0.6430 | 0.2747 |
| Macro F1 | 0.6658 | 0.2686 |

The specialized ResNet18 substantially outperformed the Stage 8 VLM as an image
classifier.


### 9.1 Architectural Difference

The two experiments should not be interpreted as equivalent training
strategies.

The ResNet18 experiment allowed the visual feature extractor itself to adapt to
HAM10000.

The Stage 8 VLM experiment instead used:

    frozen general-purpose vision encoder
                    |
                    v
          pretrained visual features
                    |
                    v
        language-side LoRA adapters
                    |
                    v
          generated class label

Only approximately 230,400 language-side LoRA parameters were trained.

Consequently, Stage 8 primarily tested whether a small language-side adaptation
could learn to map SmolVLM's existing visual representations to HAM10000 class
labels. It did not test whether adapting SmolVLM's vision encoder could improve
dermatology-specific feature extraction.

The difference in performance therefore does not establish that VLM
architectures are inherently unsuitable for this task. It establishes that,
under the specific Stage 8 adaptation strategy, the specialized ResNet18 was a
much stronger HAM10000 classifier.


### 9.2 What Stage 8 Demonstrated

Classification accuracy was not the only objective of Stage 8.

The experiment demonstrated an end-to-end NVIDIA NeMo AutoModel workflow for
multimodal adaptation, including:

- adapting a pretrained VLM using PEFT/LoRA,
- constructing multimodal image/instruction/response training examples,
- diagnosing and correcting an inappropriate label-masking objective,
- validating the training pipeline with a tiny-overfit experiment,
- working with NeMo's distributed data-loading behavior,
- designing controlled class-sampling experiments,
- selecting checkpoints using validation data,
- performing target-free generative evaluation, and
- preserving a held-out test set until the experiment was locked.

The final result also demonstrates an important modeling lesson: a
general-purpose multimodal model with a small parameter-efficient adaptation is
not automatically superior to a specialized vision model for a narrow visual
classification task.


## 10. Conclusions and Limitations

### 10.1 Conclusions

Stage 8 successfully implemented and evaluated a complete VLM adaptation
workflow using NVIDIA NeMo AutoModel and LoRA.

The experiment established several findings:

1. **The training objective must be verified, not assumed.**

   NeMo's generic label-matching fallback was inappropriate for a prompt that
   already contained the possible answer tokens. A custom response-only
   collator was required to ensure that loss was computed on the assistant
   response rather than class names appearing in the instruction.

2. **The corrected VLM training pipeline can learn the intended objective.**

   The seven-example tiny-overfit experiment reached low training loss and
   reproduced all seven labels correctly using target-free generation.

3. **Class distribution strongly affects generative classification behavior.**

   Natural sampling strongly favored the majority `nv` class. Equal-class
   sampling greatly increased minority-class recall but produced many false
   positives. Tempered sampling provided an intermediate distribution and
   achieved the highest validation macro F1 of the three strategies.

4. **Overall accuracy is insufficient for this dataset.**

   The final VLM achieved 66.24% held-out test accuracy but only 27.47%
   balanced accuracy and 26.86% macro F1. The difference is explained largely
   by strong performance on the majority `nv` class and poor performance on
   rare classes.

5. **The specialized vision model remained substantially stronger.**

   The earlier fully fine-tuned ResNet18 achieved 82.85% accuracy, 64.30%
   balanced accuracy, and 66.58% macro F1 on the held-out test set.

6. **Parameter-efficient VLM adaptation and specialized vision fine-tuning
   solve different problems.**

   Stage 8 trained only small language-side LoRA adapters while leaving
   SmolVLM's general-purpose vision encoder frozen. The ResNet18 experiment
   directly adapted its visual feature extractor to HAM10000.


### 10.2 Limitations

Several limitations should be considered when interpreting these results.

#### Frozen vision encoder

SmolVLM's vision encoder was not adapted to HAM10000. The experiment therefore
does not determine how vision-side LoRA, selective vision fine-tuning, or full
multimodal fine-tuning would perform.

#### Small VLM

Stage 8 used the 256M-parameter SmolVLM variant. Larger VLMs may produce
different results, although increasing model size alone does not guarantee
better specialized visual classification.

#### Severe class imbalance

HAM10000 contains large differences in class frequency. The rare `df` and
`vasc` classes contain relatively few training examples, making both training
and evaluation difficult.

#### Limited sampling search

Only three sampling strategies were used for the corrected full experiments:

- natural distribution,
- equal-class distribution, and
- tempered distribution with `alpha = 0.5`.

Additional sampling distributions were deliberately not explored after Stage
8C was selected in order to avoid continued optimization against the validation
set.

#### Lesion-level rather than patient-level splitting

The dataset does not provide a usable patient identifier for this project.
Splitting by lesion prevents images of the same lesion from appearing across
splits, but it cannot guarantee complete patient independence.

#### Classification is not clinical diagnosis

HAM10000 classification is used here as a machine-learning benchmark. The
models and results in this project are educational and experimental and should
not be interpreted as a clinically validated diagnostic system.


## 11. Reproducibility

### 11.1 Primary Stage 8 Files

Important implementation files include:

    src/stage8/dataset.py
    src/stage8/nemo_collator.py
    src/stage8/inference.py
    src/stage8/evaluate_validation.py

The final Stage 8C training configuration is:

    configs/stage8/smolvlm_nemo_train_tempered.yaml

The tiny-overfit diagnostic configuration is:

    configs/stage8/smolvlm_nemo_tiny_overfit.yaml


### 11.2 Final Stage 8C Checkpoint

The validation-selected checkpoint is:

    /workspace/storage/skin-lesion-ai/outputs/stage8/lora-b64-tempered/epoch_3_step_439

The `LOWEST_VAL` link in the Stage 8C output directory points to this
checkpoint.


### 11.3 Evaluation Outputs

Validation outputs:

    /workspace/storage/skin-lesion-ai/outputs/stage8/validation-evaluation-tempered/

Held-out test outputs:

    /workspace/storage/skin-lesion-ai/outputs/stage8/test-evaluation-tempered/

Each evaluation directory contains:

    predictions.csv
    metrics.txt


### 11.4 Held-Out Test Evaluation Command

The final held-out evaluation was run from the repository root with:

    export HF_HOME=/workspace/storage/skin-lesion-ai/huggingface

    python -m src.stage8.evaluate_validation \
      --checkpoint /workspace/storage/skin-lesion-ai/outputs/stage8/lora-b64-tempered/LOWEST_VAL \
      --output-dir /workspace/storage/skin-lesion-ai/outputs/stage8/test-evaluation-tempered \
      --split test


### 11.5 Experimental Sequence

The Git history preserves several important Stage 8 milestones:

    57d8c49  Correct NeMo VLM training objective and add Stage 8 evaluation
    ad4374b  Add balanced sampling experiment for Stage 8 VLM
    8d147ce  Add tempered sampling experiment for Stage 8 VLM
    cf87bb4  Support locked Stage 8 test evaluation

This history documents that the tempered Stage 8C experiment was selected and
committed before the held-out test split was enabled for Stage 8 evaluation.
