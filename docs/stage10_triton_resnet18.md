# Stage 10 Triton ResNet18 model repository

This is a component-level note. The [Stage 10 deployment record](stage10_deployment.md)
is canonical for the operational service and its limitations.

The persistent Triton model repository is at
`/workspace/storage/skin-lesion-ai/triton/model_repository/`. Its `resnet18_fp16`
model has `config.pbtxt` and `1/model.plan`. The plan is a copy of the existing
Stage 10 TensorRT FP16 engine, not a rebuilt engine. The source engine and
copied `model.plan` have matching SHA-256
`7370bba2e48f106824f7f9a4d4ab6d065a5cf2b512311de6a47c95a5ac844cb3`.

The model uses Triton's `tensorrt_plan` platform and accepts `normalized_rgb` as
an FP32 RGB tensor of shape `[N, 3, 224, 224]`. It returns `logits` as an FP32
tensor of shape `[N, 7]`. `max_batch_size` is 32; the engine's single TensorRT
optimization profile has minimum batch 1, optimum batch 8, and maximum batch
32. The **active** config enables dynamic batching with preferred batch sizes
8, 16, and 32 and a maximum queue delay of 1,000 microseconds. These are
configured preferences; the saved concurrency benchmark does not prove which
batch sizes Triton actually executed. An archived pre-batching config remains
under `/workspace/storage/skin-lesion-ai/outputs/stage10/triton/configs/`.
Image loading, Stage 4 preprocessing and normalization, softmax, argmax, and
class-label conversion remain outside Triton.

The repository layout, config fields, tensor names, and plan checksum were
checked. Live Triton server loading and single-image inference were subsequently
verified against direct TensorRT FP16 using the same normalized validation
tensor: all seven raw logits matched exactly and argmax was `bkl` (index 2).
The active-config HTTP concurrency experiment and its provenance are recorded
under persistent `outputs/stage10/triton/`. The gateway's final end-to-end
smoke and performance limitations are in the canonical deployment record.
