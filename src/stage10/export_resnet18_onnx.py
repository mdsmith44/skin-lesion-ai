"""Export the selected Stage 3/4 ResNet18 network to persistent ONNX storage.

The graph accepts already-normalized RGB float32 tensors [N, 3, 224, 224]
and returns seven logits. Preprocessing and decision logic stay outside it.
Run from the repository root: python -m src.stage10.export_resnet18_onnx
"""
import hashlib
import json
from pathlib import Path

import onnx
import torch
import torchvision

from src.stage9.resnet18_inference import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    ResNet18Inference,
)


OUTPUT_DIR = Path("/workspace/storage/skin-lesion-ai/outputs/stage10/resnet18")
ONNX_PATH = OUTPUT_DIR / "resnet18_full_finetuned.onnx"
METADATA_PATH = OUTPUT_DIR / "resnet18_full_finetuned.metadata.json"
INPUT_NAME = "normalized_rgb"
OUTPUT_NAME = "logits"
INPUT_SHAPE = ["batch_size", 3, 224, 224]
OUTPUT_SHAPE = ["batch_size", 7]
PREPROCESSING = {
    "outside_onnx_graph": True,
    "steps": ["PIL RGB conversion", "Resize", "ToTensor", "Normalize"],
    "resize_hw": [224, 224],
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "augmentation": False,
    "softmax_argmax_and_label_conversion": "outside ONNX graph",
    "reference": "notebooks/04_model_evaluation.ipynb",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifact(path):
    model = onnx.load(path, load_external_data=True)
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    if [value.name for value in model.graph.input] != [INPUT_NAME]:
        raise ValueError("Unexpected ONNX graph input name/count.")
    if [value.name for value in model.graph.output] != [OUTPUT_NAME]:
        raise ValueError("Unexpected ONNX graph output name/count.")

    def dimensions(value):
        return [dim.dim_param if dim.dim_param else dim.dim_value
                for dim in value.type.tensor_type.shape.dim]

    inp, out = model.graph.input[0], model.graph.output[0]
    if dimensions(inp) != INPUT_SHAPE or dimensions(out) != OUTPUT_SHAPE:
        raise ValueError(f"Unexpected inferred graph shapes: {dimensions(inp)}, {dimensions(out)}")
    if inp.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX input is not float32.")
    if out.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX output is not float32.")
    return model


def main():
    checkpoint_path = Path(DEFAULT_CHECKPOINT).resolve(strict=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classifier = ResNet18Inference(checkpoint_path=checkpoint_path, device="cpu")
    classifier._model.eval()
    dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    torch.onnx.export(
        classifier._model,
        (dummy,),
        str(ONNX_PATH),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=20,
        dynamic_axes={
            INPUT_NAME: {0: "batch_size"},
            OUTPUT_NAME: {0: "batch_size"},
        },
        dynamo=False,
        export_params=True,
        do_constant_folding=True,
    )

    metadata = {
        "artifact": str(ONNX_PATH),
        "onnx_sha256": sha256(ONNX_PATH),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": classifier._provenance["checkpoint_sha256"],
        "architecture": "torchvision ResNet18",
        "class_names_in_logit_order": list(CLASS_NAMES),
        "class_to_idx": classifier._provenance["class_to_idx"],
        "input": {"name": INPUT_NAME, "shape": INPUT_SHAPE, "dtype": "float32"},
        "output": {"name": OUTPUT_NAME, "shape": OUTPUT_SHAPE, "dtype": "float32",
                   "meaning": "raw seven-class logits"},
        "export_dummy_shape": [1, 3, 224, 224],
        "dynamic_batch": True,
        "opset": 20,
        "preprocessing_contract": PREPROCESSING,
        "postprocessing_contract": "softmax, argmax, and class-name conversion are outside the graph",
        "versions": {
            "python": __import__("sys").version,
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "onnx": onnx.__version__,
            "onnxruntime": None,
            "onnx_validation": "onnx.checker.check_model + ONNX shape inference",
        },
    }
    proto = onnx.load(ONNX_PATH)
    for key, value in metadata.items():
        entry = proto.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    onnx.save(proto, ONNX_PATH)

    checked_model = validate_artifact(ONNX_PATH)
    metadata["onnx_sha256"] = sha256(ONNX_PATH)
    metadata["validation"] = {
        "status": "passed",
        "checker": "onnx.checker.check_model",
        "shape_inference": "passed",
        "graph_inputs": [value.name for value in checked_model.graph.input],
        "graph_outputs": [value.name for value in checked_model.graph.output],
        "inferred_input_shape": INPUT_SHAPE,
        "inferred_output_shape": OUTPUT_SHAPE,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
