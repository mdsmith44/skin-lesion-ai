"""Build FP32 TensorRT and check one validation input; no benchmarking.

Run: python -m src.stage10.build_resnet18_fp32
Artifacts and logs are written only to persistent Stage 10 storage.
"""
import json
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image
import tensorrt as trt
import torch

from src.stage10.export_resnet18_onnx import ONNX_PATH, OUTPUT_DIR, sha256
from src.stage9.resnet18_inference import ResNet18Inference, CLASS_NAMES


def main():
    root = Path(__file__).resolve().parents[2]
    output = OUTPUT_DIR / ('fp32_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    engine_path = output / 'resnet18_full_finetuned.fp32.engine'
    report_path = output / 'metadata.json'
    command = [shutil.which('trtexec') or 'trtexec', f'--onnx={ONNX_PATH}',
               f'--saveEngine={engine_path}', '--minShapes=normalized_rgb:1x3x224x224',
               '--optShapes=normalized_rgb:8x3x224x224',
               '--maxShapes=normalized_rgb:32x3x224x224', '--noTF32',
               '--stronglyTyped', '--memPoolSize=workspace:1024', '--skipInference']
    report = {'status': 'building', 'tensorrt_version': trt.__version__,
              'onnx_path': str(ONNX_PATH), 'onnx_sha256': sha256(ONNX_PATH),
              'engine_path': str(engine_path), 'precision': 'FP32; TF32 disabled',
              'optimization_profile': {'min': [1, 3, 224, 224], 'opt': [8, 3, 224, 224],
                                       'max': [32, 3, 224, 224]},
              'build_command': shlex.join(command), 'build_argv': command,
              'benchmark_performed': False}

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    save()
    print(f'Artifacts: {output}', flush=True)
    try:
        with (output / 'build.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        report.update(engine_sha256=sha256(engine_path), engine_size_bytes=engine_path.stat().st_size,
                      status='built')
        save()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError('Engine deserialization failed')
        assert engine.num_io_tensors == 2 and engine.num_optimization_profiles == 1
        for name, shape, mode in [('normalized_rgb', (-1, 3, 224, 224), trt.TensorIOMode.INPUT),
                                  ('logits', (-1, 7), trt.TensorIOMode.OUTPUT)]:
            assert tuple(engine.get_tensor_shape(name)) == shape
            assert engine.get_tensor_dtype(name) == trt.float32
            assert engine.get_tensor_mode(name) == mode
        actual_profile = [list(s) for s in engine.get_tensor_profile_shape('normalized_rgb', 0)]
        assert actual_profile == list(report['optimization_profile'].values())
        context = engine.create_execution_context()
        for batch in (1, 8, 32):
            assert context.set_input_shape('normalized_rgb', (batch, 3, 224, 224))
            assert tuple(context.get_tensor_shape('logits')) == (batch, 7)
        report['io_validation'] = {'input': [-1, 3, 224, 224], 'output': [-1, 7],
                                   'dtype': 'float32', 'shape_checks_batches': [1, 8, 32]}
        # Only these columns are loaded; no diagnosis or test image is used.
        rows = pd.read_csv(root / 'data/processed/metadata_splits.csv',
                           usecols=['split', 'image_id', 'image_path'])
        row = rows.loc[rows['split'].eq('val')].iloc[0]
        parts = Path(row['image_path']).parts
        start = next(i for i in range(len(parts)-2) if parts[i:i+3] == ('data', 'raw', 'ham10000'))
        image_path = root.joinpath(*parts[start:]).resolve(strict=True)
        assert image_path.name == row['image_id'] + '.jpg'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        classifier = ResNet18Inference(device='cuda')
        source = json.loads((OUTPUT_DIR / 'resnet18_full_finetuned.metadata.json').read_text())
        assert report['onnx_sha256'] == source['onnx_sha256']
        assert classifier._provenance['checkpoint_sha256'] == source['source_checkpoint_sha256']
        with Image.open(image_path) as image:
            # Exactly one preprocessing invocation; both runtimes share this tensor.
            inputs = classifier._preprocess(image.convert('RGB')).unsqueeze(0).contiguous().cuda()
        with torch.inference_mode():
            expected = classifier._model(inputs)
        actual = torch.empty((1, 7), device='cuda', dtype=torch.float32)
        assert context.set_input_shape('normalized_rgb', tuple(inputs.shape))
        assert context.set_tensor_address('normalized_rgb', inputs.data_ptr())
        assert context.set_tensor_address('logits', actual.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        expected, actual = expected.cpu(), actual.cpu()
        assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
        max_diff = (expected - actual).abs().max().item()
        agrees = expected.argmax(1).item() == actual.argmax(1).item()
        close = torch.allclose(expected, actual, atol=1e-4, rtol=1e-4)
        report['equivalence'] = {
            'validation_image_id': row['image_id'], 'image_path': str(image_path),
            'image_sha256': sha256(image_path), 'reference_diagnosis_used': False,
            'input_shape': list(inputs.shape), 'preprocessing_applications': 1,
            'preprocessing': classifier._provenance['preprocessing'],
            'source_checkpoint_sha256': classifier._provenance['checkpoint_sha256'],
            'software': classifier._provenance['software'],
            'gpu': torch.cuda.get_device_name(), 'pytorch_tf32_enabled': False,
            'class_names_in_logit_order': list(CLASS_NAMES),
            'pytorch_logits': expected[0].tolist(), 'tensorrt_logits': actual[0].tolist(),
            'max_absolute_logit_difference': max_diff, 'argmax_agrees': agrees,
            'pytorch_argmax': expected.argmax(1).item(), 'tensorrt_argmax': actual.argmax(1).item(),
            'atol': 1e-4, 'rtol': 1e-4, 'allclose': close}
        report['status'] = 'passed' if close and agrees else 'equivalence_failed'
        save()
        print(json.dumps(report, indent=2), flush=True)
        if report['status'] != 'passed':
            raise RuntimeError('Single-input numerical equivalence failed')
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()
