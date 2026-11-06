"""Fail early if a shared notebook dependency or compiled detector is unusable."""
import importlib
import shutil
import sys


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Beta Translator requires Python 3.12.')
    for name in ('cv2', 'numpy', 'pandas', 'scipy', 'osqp', 'matplotlib', 'PIL', 'yaml',
                 'mediapipe', 'torch', 'torchvision', 'rtmlib', 'onnxruntime',
                 'ipywidgets', 'ipykernel', 'jupyterlab', 'detectron2._C',
                 'anny', 'roma', 'trimesh', 'safetensors', 'warp'):
        importlib.import_module(name)
    import torch
    from torchvision.ops import nms
    from detectron2.layers import nms_rotated
    nms(torch.tensor([[0., 0., 2., 2.]]), torch.tensor([1.]), .5)
    nms_rotated(torch.tensor([[0., 0., 2., 2., 0.]]), torch.tensor([1.]), .5)
    for name in ('ffmpeg', 'ffprobe'):
        if not shutil.which(name):
            raise RuntimeError(f'{name} is missing; rerun setup.sh.')
    print(f'Shared environment and compiled operators OK: {sys.executable}')


if __name__ == '__main__':
    main()
