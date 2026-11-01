"""Restore missing/LFS-pointer demo assets for both git clones and GitHub ZIPs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import tempfile
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
POINTER = b'version https://git-lfs.github.com/spec/v1'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore_asset(root, asset, url):
    path = root / asset['path']
    if path.is_file():
        with path.open('rb') as stream:
            pointer = stream.read(len(POINTER)) == POINTER
        if not pointer:
            if Path(asset['path']).parts[0] == 'outputs':
                return  # User-generated results are never overwritten by setup.
            if path.stat().st_size == asset['bytes'] and sha256(path) == asset['sha256']:
                return
            raise ValueError(f'{path} differs from the bundled input. Preserve it elsewhere '
                             'before rerunning setup; it has not been overwritten.')
    import certifi
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {asset["path"]} ({asset["bytes"] / 1e6:.1f} MB)...', flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.part', delete=False) as stream:
            temporary = Path(stream.name)
            with urlopen(url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response:
                shutil.copyfileobj(response, stream)
        if temporary.stat().st_size != asset['bytes'] or sha256(temporary) != asset['sha256']:
            raise ValueError(f'Download verification failed: {asset["path"]}')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_pose_models():
    import certifi
    from rtmlib.tools.file import download_checkpoint
    os.environ['TORCH_HOME'] = str(ROOT / 'training_data/model_cache')
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    models = [
        ('rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
         '3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23'),
        ('rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip',
         'bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054'),
    ]
    for suffix, expected in models:
        path = Path(download_checkpoint('https://download.openmmlab.com/mmpose/v1/projects/' + suffix))
        if sha256(path) != expected:
            raise ValueError(f'Pose model checksum mismatch: {path}. Preserve/remove that file and retry setup.')
    print('Frozen pose models verified; no training or video processing was run.')


def main():
    manifest = json.loads((ROOT / 'setup-assets.json').read_text())
    for asset in manifest['assets']:
        host = 'raw.githubusercontent.com' if asset.get('storage') == 'git' else 'media.githubusercontent.com/media'
        url = f'https://{host}/{manifest["repository"]}/{manifest["revision"]}/{quote(asset["path"])}'
        restore_asset(ROOT, asset, url)
    prepare_pose_models()
    print('Demo inputs and videos are ready. Existing results have been preserved.')


if __name__ == '__main__':
    main()
