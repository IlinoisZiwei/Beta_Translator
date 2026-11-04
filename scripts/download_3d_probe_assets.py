#!/usr/bin/env python3
"""Download pinned official DAV2 Small assets into a separate local cache."""
from pathlib import Path
import argparse,hashlib,io,json,shutil,ssl,tempfile,urllib.request,zipfile

REVISION='a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'
WEIGHT_SHA256='715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378'


def fetch(url):
    # Keep TLS verification enabled, including on Python.org macOS installations.
    bundle=Path('/etc/ssl/cert.pem')
    context=ssl.create_default_context(cafile=str(bundle) if bundle.exists() else None)
    return urllib.request.urlopen(url,timeout=60,context=context)


def prepare(directory):
    directory=Path(directory).expanduser();directory.mkdir(parents=True,exist_ok=True)
    source_url=f'https://github.com/DepthAnything/Depth-Anything-V2/archive/{REVISION}.zip'
    source=directory/'Depth-Anything-V2';manifest=directory/'source_manifest.json'
    cached=False
    if manifest.exists():
        m=json.loads(manifest.read_text())
        cached=m.get('revision')==REVISION and all((source/p).is_file() and hashlib.sha256((source/p).read_bytes()).hexdigest()==h for p,h in m['files'].items())
    if not cached:
        with fetch(source_url) as response:blob=response.read()
        with tempfile.TemporaryDirectory(prefix='dav2-source-',dir=directory) as temp:
            unpacked=Path(temp)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for entry in z.namelist():
                    rel=Path(*Path(entry).parts[1:])
                    if '..' in rel.parts or rel.is_absolute():raise ValueError('Unsafe source archive.')
                    if entry.endswith('/') or not rel.parts:continue
                    if rel.parts[0] not in ('depth_anything_v2','LICENSE','README.md'):continue
                    dst=unpacked/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(z.read(entry))
            source.mkdir(exist_ok=True)
            for p in unpacked.rglob('*'):
                if p.is_file():
                    dst=source/p.relative_to(unpacked);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        manifest.write_text(json.dumps(dict(revision=REVISION,url=source_url,files=hashes),indent=2)+'\n')
    weight=directory/'depth_anything_v2_vits.pth'
    if not weight.exists() or hashlib.sha256(weight.read_bytes()).hexdigest()!=WEIGHT_SHA256:
        url='https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth'
        with tempfile.NamedTemporaryFile(dir=directory,delete=False) as f:
            candidate=Path(f.name)
            try:
                with fetch(url) as response:
                    while block:=response.read(1024*1024):f.write(block)
                f.flush()
                if hashlib.sha256(candidate.read_bytes()).hexdigest()!=WEIGHT_SHA256:raise ValueError('Official weight hash changed; refusing to execute it.')
                candidate.replace(weight)
            finally:
                if candidate.exists():candidate.unlink()
    print('Verified official Small assets:',directory)
    return directory


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-dir',default=str(Path.home()/'.cache/beta-translator/depth-anything-v2-small'))
    prepare(p.parse_args().assets_dir)
