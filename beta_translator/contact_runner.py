"""Refresh notebook proposals in the isolated contact-model Python environment."""
import argparse
import json
from pathlib import Path

import numpy as np

from .automatic_contacts import run_automatic_contacts
from .wholebody_pose import WholebodyExtractor, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    pose_cache = args.outputs / 'automatic/wholebody.npz'
    config = json.loads((args.package / 'deployment.json').read_text())
    # Check the frozen feature cache before constructing RTMW, which would otherwise
    # download/load large ONNX weights even for the bundled, already-extracted clip.
    if pose_cache.exists():
        with np.load(pose_cache, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
        expected = {**config['pose_backbone'], 'video_sha256': sha256(args.video)}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError('Stale whole-body cache. Choose a separate output folder for this video/model.')
        print('Using verified frozen whole-body features.', flush=True)
    else:
        extractor = WholebodyExtractor(args.outputs / 'automatic/model_cache', device='cpu')
        extractor.extract(args.video, pose_cache)
    summary = run_automatic_contacts(args.video, pose_cache, args.outputs, args.package,
                                    args.outputs / 'automatic/predictions', device=args.device)
    print(f"Saved {len(summary['moves'])} movement proposals; all proposals still require the configured review gate.")


if __name__ == '__main__':
    main()
