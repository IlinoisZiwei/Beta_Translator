"""Run an installed contact model against the existing reviewed pipeline inputs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beta_translator.automatic_contacts import run_automatic_contacts

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('video', 'pose-cache', 'reviewed-output', 'package', 'destination'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    result = run_automatic_contacts(a.video, a.pose_cache, a.reviewed_output, a.package, a.destination, a.device)
    print(json.dumps(result, indent=2))
