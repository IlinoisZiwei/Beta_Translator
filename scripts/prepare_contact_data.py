"""Build participant-disjoint features after native-rate pose extraction."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beta_translator.wayup_data import prepare

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--poses', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    prepare(a.dataset, a.poses, a.output)
