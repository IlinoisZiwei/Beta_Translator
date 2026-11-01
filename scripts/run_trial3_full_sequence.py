"""Create the separate 180 cm full-sequence diagnostic run from verified caches."""
from pathlib import Path
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from beta_translator.full_sequence_pipeline import run_full_sequence


if __name__ == '__main__':
    result = run_full_sequence(PROJECT, progress=lambda message: print(message, flush=True))
    print(json.dumps({k:result[k] for k in [
        'video_file', 'frame_count', 'optimizer', 'flag_counts',
        'max_contact_error_cm', 'max_segment_length_error_cm',
    ]}, indent=2))
