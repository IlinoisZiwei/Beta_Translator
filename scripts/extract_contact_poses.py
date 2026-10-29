"""Extract all public videos at native frame rate, or one pipeline video."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beta_translator.wholebody_pose import WholebodyExtractor


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', type=Path)
    source.add_argument('--video', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-cache', type=Path, default=Path('training_data/model_cache'))
    parser.add_argument('--device', choices=['cpu', 'coreml'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--clips', nargs='+', help='Optional participant_route clip IDs')
    args = parser.parse_args()
    extractor = WholebodyExtractor(args.model_cache, args.device, args.threads)
    if args.video:
        extractor.extract(args.video, args.output)
    else:
        videos = sorted(args.dataset.glob('*/*.mp4'))
        videos = [v for v in videos if '_large' not in v.stem]
        if args.clips:
            videos = [v for v in videos if f'{v.parent.name}_{v.stem}' in args.clips]
        print(f'Processing {len(videos)} videos', flush=True)
        for video in videos:
            extractor.extract(video, args.output / f'{video.parent.name}_{video.stem}.npz')


if __name__ == '__main__':
    main()
