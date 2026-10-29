"""Strict The Way Up annotation parsing and participant-disjoint data preparation."""
import csv
import json
from pathlib import Path
import re

import numpy as np

from .contact_features import build_features, FEATURE_VERSION
from .wholebody_pose import sha256

SPLITS = {'train': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6'],
          'validation': ['p7', 'p8'], 'test': ['p9', 'p10']}
LIMB_INDEX = {'lh': 0, 'rh': 1, 'lf': 2, 'rf': 3}


def participant_id(name):
    match = re.fullmatch(r'(p\d+)[ab]?', name)
    if match is None:
        raise ValueError(f'Unexpected participant: {name}')
    return match[1]


def read_labels(path, frames, holds):
    """Treat endpoints as inclusive; mask conflicting labels instead of overwriting.

    Zero-based indexing is corroborated by final endFrame == frame_count - 1.
    Frames outside each limb's first/last annotation are excluded from supervision.
    Occlusion is kept as an evaluation stratum, never an input feature.
    """
    labels = np.full((frames, 4), holds, np.int64)
    annotated = np.zeros((frames, 4), bool)
    occluded = np.zeros((frames, 4), bool)
    conflicts = np.zeros((frames, 4), bool)
    unknown = np.zeros((frames, 4), bool)
    events, warnings = [], []
    with open(path, newline='') as stream:
        rows = csv.DictReader(stream)
        name_key = rows.fieldnames[0]
        for row_num, row in enumerate(rows, 2):
            if not row[name_key].strip():
                continue
            raw_name = row[name_key].strip().lower()
            if re.fullmatch(r'(lh|rh|lf|rf)\s*\(wrong hold\)', raw_name):
                limb = LIMB_INDEX[raw_name[:2]]
                a, b = int(row['startFrame']), int(row['endFrame'])
                if not 0 <= a <= b < frames:
                    raise ValueError(f'{path}:{row_num}: invalid unknown-hold interval')
                unknown[a:b+1, limb] = True
                warnings.append(f'Row {row_num}: unidentified off-route hold excluded')
                continue
            match = re.fullmatch(r'(lh|rh|lf|rf)\s*(\d+|w)', raw_name)
            if match is None:
                raise ValueError(f'{path}:{row_num}: invalid limb / hold label')
            limb = LIMB_INDEX[match[1]]
            hold = holds+1 if match[2] == 'w' else int(match[2])
            if hold != holds+1 and not 0 <= hold < holds:
                raise ValueError(f'{path}:{row_num}: unknown hold {hold}')
            start, end = int(row['startFrame']), int(row['endFrame'])
            if end == frames:
                end = frames-1
                warnings.append(f'Row {row_num}: terminal endFrame clipped to last decoded frame')
            if start < 0 or end < start or start >= frames or end >= frames:
                raise ValueError(f'{path}:{row_num}: invalid frame range {start}:{end}/{frames}')
            span = slice(start, end+1)
            conflicts[span, limb] |= annotated[span, limb] & (labels[span, limb] != hold)
            labels[span, limb] = hold
            annotated[span, limb] = True
            ranges = []
            for value in (row.get('framesOccluded') or '').split(','):
                if not value.strip(): continue
                pair = re.fullmatch(r'\s*(\d+)\s*(?:-\s*(\d+))?\s*', value)
                if pair is None:
                    raise ValueError(f'{path}:{row_num}: invalid occlusion range {value}')
                a, b = int(pair[1]), int(pair[2] or pair[1])
                if a < start or b > end or b < a:
                    warnings.append(f'Row {row_num}: occlusion clipped to contact interval')
                a, b = max(start, a), min(end, b)
                if a <= b:
                    occluded[a:b+1, limb] = True
                    ranges.append([a, b])
            events.append({'limb': limb, 'hold': hold, 'start': start, 'end': end, 'occluded': ranges})
    for limb in range(4):
        used = np.flatnonzero(annotated[:, limb])
        if len(used):
            labels[:used[0], limb] = -100
            labels[used[-1]+1:, limb] = -100
        else:
            labels[:, limb] = -100
    labels[conflicts | unknown] = -100
    return labels, occluded, events, {'conflicting_limb_frames': int(conflicts.sum()),
                                    'unidentified_hold_frames': int(unknown.sum()), 'warnings': warnings}


def prepare(dataset, pose_dir, output):
    dataset, pose_dir, output = map(Path, (dataset, pose_dir, output))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'feature_version': FEATURE_VERSION, 'splits': SPLITS, 'clips': []}
    for video in sorted(dataset.glob('*/*.mp4')):
        if '_large' in video.stem: continue
        clip = f'{video.parent.name}_{video.stem}'
        pose_file = pose_dir / f'{clip}.npz'
        if not pose_file.exists():
            raise FileNotFoundError(f'Pose extraction is incomplete: {pose_file}')
        with np.load(pose_file, allow_pickle=False) as cache:
            meta = json.loads(str(cache['metadata']))
            backbone = {key: meta[key] for key in ('model', 'pose_sha256', 'detector_sha256',
                                                  'tracking_version', 'keypoint_format')}
            if 'pose_backbone' in manifest and manifest['pose_backbone'] != backbone:
                raise ValueError('Public pose caches mix different feature extractors.')
            manifest['pose_backbone'] = backbone
            if meta['video_sha256'] != sha256(video):
                raise ValueError(f'Pose/video mismatch: {clip}')
            annotation = video.with_name(video.stem + '_routeAnnotation.json')
            holds = json.loads(annotation.read_text())['holds']
            boxes = np.array([[h['topLeftCorner']['x'], h['topLeftCorner']['y'],
                               h['bottomRightCorner']['x'], h['bottomRightCorner']['y']] for h in holds])
            label_file = video.with_name(video.stem + '_holdUsage.csv')
            labels, occluded, events, audit = read_labels(label_file, meta['frames'], len(holds))
            features, distances = build_features(cache['keypoints'], cache['scores'], boxes, meta['fps'])
        participant = participant_id(video.parent.name)
        split = next(key for key, people in SPLITS.items() if participant in people)
        entry = {'id': clip, 'participant': participant, 'split': split, 'fps': meta['fps'],
                 'frames': meta['frames'], 'holds': len(holds), 'events': len(events),
                 'label_sha256': sha256(label_file), 'pose_sha256': sha256(pose_file),
                 'hold_map_sha256': sha256(annotation), 'audit': audit}
        np.savez_compressed(output / f'{clip}.npz', features=features.astype(np.float16),
                            distances=distances.astype(np.float16), labels=labels, occluded=occluded,
                            metadata=json.dumps(entry), events=json.dumps(events))
        manifest['clips'].append(entry)
        print(f'Prepared {clip}: {split}, {len(events)} events, {audit}', flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
