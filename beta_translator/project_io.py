"""Local video, reviewed-input, and cache I/O for the notebook."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import cv2
import numpy as np

from .inverse_kinematics import LIMBS, make_body


def file_sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_frame(video_file, frame_index):
    cap = cv2.VideoCapture(str(video_file))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f'Could not read frame {frame_index} from {video_file}.')
    return frame


def find_ffmpeg():
    executable = shutil.which('ffmpeg')
    if executable is None and Path('/opt/homebrew/bin/ffmpeg').is_file():
        executable = '/opt/homebrew/bin/ffmpeg'
    if executable is None:
        raise FileNotFoundError('Install FFmpeg locally, for example with brew install ffmpeg.')
    return executable


def convert_video(source, destination):
    subprocess.run([find_ffmpeg(), '-nostdin', '-y', '-loglevel', 'error', '-i', str(source),
                    '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(destination)], check=True)


def save_npz(path, **arrays):
    """Write a complete cache before replacing the old one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_reviewed_holds(path, video_file):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata'].item()))
        if metadata['video_sha256'] != file_sha256(video_file):
            raise ValueError('This hold map belongs to a different video. Use a new output folder and fresh detection.')
        if metadata['reference_frame_number'] != 0:
            raise ValueError('This workflow requires a frame-0 hold map.')
        records = metadata['records']
        masks = {record['hold_id']: saved[record['hold_id']].astype(bool) for record in records}
        reference = saved['reference_frame']
        if any(mask.shape != reference.shape[:2] or not mask.any() for mask in masks.values()):
            raise ValueError('Saved hold masks have invalid dimensions or empty regions.')
    selected = metadata['selected_hold_ids']
    if len(masks) != len(records) or len(selected) != len(set(selected)) or not set(selected).issubset(masks):
        raise ValueError('Saved hold IDs are duplicated or missing.')
    return records, masks, metadata['candidate_run_id'], selected


def restore_reviewed_map(bundle, destination, video_file):
    """Restore only the reviewed map; poses and alignment follow their own cache checks."""
    if not Path(bundle).is_file():
        raise FileNotFoundError('No reviewed map found. Set USE_SAVED_HOLD_MAP = False to detect new holds.')
    with zipfile.ZipFile(bundle) as archive:
        name = ('state/reviewed_hold_map.npz' if 'state/reviewed_hold_map.npz' in archive.namelist()
                else 'reviewed_hold_map.npz')
        data = archive.read(name)
    load_reviewed_holds(io.BytesIO(data), video_file)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def save_reviewed_holds(path, video_file, reference, records, masks, run_id, selected):
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(masks):
        raise ValueError('Select unique reviewed hold IDs before saving.')
    metadata = dict(video_sha256=file_sha256(video_file), reference_frame_number=0,
                    candidate_run_id=run_id, records=records, selected_hold_ids=list(selected))
    save_npz(path, metadata=np.array(json.dumps(metadata)), reference_frame=reference, **masks)


def save_alignment(path, transforms, video_file, pose_file, settings):
    frames = sorted(transforms)
    save_npz(path, frames=np.array(frames), matrices=np.array([transforms[f][0] for f in frames]),
             inliers=np.array([transforms[f][1] for f in frames]),
             statuses=np.array([transforms[f][2] for f in frames]),
             video_sha256=np.array(file_sha256(video_file)), pose_sha256=np.array(file_sha256(pose_file)),
             settings=np.array(json.dumps(settings, sort_keys=True)))


def load_alignment(path, video_file, pose_file, settings):
    """Return None for a missing or outdated cache so the notebook recomputes it."""
    if not Path(path).is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        required = {'video_sha256', 'pose_sha256', 'settings', 'frames', 'matrices', 'inliers', 'statuses'}
        if not required.issubset(saved.files):
            return None
        if (str(saved['video_sha256'].item()) != file_sha256(video_file)
                or str(saved['pose_sha256'].item()) != file_sha256(pose_file)
                or json.loads(str(saved['settings'].item())) != settings):
            return None
        lengths = [len(saved[name]) for name in ('frames', 'matrices', 'inliers', 'statuses')]
        if len(set(lengths)) != 1 or not lengths[0] or not np.isfinite(saved['matrices']).all():
            return None
        return {int(f): (H.copy(), int(n), str(status)) for f, H, n, status in
                zip(saved['frames'], saved['matrices'], saved['inliers'], saved['statuses'])}


def validate_annotations(data, video_file, run_id, hold_ids, frame_count):
    if (data['video_sha256'] != file_sha256(video_file) or data['candidate_run_id'] != run_id
            or data['selected_hold_ids'] != list(hold_ids)):
        raise ValueError('These endpoint annotations belong to a different video or reviewed route.')
    frames, contacts = data['frames'], data['contacts']
    if set(frames) != {'before', 'after'} or set(contacts) != set(frames):
        raise ValueError('Annotations must contain BEFORE and AFTER endpoints.')
    allowed = set(hold_ids) | {'unknown', 'free', 'wall_contact'}
    for phase, frame in frames.items():
        if frame is not None and (type(frame) is not int or not 0 <= frame < frame_count):
            raise ValueError(f'Invalid {phase} frame.')
        if set(contacts[phase]) != set(LIMBS) or any(value not in allowed for value in contacts[phase].values()):
            raise ValueError(f'Invalid {phase} contact labels.')
    if all(frame is not None for frame in frames.values()) and frames['before'] >= frames['after']:
        raise ValueError('BEFORE must be earlier than AFTER.')
    if data['moving_limb'] not in LIMBS:
        raise ValueError('Choose one of the four limbs.')
    body = data['body']
    make_body(body['height_cm'], body['ratios'])
    scale = body['px_per_cm']
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError('Wall scale must be positive, or None when unmeasured.')
    return data


def save_annotations(path, data, video_file, run_id, hold_ids, frame_count):
    validate_annotations(data, video_file, run_id, hold_ids, frame_count)
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def load_annotations(path, video_file, run_id, hold_ids, frame_count):
    return validate_annotations(json.loads(Path(path).read_text()), video_file, run_id, hold_ids, frame_count)
