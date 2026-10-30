"""Verified automatic drafts for the existing human-reviewed notebook workflow.

This module intentionally has no Torch/RTMW dependency. Inference can run in a
separate environment while the MediaPipe/IK notebook keeps its existing kernel.
"""
import copy
import json
from pathlib import Path

import pandas as pd

from .inverse_kinematics import LIMBS
from .project_io import file_sha256, load_reviewed_holds


def load_review_bundle(video, reviewed_output, pose_cache, package, destination):
    """Reject stale inputs, changed model weights, or incomplete prediction tables."""
    video, reviewed_output, pose_cache, package, destination = map(Path,
        (video, reviewed_output, pose_cache, package, destination))
    prediction = json.loads((destination / 'automatic_contacts.json').read_text())
    inputs = {
        'video_sha256': video,
        'pose_cache_sha256': pose_cache,
        'alignment_sha256': reviewed_output / 'state/alignment_cache.npz',
        'reviewed_map_sha256': reviewed_output / 'state/reviewed_hold_map.npz',
        'model_config_sha256': package / 'deployment.json',
    }
    for key, path in inputs.items():
        if prediction.get(key) != file_sha256(path):
            raise ValueError(f'Stale automatic predictions: {key}. Refresh Section 6B.')
    config = json.loads((package / 'deployment.json').read_text())
    for entry in config['checkpoints']:
        if Path(entry['file']).name != entry['file']:
            raise ValueError('Invalid checkpoint filename.')
        if file_sha256(package / entry['file']) != entry['sha256']:
            raise ValueError('Automatic model checkpoint changed.')
    _, _, run_id, hold_ids = load_reviewed_holds(
        reviewed_output / 'state/reviewed_hold_map.npz', video)
    if prediction['candidate_run_id'] != run_id or prediction['selected_hold_ids'] != hold_ids:
        raise ValueError('Automatic predictions belong to a different hold selection.')
    for name in ('limb_contacts.csv', 'contact_events.csv'):
        if prediction.get('artifact_sha256', {}).get(name) != file_sha256(destination / name):
            raise ValueError(f'Unverified automatic output: {name}. Refresh Section 6B.')
    contacts = pd.read_csv(destination / 'limb_contacts.csv')
    if contacts.empty or contacts.duplicated(['frame', 'limb']).any():
        raise ValueError('Automatic contacts are empty or have duplicate frame/limb rows.')
    expected = pd.MultiIndex.from_product(
        [range(int(contacts.frame.max()) + 1), LIMBS], names=['frame', 'limb'])
    if set(contacts.set_index(['frame', 'limb']).index) != set(expected):
        raise ValueError('Automatic contacts must include all four limbs on every frame.')
    if not set(contacts.hold_id).issubset(set(hold_ids) | {'unknown', 'free'}):
        raise ValueError('Automatic contacts contain an unknown hold ID.')
    return prediction, contacts


def review_annotation_from_prediction(template, move, prediction, contacts):
    """Prefill a separate draft; predicted timing never becomes human confirmation."""
    for key in ('video_sha256', 'candidate_run_id', 'selected_hold_ids'):
        if template[key] != prediction[key]:
            raise ValueError('Template and prediction refer to different inputs.')
    annotation = copy.deepcopy(template)
    annotation['frames'] = {'before': move['before_frame'], 'after': move['after_frame']}
    annotation['moving_limb'] = move['moving_limb']
    indexed = contacts.set_index(['frame', 'limb'], verify_integrity=True)
    annotation['contacts'] = {
        phase: {limb: str(indexed.loc[(frame, limb), 'hold_id']) for limb in LIMBS}
        for phase, frame in annotation['frames'].items()}
    annotation.pop('motion', None)
    annotation['annotation_source'] = 'automatic_review_draft'
    annotation['suggested_motion'] = {
        'release_frame': move['release_frame'], 'arrival_frame': move['arrival_frame'],
        'release_frame_range': move.get('release_frame_range'),
        'arrival_frame_range': move.get('arrival_frame_range'),
        'support_contacts_maintained': False}
    annotation['notes'] = ('Automatically prefilled draft. Review contact identities, '
                           'timing, and support continuity before IK.')
    return annotation
