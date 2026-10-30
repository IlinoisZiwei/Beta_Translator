"""Deploy trained contacts beside reviewed outputs, preserving source-frame identities."""
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch

from .contact_decoding import ensemble_probabilities, decode_probabilities, export_predictions
from .contact_features import build_features, FEATURE_VERSION, LANDMARKS
from .contact_network import ContactTCN, predict_logits
from .project_io import load_reviewed_holds
from .contact_review import review_annotation_from_prediction
from .wholebody_pose import sha256, POSE_SPEC


def load_package(directory, device='cpu'):
    directory = Path(directory)
    config = json.loads((directory/'deployment.json').read_text())
    if config['feature_version'] != FEATURE_VERSION:
        raise ValueError('Unsupported feature version; use matching model and code.')
    models = []
    for entry in config['checkpoints']:
        filename = Path(entry['file'])
        if filename.name != entry['file']:
            raise ValueError('Invalid checkpoint path.')
        path = directory/filename
        if sha256(path) != entry['sha256']:
            raise ValueError(f'Model integrity check failed: {filename}')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['feature_version'] != FEATURE_VERSION:
            raise ValueError('Checkpoint feature version mismatch.')
        model = ContactTCN(**saved['model']).to(device)
        model.load_state_dict(saved['state_dict']); model.eval(); models.append(model)
    return config, models


def predict_contacts(points, scores, hold_boxes, hold_ids, fps, package, device='cpu', alignment_valid=None):
    """Accept stabilized whole-body points and a hold map; return native-frame results."""
    config, models = load_package(package, device)
    count = len(points)
    if count < 2:
        raise ValueError('Provide at least two consecutive video frames.')
    original_time = np.arange(count) / fps
    rate = config['training_fps']
    model_time = np.arange(int(np.floor(original_time[-1]*rate))+1) / rate
    if len(model_time) < 2:
        raise ValueError('Video is too short for temporal inference.')
    # Resampling establishes the temporal scale without claiming subframe precision.
    sampled_points = interp1d(original_time, points, axis=0)(model_time)
    sampled_scores = interp1d(original_time, scores, axis=0)(model_time)
    features, _ = build_features(sampled_points, sampled_scores, hold_boxes, rate)
    logits = [predict_logits(model, features, device) for model in models]
    probabilities = ensemble_probabilities(logits, config['temperature'])
    probabilities = interp1d(model_time, probabilities, axis=0, bounds_error=False,
                             fill_value=(probabilities[0], probabilities[-1]))(original_time)
    decoded = decode_probabilities(probabilities, config['switch_penalty'] * fps / rate)
    observed = np.stack([np.isfinite(points[:, indices]).all(axis=(1, 2)) &
                          np.isfinite(scores[:, indices]).all(axis=1) &
                          (np.mean(scores[:, indices], axis=1) > .5) for indices in LANDMARKS], axis=1)
    if alignment_valid is not None:
        observed &= np.asarray(alignment_valid, bool)[:, None]
    rows, events, moves = export_predictions(probabilities, decoded, hold_ids, fps,
        config['accept_threshold'], observed, config.get('max_timing_width_sec', -1.),
        config.get('movement_accept_threshold', 1.01))
    event_columns = ['limb', 'hold_id', 'start_frame', 'end_frame', 'start_sec', 'end_sec',
                     'confidence', 'accepted_fraction', 'requires_review']
    return {'contacts': pd.DataFrame(rows), 'events': pd.DataFrame(events, columns=event_columns), 'moves': moves,
            'probabilities': probabilities, 'config': config}


def run_automatic_contacts(video, pose_cache, reviewed_output, package, destination, device='cpu'):
    """Use existing reviewed hold map and alignment; write only to a separate output folder."""
    video, pose_cache, reviewed_output, package, destination = map(Path,
        (video, pose_cache, reviewed_output, package, destination))
    if destination.resolve() == reviewed_output.resolve():
        raise ValueError('Choose a separate automatic-output directory to preserve reviewed tables.')
    records, masks, run_id, hold_ids = load_reviewed_holds(reviewed_output/'state/reviewed_hold_map.npz', video)
    boxes = []
    for hold in hold_ids:
        ys, xs = np.nonzero(masks[hold])
        boxes.append([xs.min(), ys.min(), xs.max()+1, ys.max()+1])
    with np.load(pose_cache, allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata']))
        if (metadata['video_sha256'] != sha256(video) or metadata['model'] != POSE_SPEC
                or metadata.get('tracking_version') != 2):
            raise ValueError('Whole-body cache belongs to another video or pose model.')
        expected = json.loads((package/'deployment.json').read_text()).get('pose_backbone')
        if expected is not None and any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError('Pose extractor does not match the trained contact package.')
        points, scores = cache['keypoints'].copy(), cache['scores'].copy()
    alignment = reviewed_output/'state/alignment_cache.npz'
    valid = np.zeros(len(points), bool)
    with np.load(alignment, allow_pickle=False) as saved:
        if str(saved['video_sha256'].item()) != sha256(video):
            raise ValueError('Alignment belongs to a different video.')
        for frame, H, inliers, status in zip(saved['frames'], saved['matrices'], saved['inliers'], saved['statuses']):
            frame = int(frame)
            if not 0 <= frame < len(points): continue
            if status == 'ok' and inliers >= 25 and np.isfinite(H).all() and np.linalg.cond(H) < 1e12:
                points[frame] = cv2.perspectiveTransform(points[frame, :, None], np.linalg.inv(H))[:, 0]
                valid[frame] = True
    points[~valid] = np.nan; scores[~valid] = 0
    result = predict_contacts(points, scores, np.asarray(boxes), hold_ids, metadata['fps'], package, device, valid)
    destination.mkdir(parents=True, exist_ok=True)
    result['contacts'].to_csv(destination/'limb_contacts.csv', index=False)
    result['events'].to_csv(destination/'contact_events.csv', index=False)
    summary = {'schema_version': 1, 'video_sha256': sha256(video), 'pose_cache_sha256': sha256(pose_cache),
        'alignment_sha256': sha256(alignment), 'model_config_sha256': sha256(package/'deployment.json'),
        'reviewed_map_sha256': sha256(reviewed_output/'state/reviewed_hold_map.npz'),
        'candidate_run_id': run_id, 'selected_hold_ids': hold_ids, 'fps': metadata['fps'],
        'frame_indexing': 'Zero-based native video frames', 'model_fps': result['config']['training_fps'],
        'accept_threshold': result['config']['accept_threshold'],
        'max_timing_width_sec': result['config'].get('max_timing_width_sec', -1.),
        'movement_accept_threshold': result['config'].get('movement_accept_threshold', 1.01),
        'accepted_frame_fraction': float((result['contacts']['status'] == 'accepted_model').mean()),
        'eligible_moves': sum(m['eligible_for_ik'] for m in result['moves']), 'moves': result['moves'],
        'artifact_sha256': {name: sha256(destination/name)
                           for name in ('limb_contacts.csv', 'contact_events.csv')}}
    (destination/'automatic_contacts.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez_compressed(destination/'contact_probabilities.npz', probabilities=result['probabilities'],
                        hold_ids=np.asarray(hold_ids), fps=np.array(metadata['fps']))
    return summary


def motion_annotation_from_prediction(template, move, prediction):
    """Create an IK-compatible annotation only for a fully supported automatic move.

    The template supplies measured wall scale and anthropometry, not contact labels.
    Return a new object; the user's reviewed annotation file is never overwritten.
    """
    if not move.get('eligible_for_ik') or move.get('requires_review'):
        raise ValueError('This movement needs contact/timing review before IK.')
    for key in ('video_sha256', 'candidate_run_id', 'selected_hold_ids'):
        if template[key] != prediction[key]:
            raise ValueError('Template and prediction refer to different video/hold inputs.')
    annotation = copy.deepcopy(template)
    moving = move['moving_limb']
    annotation['frames'] = {'before': move['before_frame'], 'after': move['after_frame']}
    annotation['moving_limb'] = moving
    annotation['contacts'] = {
        'before': {**move['support_contacts'], moving: move['from_hold']},
        'after': {**move['support_contacts'], moving: move['to_hold']}}
    annotation['motion'] = {'release_frame': move['release_frame'], 'arrival_frame': move['arrival_frame'],
                            'support_contacts_maintained': move['supports_maintained']}
    annotation['annotation_source'] = 'automatic_contact_model'
    annotation['automatic_evidence'] = {'model_config_sha256': prediction['model_config_sha256'],
        'accept_threshold': prediction['accept_threshold'],
        'release_frame_range': move.get('release_frame_range'),
        'arrival_frame_range': move.get('arrival_frame_range'),
        'timing_range_semantics': move.get('timing_range_semantics')}
    from .motion_ik import validate_motion_annotation
    validate_motion_annotation(annotation)
    return annotation



def run_automatic_motion(video, reviewed_output, prediction_file, move_index, destination, settings=None):
    """Run existing IK for an eligible proposal in an isolated output directory."""
    video, reviewed_output, prediction_file, destination = map(Path,
        (video, reviewed_output, prediction_file, destination))
    if destination.resolve() == reviewed_output.resolve():
        raise ValueError('Automatic IK needs a separate destination.')
    prediction = json.loads(prediction_file.read_text())
    for key, source in [('video_sha256', video),
                        ('reviewed_map_sha256', reviewed_output/'state/reviewed_hold_map.npz'),
                        ('alignment_sha256', reviewed_output/'state/alignment_cache.npz')]:
        if prediction[key] != sha256(source):
            raise ValueError('Automatic prediction inputs changed; rerun inference.')
    template = json.loads((reviewed_output/'state/endpoint_annotations.json').read_text())
    move = prediction['moves'][move_index]
    annotation = motion_annotation_from_prediction(template, move, prediction)
    for name in ('tables/pose_coordinates.csv', 'state/reviewed_hold_map.npz', 'state/alignment_cache.npz'):
        target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reviewed_output/name, target)
    (destination/'state/endpoint_annotations.json').write_text(json.dumps(annotation, indent=2)+'\n')
    from .motion_pipeline import run_motion_analysis
    return run_motion_analysis(video, destination, settings=settings)
