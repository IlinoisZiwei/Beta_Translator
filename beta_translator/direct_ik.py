"""User-enabled automatic contact-to-IK bridge, separate from validated model gates."""

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .contact_review import load_review_bundle
from .inverse_kinematics import LIMBS, make_body
from .motion_ik import validate_motion_annotation
from .project_io import file_sha256, load_alignment, save_annotations


def build_motion_candidates(prediction, contacts, body, pose_frames, alignment_frames):
    """Bypass confidence gates, but retain the solver's actual contact assumptions."""
    make_body(body['height_cm'], body['ratios'])
    allowed = set(prediction['selected_hold_ids'])
    indexed = contacts.set_index(['frame', 'limb'])
    if not indexed.index.is_unique:
        raise ValueError('Duplicate contact frame/limb rows.')
    rows = []
    for index, move in enumerate(prediction['moves']):
        before, after = move['before_frame'], move['after_frame']
        release, arrival = move['release_frame'], move['arrival_frame']
        moving = move['moving_limb']
        reasons, support_states, supports = [], {}, {}
        frames = list(range(before, after + 1))
        if not before < release < arrival <= after:
            reasons.append('No ordered contact / free / contact interval.')
        if not set(frames).issubset(pose_frames):
            reasons.append('Observed body poses are missing in this interval.')
        if not set(frames).issubset(alignment_frames):
            reasons.append('Wall alignment is unavailable in this interval.')
        for limb in LIMBS:
            keys = [(frame, limb) for frame in frames]
            if any(key not in indexed.index for key in keys):
                reasons.append(f'{limb}: missing contact rows.')
                continue
            # candidate_hold retains the decoded state even below the confidence threshold.
            labels = indexed.loc[keys, 'candidate_hold'].astype(str).tolist()
            if limb == moving:
                expected = [move['from_hold'] if f < release else
                            'free' if f < arrival else move['to_hold'] for f in frames]
                if labels != expected:
                    reasons.append('Moving-limb labels differ from the proposed release/arrival schedule.')
            else:
                states = list(dict.fromkeys(labels))
                support_states[limb] = states
                if len(states) == 1 and states[0] in allowed:
                    supports[limb] = states[0]
                elif len(states) == 1:
                    reasons.append(f'{limb}: {states[0]}; the current solver requires a fixed hold support.')
                else:
                    reasons.append(f'{limb}: contact changes within the proposed interval.')
        annotation = None
        if not reasons:
            annotation = {
                'video_sha256': prediction['video_sha256'],
                'candidate_run_id': prediction['candidate_run_id'],
                'selected_hold_ids': list(prediction['selected_hold_ids']),
                'reference_frame_number': prediction.get('reference_frame_number', 0),
                'frames': {'before': before, 'after': after},
                'moving_limb': moving,
                'contacts': {'before': {**supports, moving: move['from_hold']},
                             'after': {**supports, moving: move['to_hold']}},
                'body': copy.deepcopy(body),
                'motion': {'release_frame': release, 'arrival_frame': arrival,
                           'support_contacts_maintained': True},
                'annotation_source': 'automatic_predictions_experimental',
                'notes': 'Automatic decoded contacts accepted through the user-enabled bridge. '
                         'Support continuity is inferred from decoded labels, not individually human-verified.',
                'automatic_evidence': {'original_requires_review': move['requires_review'],
                    'original_eligible_for_ik': move['eligible_for_ik'],
                    'support_evidence': 'Same decoded hold on every frame; confidence gate bypassed.',
                    'release_frame_range': move.get('release_frame_range'),
                    'arrival_frame_range': move.get('arrival_frame_range')},
            }
            validate_motion_annotation(annotation)
        rows.append(dict(move_index=index, moving_limb=moving, from_hold=move['from_hold'],
                         to_hold=move['to_hold'], before_frame=before, after_frame=after,
                         release_frame=release, arrival_frame=arrival,
                         support_states=support_states, solver_compatible=annotation is not None,
                         reasons=reasons, annotation=annotation))
    return rows


def geometry_blockers(px_per_cm, single_panel_confirmed):
    reasons = []
    if px_per_cm is None or not np.isfinite(px_per_cm) or px_per_cm <= 0:
        reasons.append('PX_PER_CM is missing: local hole scales are not a whole-wall calibration.')
    if single_panel_confirmed is not True:
        reasons.append('A suitable single-panel interval has not been established for the 2D solver.')
    return reasons


def direct_ik_table(plan):
    """Compact notebook table; keep detailed evidence in the JSON artifact."""
    return pd.DataFrame([dict(move=row['move_index'], limb=row['moving_limb'],
        transition=f"{row['from_hold']} -> {row['to_hold']}",
        release=row['release_frame'], arrival=row['arrival_frame'],
        solver_compatible=row['solver_compatible'], status=row['status'],
        reason='; '.join(row['reasons'])) for row in plan['moves']])


def automatic_ik_artifacts(output_dir):
    root = Path(output_dir)
    paths = [root / 'state/direct_ik_plan.json', root / 'state/automatic_ik_user_review.json']
    return [path for path in paths if path.is_file()]


def run_direct_ik(video, output_dir, pose_cache, package, predictions,
                  height_cm, ratios, px_per_cm, single_panel_confirmed,
                  enabled=True, execute=True):
    """Automatically prepare and run compatible moves without the manual widget form.

    The original model package and its held-out validation remain unchanged.
    Missing metric geometry and unsupported schedules are reported, never fabricated.
    """
    from .motion_pipeline import run_motion_analysis, current_motion_outputs

    root, predictions = Path(output_dir), Path(predictions)
    prediction, contacts = load_review_bundle(video, root, pose_cache, package, predictions)
    pose_path = root / 'tables/pose_coordinates.csv'
    alignment_path = root / 'state/alignment_cache.npz'
    with np.load(alignment_path, allow_pickle=False) as cached:
        settings = json.loads(str(cached['settings'].item()))
    transforms = load_alignment(alignment_path, video, pose_path, settings)
    if transforms is None:
        raise ValueError('The pose/alignment inputs changed. Rerun Section 5.')
    poses = pd.read_csv(pose_path)
    body = dict(height_cm=height_cm, ratios=dict(ratios), px_per_cm=px_per_cm)
    rows = build_motion_candidates(prediction, contacts, body, set(poses.frame),
                                   {f for f, v in transforms.items() if v[2] == 'ok'})
    provenance = {
        'prediction_sha256': file_sha256(predictions / 'automatic_contacts.json'),
        'contact_table_sha256': file_sha256(predictions / 'limb_contacts.csv'),
        'pose_sha256': file_sha256(pose_path), 'alignment_sha256': file_sha256(alignment_path),
        'bridge_sha256': file_sha256(__file__),
    }
    review_file = root / 'state/automatic_ik_user_review.json'
    review = json.loads(review_file.read_text()) if review_file.exists() else None
    reviewed_current_prediction = bool(review and
        review.get('prediction_sha256') == provenance['prediction_sha256'] and
        review.get('contact_table_sha256') == provenance['contact_table_sha256'])
    blockers = geometry_blockers(px_per_cm, single_panel_confirmed)
    plan = dict(schema_version=1, enabled=bool(enabled),
        mode='experimental_direct_predictions', execute_requested=bool(execute),
        statistical_gate_bypassed=bool(enabled),
        current_prediction_user_accepted=reviewed_current_prediction,
        review_note=review if reviewed_current_prediction else None,
        body=body, single_panel_confirmed=bool(single_panel_confirmed),
        geometry_blockers=blockers, provenance=provenance, moves=rows,
        interpretation='Automatic kinematic candidates, not validated climbing-success predictions.')
    total_frames = int(contacts.frame.max()) + 1
    for row in rows:
        if not enabled:
            row['status'] = 'disabled'
        elif not row['solver_compatible']:
            row['status'] = 'unsupported_contact_schedule'
        elif blockers:
            row['status'] = 'pending_geometry'
        elif not execute:
            row['status'] = 'ready'
        else:
            destination = root / 'automatic/ik' / f"move_{row['move_index']:03d}"
            annotation = row['annotation']
            annotation['automatic_evidence'].update(provenance)
            annotation['automatic_evidence']['current_prediction_user_accepted'] = reviewed_current_prediction
            for name in ('tables/pose_coordinates.csv', 'state/reviewed_hold_map.npz', 'state/alignment_cache.npz'):
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, target)
            save_annotations(destination / 'state/endpoint_annotations.json', annotation, video,
                             prediction['candidate_run_id'], prediction['selected_hold_ids'], total_frames)
            try:
                try:
                    paths = current_motion_outputs(video, destination)
                except (ValueError, FileNotFoundError):
                    paths = []
                if not paths:
                    run_motion_analysis(video, destination)
                row['status'] = 'generated'
                row['output_dir'] = str(destination)
                row['video_file'] = str(destination / 'videos/motion_ik_h264.mp4')
            except (ValueError, RuntimeError) as error:
                row['status'] = 'solver_failed'
                row['reasons'].append(str(error))
    plan['proposal_count'] = len(rows)
    plan['solver_compatible_count'] = sum(row['solver_compatible'] for row in rows)
    plan['generated_count'] = sum(row['status'] == 'generated' for row in rows)
    path = root / 'state/direct_ik_plan.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    return plan
