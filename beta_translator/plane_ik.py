"""Verified single-plane geometry for the existing 2D motion solver."""
import json
from pathlib import Path

import numpy as np

from .project_io import file_sha256, load_reviewed_holds, validate_reference_alignment
from .pose_adapter import checked_plane_transform
from .portable_paths import artifact_paths, project_path


def load_ik_plane_geometry(video, root, calibration_path):
    from .trial_pipeline import valid_cache
    from . import plane_calibration
    root, calibration_path = Path(root), Path(calibration_path)
    manifest_path = calibration_path.parent/'plane_calibration_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    inputs = dict(video_sha256=file_sha256(video),
        observation_sha256=file_sha256(project_path(calibration['observation_file'], calibration_path)),
        hold_map_sha256=file_sha256(root/'state/reviewed_hold_map.npz'),
        pose_sha256=file_sha256(root/'automatic/wholebody.npz'),
        module_sha256=file_sha256(plane_calibration.__file__))
    if (valid_cache(manifest_path, inputs) is None or
            artifact_paths(manifest_path, manifest).get(calibration_path.resolve()) != file_sha256(calibration_path)):
        raise ValueError('Plane calibration is stale; rerun the grid calibration.')
    _, _, run_id, selected = load_reviewed_holds(root/'state/reviewed_hold_map.npz', video)
    reference = validate_reference_alignment(root/'state/reviewed_hold_map.npz', root/'state/alignment_cache.npz')
    if (calibration['video_sha256'] != inputs['video_sha256'] or calibration['candidate_run_id'] != run_id
            or calibration['reference_frame_number'] != reference or calibration['status'] != 'plane_homography_fitted'):
        raise ValueError('Plane calibration does not match the current video, map, and alignment.')
    matrix = checked_plane_transform(calibration['image_to_wall_cm'])
    supported = [h['hold_id'] for h in calibration['selected_hold_positions']
                 if h['inside_plane'] and h['inside_grid_hull'] and h['hold_id'] in selected]
    return dict(mode='calibrated_wall_plane', plane_id=calibration['plane_id'],
        image_to_wall_cm=matrix.tolist(), video_sha256=inputs['video_sha256'],
        hold_map_sha256=inputs['hold_map_sha256'], candidate_run_id=run_id,
        reference_frame_number=reference, calibration_sha256=file_sha256(calibration_path),
        supported_hold_ids=supported,
        hold_positions_cm={h['hold_id']:h['wall_xy_cm'] for h in calibration['selected_hold_positions'] if h['inside_plane']},
        interpretation='2D projection onto the calibrated wall; body and hold depth are omitted.')


def plane_contact_blockers(annotation, geometry):
    used = {hold for phase in annotation['contacts'].values() for hold in phase.values()}
    missing = sorted(used-set(geometry['supported_hold_ids']))
    return [f'Contacts outside calibrated hole coverage: {", ".join(missing)}.'] if missing else []


def contact_span_blockers(annotation, geometry, body, tolerance_cm=.1):
    """Conservative reach bound; failure applies to this body model and contact plan."""
    from itertools import combinations
    from .inverse_kinematics import make_body, forward_kinematics, LIMBS
    lengths = make_body(body['height_cm'], body['ratios'])
    roots = forward_kinematics(np.zeros(11), lengths).limbs
    reach = {limb: lengths['upper_arm']+lengths['forearm'] if limb.endswith('hand')
             else lengths['thigh']+lengths['shank'] for limb in LIMBS}
    positions = geometry['hold_positions_cm']
    reasons = []
    for phase, contacts in annotation['contacts'].items():
        for a,b in combinations(LIMBS,2):
            distance = np.linalg.norm(np.array(positions[contacts[a]])-positions[contacts[b]])
            bound = reach[a]+reach[b]+np.linalg.norm(roots[a].root-roots[b].root)
            if distance > bound+2*tolerance_cm:
                reasons.append(f'{phase}: {a}/{b} targets are {distance:.2f} cm apart; '
                               f'this body model can span at most {bound:.2f} cm.')
    return reasons


def validate_saved_plane_geometry(annotation, video, root, reference):
    geometry = annotation.get('plane_geometry')
    if geometry is None:
        return None
    root = Path(root)
    snapshot = root/'state/plane_calibration.json'
    if (file_sha256(snapshot) != geometry['calibration_sha256']
            or file_sha256(root/'state/reviewed_hold_map.npz') != geometry['hold_map_sha256']
            or file_sha256(video) != geometry['video_sha256']
            or geometry['candidate_run_id'] != annotation['candidate_run_id']
            or geometry['reference_frame_number'] != reference):
        raise ValueError('Motion plane geometry is stale or belongs to another reference.')
    calibration = json.loads(snapshot.read_text())
    expected = {h['hold_id'] for h in calibration['selected_hold_positions']
                if h['inside_plane'] and h['inside_grid_hull']}
    if (not set(geometry['supported_hold_ids']).issubset(expected)
            or calibration['video_sha256'] != geometry['video_sha256']
            or calibration['candidate_run_id'] != geometry['candidate_run_id']
            or calibration['reference_frame_number'] != reference
            or not np.array_equal(calibration['image_to_wall_cm'], geometry['image_to_wall_cm'])):
        raise ValueError('Motion plane geometry differs from its saved calibration.')
    blockers = plane_contact_blockers(annotation, geometry)
    if blockers:
        raise ValueError(' '.join(blockers))
    return checked_plane_transform(geometry['image_to_wall_cm'])
