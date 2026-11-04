"""Conservative source-image ground evidence for the Trial1 2D experiment.

Ground polygons and visibility/stability can propose candidates, NEVER support
forces. Only explicitly reviewed intervals create conditional located support.
Image registration is used to share a drawing coordinate system with the wall;
it is NOT a floor-plane metric calibration. Raw model predictions are retained.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from .pose_adapter import to_frame0, project_plane
from .project_io import file_sha256, load_alignment
from .trial1_sequence import TRIAL1_VIDEO_SHA256


def classify_ground(review, visible, inside_region, stable):
    if review == 'observed_ground':
        return 'observed_ground' if visible else 'unknown'
    if review == 'off_ground':
        return 'off_ground'
    if not visible:
        return 'unknown'
    return 'ground_candidate' if inside_region and stable else 'unknown'


def validate_annotation(a, video_sha, frame_count):
    if a['schema_version'] != 1 or a['video_sha256'] != video_sha or video_sha != TRIAL1_VIDEO_SHA256:
        raise ValueError('Ground annotations belong to this Trial1 video only.')
    if a['frame_count'] != frame_count or a['image_size_wh'] != [720, 1280]:
        raise ValueError('Ground annotation dimensions do not match Trial1.')
    if a['reviewer'] != 'assistant_visual_review' or a['user_confirmed'] is not False:
        raise ValueError('This experiment must preserve the actual review provenance.')
    if a['normal_model_xy'] != [0, 1] or a['geometry_model'] != 'projected_image_approximation':
        raise ValueError('Only explicit image-up projected ground geometry is implemented.')
    if not np.isfinite(a['sole_offset_height_ratio']) or not 0 < a['sole_offset_height_ratio'] < .1:
        raise ValueError('Invalid assumed ankle-to-sole offset.')
    if not 0 < a['ground_fit_weight'] <= 1:
        raise ValueError('Ground fit weight must lie in (0, 1].')
    for limb in ('left_foot', 'right_foot'):
        occupied = set()
        for r in a['intervals'][limb]:
            start, end = r['start'], r['end']
            if not (type(start) is int and type(end) is int and 0 <= start <= end < frame_count):
                raise ValueError('Invalid reviewed ground interval.')
            frames = set(range(start, end + 1))
            if occupied & frames or r['state'] not in ('observed_ground', 'off_ground'):
                raise ValueError('Overlapping or invalid ground evidence.')
            occupied |= frames
            if r['state'] == 'observed_ground' and not r.get('support_id', '').startswith('ground_'):
                raise ValueError('Observed ground requires an independent support ID.')
    keys = a['floor_region_keyframes']
    if len(keys) < 2 or any(keys[i]['frame'] >= keys[i+1]['frame'] for i in range(len(keys)-1)):
        raise ValueError('Floor image keyframes must be ordered.')
    for k in keys:
        poly = np.asarray(k['polygon_xy'], float)
        if poly.shape != (4, 2) or not np.isfinite(poly).all() or not 0 <= k['frame'] < frame_count:
            raise ValueError('Invalid floor image polygon.')


def overlay_ground(schedule, evidence, height_cm, annotation):
    """Add independent evidence without changing any decoded model label.

    A conflict with a decoded hold is left unresolved, not silently overwritten.
    Ground geometry is always conditional, even when contact is visually clear.
    """
    if not np.isfinite(height_cm) or height_cm <= 0:
        raise ValueError('Target height must be positive.')
    s = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in schedule.items()}
    shape = s['known'].shape
    s['source_confidence'] = s['confidence'].copy()
    s['source_known'] = s['known'].copy()
    s['support_ids'] = np.where(s['known'], s['labels'], '').astype('<U64')
    s['contact_type'] = np.where(s['known'], 'hold', 'unknown').astype('<U32')
    s['contact_offsets_cm'] = np.zeros((*shape, 2))
    s['fit_weight'] = s['confidence'].copy()
    s['ground_evidence'] = np.full(shape, 'not_reviewed', dtype='<U32')
    s['ground_conflict'] = np.zeros(shape, bool)
    s['ground_timing_uncertain'] = np.zeros(shape, bool)
    for row in evidence:
        t, i = row['frame'], row['limb_index']
        state = row['state']; s['ground_evidence'][t, i] = state
        if state != 'observed_ground':
            continue
        if s['known'][t, i] or str(s['labels'][t, i]) not in ('free', 'unknown'):
            s['ground_conflict'][t, i] = True
            continue
        target = np.asarray(row['contact_point_cm'], float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('Observed ground needs a finite located sole point.')
        s['known'][t, i] = True
        s['supported'][t, i] = False  # No independent metric floor calibration.
        s['targets'][t, i] = target
        s['states'][t, i] = 'ground_geometry_uncertain'
        s['contact_type'][t, i] = 'ground'
        s['support_ids'][t, i] = row['support_id']
        s['contact_offsets_cm'][t, i, 1] = -height_cm * annotation['sole_offset_height_ratio']
        s['fit_weight'][t, i] = annotation['ground_fit_weight']
    # A last observed contact -> ambiguous/occluded frame is a support cut-off,
    # not a claim to know the true first airborne frame.
    for t in range(1, len(s['frames'])):
        for i in (2, 3):
            if s['contact_type'][t-1, i] == 'ground' and s['support_ids'][t, i] != s['support_ids'][t-1, i]:
                s['ground_timing_uncertain'][t, i] = s['ground_evidence'][t, i] != 'off_ground'
    return s


def load_ground_inputs(project, inputs, height_cm, annotation_path=None):
    project = Path(project)
    path = Path(annotation_path) if annotation_path else project/'data/trial1_ground_support.json'
    a = json.loads(path.read_text())
    validate_annotation(a, file_sha256(project/'videos/trial1.mp4'), len(inputs['schedule']['frames']))
    pose = pd.read_csv(project/'outputs/tables/pose_coordinates.csv').set_index('frame')
    with np.load(project/'outputs/state/alignment_cache.npz', allow_pickle=False) as z:
        settings = json.loads(str(z['settings'].item()))
    align = load_alignment(project/'outputs/state/alignment_cache.npz', project/'videos/trial1.mp4',
                           project/'outputs/tables/pose_coordinates.csv', settings)
    if align is None:
        raise ValueError('Ground image registration is stale.')
    keys = a['floor_region_keyframes']; times = [k['frame'] for k in keys]
    polygons = np.asarray([k['polygon_xy'] for k in keys], float)
    rows = []
    for i, limb in ((2, 'left_foot'), (3, 'right_foot')):
        heel = pose[[limb.replace('foot', 'heel') + '_x_px', limb.replace('foot', 'heel') + '_y_px']].to_numpy()
        toe = pose[[limb + '_x_px', limb + '_y_px']].to_numpy()
        # Shoe-floor point proxy, not a measured center of pressure or sole mesh.
        points = np.where((heel[:, 1] > toe[:, 1])[:, None], heel, toe)
        visible = np.minimum(pose[limb+'_visibility'], pose[limb.replace('foot','heel')+'_visibility']).to_numpy() >= .5
        for t in range(len(pose)):
            review = next((r for r in a['intervals'][limb] if r['start'] <= t <= r['end']), None)
            in_range = times[0] <= t <= times[-1]
            polygon = np.array([[np.interp(t, times, polygons[:, k, j]) for j in range(2)] for k in range(4)])
            inside = bool(in_range and cv2.pointPolygonTest(polygon.astype(np.float32), tuple(map(float, points[t])), False) >= 0)
            # Three-frame stability is proposal evidence only. No camera-plane
            # assumption can turn it into a confirmed support observation.
            stable = t >= 2 and np.max(np.linalg.norm(np.diff(points[t-2:t+1], axis=0), axis=1)) < 5.
            state = classify_ground(review['state'] if review else None, bool(visible[t]), inside, stable)
            ref_xy = to_frame0(points[t:t+1], align, t)
            model_xy = project_plane(ref_xy, inputs['geometry']['image_to_wall_cm'])[0]
            rows.append(dict(frame=t, limb=limb, limb_index=i, state=state,
                reviewed_state=review['state'] if review else 'not_reviewed',
                support_id=review.get('support_id','') if review else '',
                visible=bool(visible[t]), inside_floor_region=inside, stable=bool(stable),
                sole_x_px=float(points[t,0]), sole_y_px=float(points[t,1]),
                contact_point_cm=model_xy.tolist(),
                evidence_source='assistant_visual_review' if review else 'geometric_temporal_candidate_only',
                user_confirmed=False, location_source='lower_of_observed_heel_and_toe_image_proxy',
                geometry_status='uncalibrated_projected_2D', timing_note=review.get('note','') if review else 'Unreviewed'))
    s = overlay_ground(inputs['schedule'], rows, height_cm, a)
    return {**inputs, 'schedule': s, 'ground_annotation': a, 'ground_rows': rows,
        'render_assumptions':'CONDITIONAL GROUND + UNIFORM SCALE | assumed mass {mass:g} kg | image-up normal; no depth',
        'render_contact_legend':'Ground: visually observed interval + approximate sole point. Conditional only; candidates have no ground force.'}
