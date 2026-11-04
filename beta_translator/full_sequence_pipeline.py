"""Read verified Trial 3 caches and publish a separate diagnostic full-video run."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import tempfile
import textwrap
import uuid

import cv2
import numpy as np
import pandas as pd

from . import full_sequence
from .contact_review import load_review_bundle
from .full_sequence import build_schedule, solve_full_sequence, diagnose_sequence, DEFAULT_SETTINGS
from .inverse_kinematics import LIMBS, RATIOS, POSE_VECTOR_NAMES, make_body
from .plane_ik import load_ik_plane_geometry
from .pose_adapter import LIMB_LANDMARKS, to_frame0, pose_row_to_vector, angle_visibility_weights, project_plane
from .project_io import file_sha256, load_alignment, load_reviewed_holds, convert_video


def _json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def input_provenance(project):
    project = Path(project).resolve()
    relative = ['videos/trial3.mp4', 'models/contact_v1/deployment.json',
        'outputs/trial3/state/reviewed_hold_map.npz', 'outputs/trial3/state/alignment_cache.npz',
        'outputs/trial3/state/white_wall_calibration.json', 'outputs/trial3/tables/pose_coordinates.csv',
        'outputs/trial3/tables/source_frame_map.csv', 'outputs/trial3/automatic/wholebody.npz',
        'outputs/trial3/automatic/predictions/automatic_contacts.json',
        'outputs/trial3/automatic/predictions/limb_contacts.csv',
        'outputs/trial3/automatic/predictions/contact_events.csv',
        'outputs/trial3/state/identity_guard.npz', 'data/trial3_wall_grid.json',
        'outputs/trial3/state/white_wall_grid_points.json',
        'outputs/trial3/state/plane_calibration_manifest.json',
        'models/contact_v1/model_1.pt', 'models/contact_v1/model_2.pt', 'models/contact_v1/model_3.pt']
    files = {name:file_sha256(project/name) for name in relative}
    for name in ['inverse_kinematics.py', 'motion_ik.py', 'pose_adapter.py', 'plane_ik.py',
                 'plane_calibration.py', 'contact_review.py', 'identity_tracking.py', 'project_io.py']:
        files['beta_translator/'+name] = file_sha256(project/'beta_translator'/name)
    files['beta_translator/full_sequence.py'] = file_sha256(full_sequence.__file__)
    files['beta_translator/contact_geometry.py'] = file_sha256(Path(__file__).with_name('contact_geometry.py'))
    files['beta_translator/full_sequence_pipeline.py'] = file_sha256(__file__)
    return files


def load_full_sequence_inputs(project):
    project = Path(project).resolve()
    root, video = project/'outputs/trial3', project/'videos/trial3.mp4'
    prediction, contacts = load_review_bundle(video, root, root/'automatic/wholebody.npz',
                                              project/'models/contact_v1', root/'automatic/predictions')
    geometry = load_ik_plane_geometry(video, root, root/'state/white_wall_calibration.json')
    calibration = json.loads((root/'state/white_wall_calibration.json').read_text())
    records, masks, _, selected = load_reviewed_holds(root/'state/reviewed_hold_map.npz', video)
    with np.load(root/'state/reviewed_hold_map.npz', allow_pickle=False) as z:
        background = z['reference_frame'].copy()
    with np.load(root/'state/alignment_cache.npz', allow_pickle=False) as z:
        alignment_settings = json.loads(str(z['settings'].item()))
    transforms = load_alignment(root/'state/alignment_cache.npz', video,
                                 root/'tables/pose_coordinates.csv', alignment_settings)
    if transforms is None:
        raise ValueError('Pose/alignment provenance mismatch.')
    cap = cv2.VideoCapture(str(video))
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if count != 503 or not np.isclose(fps, 30):
        raise ValueError('This Trial 3 entry point requires its 503-frame, 30 FPS analysis video.')
    frames = np.arange(count)
    rows = pd.read_csv(root/'tables/pose_coordinates.csv').set_index('frame')
    if not rows.index.is_unique:
        raise ValueError('Duplicate pose frames.')
    if set(rows.index) != set(frames):
        raise ValueError('Every source frame needs a pose observation; no hidden display interpolation.')
    schedule = build_schedule(contacts, frames, geometry, prediction['accept_threshold'])
    grid = json.loads((root/'state/white_wall_grid_points.json').read_text())
    hull = cv2.convexHull(np.asarray(grid['image_xy'], np.float32))
    polygon = np.asarray(calibration['plane_polygon_xy'], np.float32)
    names = list(dict.fromkeys(n for chain in LIMB_LANDMARKS.values() for n in chain))
    matrix = np.asarray(geometry['image_to_wall_cm'])
    references, weights, endpoints, outside, aligned = [], [], [], [], []
    for frame in frames:
        if frame not in transforms or transforms[frame][2] != 'ok':
            raise ValueError(f'Frame {frame}: wall alignment is not valid.')
        row = rows.loc[frame].copy()
        if ('pose_source' in row and not str(row.pose_source).startswith('observed')):
            raise ValueError(f'Frame {frame}: an interpolated pose cannot become evidence.')
        points = to_frame0([[row[n+'_x_px'], row[n+'_y_px']] for n in names], transforms, int(frame))
        for name, point in zip(names, points):
            row[name+'_x_px'], row[name+'_y_px'] = point
        references.append(pose_row_to_vector(row, None, None, matrix))
        weights.append(angle_visibility_weights(row))
        ends = [[row[LIMB_LANDMARKS[l][-1]+'_x_px'], row[LIMB_LANDMARKS[l][-1]+'_y_px']] for l in LIMBS]
        endpoints.append(project_plane(ends, matrix))
        outside.append(any(cv2.pointPolygonTest(hull, tuple(map(float, p)), False) < 0 or
                           cv2.pointPolygonTest(polygon, tuple(map(float, p)), False) < 0 for p in points))
        aligned.append(points)
    return dict(reference=np.asarray(references), visibility=np.asarray(weights),
        reference_endpoints=np.asarray(endpoints), reference_outside=np.asarray(outside),
        aligned_reference=np.asarray(aligned), schedule=schedule, fps=float(fps),
        geometry=geometry, calibration=calibration, background=background,
        records=records, masks=masks, selected=selected, prediction=prediction)


def _intervals(diagnostics, fps):
    rows = []
    for issue, mask in diagnostics['flags'].items():
        changes = np.diff(np.r_[False, mask, False].astype(int))
        for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            rows.append(dict(issue=issue, start_frame=int(start), end_frame=int(stop-1),
                             start_sec=float(start/fps), end_exclusive_sec=float(stop/fps)))
    return sorted(rows, key=lambda r:(r['start_frame'], r['issue']))


def _tables(result, diagnostic, fps):
    poses, limbs = [], []
    s = result['schedule']
    for t, frame in enumerate(result['frames']):
        row = dict(frame=int(frame), time_sec=float(frame/fps), status=diagnostic['status'][t],
            issues=diagnostic['issues'][t], max_contact_error_cm=diagnostic['max_contact_error_cm'][t],
            span_excess_cm=diagnostic['span_excess_cm'][t], span_pairs=';'.join(diagnostic['span_pairs'][t]),
            angle_step_deg=diagnostic['angle_step_deg'][t], hip_step_cm=diagnostic['hip_step_cm'][t])
        row.update(zip(POSE_VECTOR_NAMES, result['poses'][t]))
        row.update({name:bool(values[t]) for name,values in diagnostic['flags'].items()})
        poses.append(row)
        for i, limb in enumerate(LIMBS):
            limbs.append(dict(frame=int(frame), limb=limb, decoded_label=s['labels'][t,i],
                support_interpretation=s['states'][t,i], confidence=s['confidence'][t,i],
                low_confidence=bool(s['low_confidence'][t,i]),
                contact_target_used=bool(s['known'][t,i]), inside_calibrated_coverage=bool(s['supported'][t,i]),
                target_x_cm=s['targets'][t,i,0] if s['known'][t,i] else np.nan,
                target_y_cm=s['targets'][t,i,1] if s['known'][t,i] else np.nan,
                fitted_x_cm=result['joints'][t,i,2,0], fitted_y_cm=result['joints'][t,i,2,1],
                contact_error_cm=diagnostic['contact_errors_cm'][t,i]))
    return pd.DataFrame(poses), pd.DataFrame(limbs)


ISSUE_TEXT = {
    'contact_residual': 'Contact target not met within 0.1 cm.',
    'large_contact_residual': 'At least one contact misses by over 5 cm.',
    'span_conflict': 'Decoded contacts exceed this body span.',
    'hold_extrapolation': 'H053: outside checked grid coverage.',
    'unknown_contact': 'A contact has no calibrated target.',
    'unresolved_foot_support': 'Free foot: ground/wall/air unresolved.',
    'wall_contact_unlocated': 'Wall contact has no measured location.',
    'low_contact_confidence': 'At least one model confidence < 0.80.',
    'low_pose_visibility': 'Reference body landmark visibility < 0.50.',
    'reference_extrapolation': 'Some reference joints are outside grid/panel.',
    'rapid_pose_change': 'Large frame-to-frame pose change.',
    'optimizer_unconverged': 'Global solver did not converge.',
}
COLORS = dict(green=(65,145,30), amber=(0,150,220), red=(45,50,210),
              purple=(180,65,160), ink=(45,45,45), gray=(135,135,135))


def render_full_sequence(inputs, result, diagnostic, destination):
    """A target-only video, with a stable full-wall view and per-frame diagnostics."""
    destination = Path(destination)
    background = inputs['background'].copy()
    matrix = np.asarray(inputs['geometry']['image_to_wall_cm'])
    inverse = np.linalg.inv(matrix)
    pixels = project_plane(result['joints'].reshape(-1, 2), inverse).reshape(-1, 4, 3, 2)
    h, w = background.shape[:2]
    # Include all fitted joints at all times, with a fixed camera and padding.
    lo = np.minimum([0,0], np.floor(pixels.min(axis=(0,1,2))-20)).astype(int)
    hi = np.maximum([w,h], np.ceil(pixels.max(axis=(0,1,2))+20)).astype(int)
    if np.any(hi-lo > 8000):
        raise ValueError('Projection diverged; cannot render a meaningful full-wall view.')
    for record in inputs['records']:
        hold = record['hold_id']
        if hold not in inputs['selected']:
            continue
        contours, _ = cv2.findContours(inputs['masks'][hold].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = COLORS['purple'] if hold not in inputs['geometry']['supported_hold_ids'] else (110,110,110)
        cv2.drawContours(background, contours, -1, color, 1)
        cv2.putText(background, hold, (int(record['cx_px'])+8,int(record['cy_px'])-7), 0, .53, color, 2, cv2.LINE_AA)
    padded = np.full((hi[1]-lo[1], hi[0]-lo[0], 3), 235, np.uint8)
    padded[-lo[1]:h-lo[1], -lo[0]:w-lo[0]] = background
    panel_w = 600
    panel_h = int(round(padded.shape[0]/padded.shape[1]*panel_w/2))*2
    header, footer, sidebar = 100, 96, 424
    width, height = panel_w+sidebar, panel_h+header+footer
    raw = destination/'encoding.mp4'
    video = destination/f'trial3_target_{result["body"]["height_cm"]:g}cm_full_sequence.mp4'
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), inputs['fps'], (width,height))
    if not writer.isOpened():
        raise RuntimeError('Could not open video writer.')
    snapshots = []
    selected_frames = {0, 92, 147, 250, 314, 428, 502}
    status_colors = {'constraint_warning':COLORS['red'], 'uncertain':COLORS['amber'],
                     'within_numeric_tolerance':COLORS['green']}
    def put(canvas, text, x, y, scale=.50, color=None, thickness=1):
        cv2.putText(canvas, text, (x,y), 0, scale, color or COLORS['ink'], thickness, cv2.LINE_AA)
    try:
        for t, frame in enumerate(result['frames']):
            wall = padded.copy()
            joints = np.rint(pixels[t]-lo).astype(int)
            cv2.polylines(wall, [joints[[0,1,3,2,0],0]], False, (85,115,100), 4, cv2.LINE_AA)
            s = result['schedule']
            for i, limb in enumerate(LIMBS):
                if s['known'][t,i] and not s['supported'][t,i]:
                    color = COLORS['purple']
                elif s['known'][t,i] and diagnostic['contact_errors_cm'][t,i] > result['settings']['contact_tolerance_cm']:
                    color = COLORS['red']
                elif not s['known'][t,i] or s['low_confidence'][t,i]:
                    color = COLORS['amber']
                else:
                    color = COLORS['green']
                cv2.polylines(wall, [joints[i]], False, color, 4, cv2.LINE_AA)
                for xy in joints[i]:
                    cv2.circle(wall, tuple(xy), 5, color, -1, cv2.LINE_AA)
                if s['known'][t,i]:
                    target = project_plane(s['targets'][t,i][None], inverse)[0]-lo
                    cv2.drawMarker(wall, tuple(np.rint(target).astype(int)), color, cv2.MARKER_CROSS, 15, 2)
                    if diagnostic['contact_errors_cm'][t,i] > result['settings']['contact_tolerance_cm']:
                        for u in np.arange(0, 1, .15):
                            a = joints[i,2]*(1-u)+target*u
                            b = joints[i,2]*(1-min(u+.07,1))+target*min(u+.07,1)
                            cv2.line(wall, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), color, 2, cv2.LINE_AA)
            canvas = np.full((height,width,3), 250, np.uint8)
            canvas[header:header+panel_h,:panel_w] = cv2.resize(wall,(panel_w,panel_h),interpolation=cv2.INTER_AREA)
            put(canvas, f'{result["body"]["height_cm"]:g} cm | Full-sequence 2D body adaptation', 18, 29, .72, thickness=2)
            put(canvas, f'Frame {frame:03d} / 502 | {frame/inputs["fps"]:.2f} s | 1.00x | fixed body lengths', 18, 55, .54)
            color = status_colors[diagnostic['status'][t]]
            put(canvas, f'{diagnostic["status"][t].upper()} | max contact gap {diagnostic["max_contact_error_cm"][t]:.2f} cm', 18, 82, .54, color, 2)
            x = panel_w+18
            put(canvas, 'PREDICTED CONTACTS', x, header+25, .62, thickness=2)
            for i, limb in enumerate(LIMBS):
                y = header+61+i*93
                label = s['labels'][t,i]
                put(canvas, f'{limb.replace("_"," ").title()} : {label}', x, y, .57, thickness=2)
                put(canvas, f'Model confidence: {s["confidence"][t,i]:.2f}', x, y+23, .48)
                if s['known'][t,i]:
                    error = diagnostic['contact_errors_cm'][t,i]
                    coverage = 'in grid' if s['supported'][t,i] else 'EXTRAPOLATED'
                    put(canvas, f'Gap: {error:.2f} cm | {coverage}', x, y+46, .48,
                        COLORS['purple'] if not s['supported'][t,i] else COLORS['red'] if error>result['settings']['contact_tolerance_cm'] else COLORS['green'])
                else:
                    text = 'Ground / wall / air unresolved' if i>=2 else 'No fixed target; reference-guided'
                    put(canvas, text, x, y+46, .47, COLORS['amber'])
            y = header+460
            put(canvas, 'FRAME DIAGNOSTICS', x, y, .59, thickness=2)
            active = [k for k,v in diagnostic['flags'].items() if v[t]]
            for key in active:
                for line in textwrap.wrap(ISSUE_TEXT[key], 43):
                    y += 22
                    put(canvas, line, x, y, .45, COLORS['red'] if key in ('span_conflict','large_contact_residual','optimizer_unconverged') else COLORS['ink'])
            if not active:
                y += 25
                put(canvas, 'Within configured numeric tolerance.', x, y, .45)
            y = max(y+38, header+800)
            for text, color in [('Red: contact mismatch',COLORS['red']),
                                ('Purple: outside checked grid',COLORS['purple']),
                                ('Amber: free / uncertain',COLORS['amber']),
                                ('Green: gap <= 0.1 cm',COLORS['green'])]:
                put(canvas, text, x, y, .48, color); y += 25
            base = header+panel_h
            for j,status in enumerate(diagnostic['status']):
                a, b = round(j*width/len(result['frames'])), round((j+1)*width/len(result['frames']))
                cv2.rectangle(canvas,(a,base+4),(b,base+19),status_colors[status],-1)
            cursor = round(t*(width-1)/(len(result['frames'])-1))
            cv2.line(canvas,(cursor,base+1),(cursor,base+24),(0,0,0),3)
            put(canvas, 'One global fit of frames 0-502. Dashed lines show missed contact targets.', 16, base+47, .48)
            put(canvas, 'Predicted contacts and projected 2D body only. No force, balance, collision or success model.', 16, base+73, .47)
            writer.write(canvas)
            if int(frame) in selected_frames:
                small = cv2.resize(canvas, (512, height//2))
                snapshots.append(small)
                if int(frame) == 314:
                    cv2.imwrite(str(destination/'poster.png'), canvas)
    finally:
        writer.release()
    convert_video(raw, video)
    cap = cv2.VideoCapture(str(video))
    actual, fps = int(cap.get(7)), cap.get(5)
    cap.release()
    if actual != len(result['frames']) or not np.isclose(fps, inputs['fps']):
        raise RuntimeError('Encoded video does not match all source frames at native speed.')
    # The encoding scratch stays inside TemporaryDirectory, never in the project.
    storyboard = np.hstack(snapshots)
    cv2.imwrite(str(destination/'storyboard.png'), storyboard)
    return [video, destination/'poster.png', destination/'storyboard.png']


def run_full_sequence(project, height_cm=180, ratios=None, settings=None, progress=print):
    """Create/reuse an immutable run directory; never replace earlier results."""
    project = Path(project).resolve()
    body = dict(height_cm=float(height_cm), ratios=dict(RATIOS if ratios is None else ratios))
    make_body(**body)
    video_name = f'trial3_target_{height_cm:g}cm_full_sequence.mp4'
    config = {**DEFAULT_SETTINGS, **(settings or {})}
    before = input_provenance(project)
    identity = dict(inputs=before, body=body, settings=config)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    destination = project/'outputs/trial3/full_sequence'/f'target_{height_cm:g}cm_{key}'
    if (destination/'summary.json').is_file():
        summary = json.loads((destination/'summary.json').read_text())
        if summary.get('provenance') == identity and all((destination/name).is_file() and
                file_sha256(destination/name) == digest for name,digest in summary['output_sha256'].items()):
            if progress: progress('Reused verified full-sequence outputs.')
            return {**summary, 'output_dir':str(destination), 'video_file':str(destination/video_name)}
    if destination.exists():
        destination = destination.with_name(destination.name+'_'+uuid.uuid4().hex[:8])
    inputs = load_full_sequence_inputs(project)
    if progress: progress('Verified all 503 frames, pose/contact caches, and wall calibration.')
    result = solve_full_sequence(inputs['reference'], inputs['visibility'], inputs['reference_endpoints'],
        inputs['schedule'], body, inputs['fps'], config, progress)
    diagnostic = diagnose_sequence(result, inputs['reference_outside'])
    poses, contacts = _tables(result, diagnostic, inputs['fps'])
    with tempfile.TemporaryDirectory(prefix='beta-full-sequence-') as tmp:
        stage = Path(tmp)
        poses.to_csv(stage/'frame_diagnostics.csv', index=False)
        contacts.to_csv(stage/'limb_diagnostics.csv', index=False)
        pd.DataFrame(_intervals(diagnostic, inputs['fps'])).to_csv(stage/'issue_intervals.csv', index=False)
        np.savez_compressed(stage/'trajectory.npz', frames=result['frames'], poses=result['poses'],
            joints_cm=result['joints'], reference_poses=result['reference'],
            visibility=result['visibility'], reference_endpoints_cm=result['reference_endpoints'],
            decoded_labels=result['schedule']['labels'], targets_cm=result['schedule']['targets'],
            active=result['schedule']['known'], in_coverage=result['schedule']['supported'],
            contact_errors_cm=diagnostic['contact_errors_cm'], confidence=result['schedule']['confidence'])
        if progress: progress('Rendering the complete target-body video and diagnostic tables.')
        files = render_full_sequence(inputs, result, diagnostic, stage)
        files += [stage/name for name in ['trajectory.npz','frame_diagnostics.csv','limb_diagnostics.csv','issue_intervals.csv']]
        summary = dict(schema_version=1, provenance=identity, video_name=video_name, frame_count=len(result['frames']),
            first_frame=int(result['frames'][0]), last_frame=int(result['frames'][-1]),
            fps=inputs['fps'], duration_seconds=len(result['frames'])/inputs['fps'],
            body=body, source_climber_height_cm=None, method=result['method'],
            optimizer=result['optimizer'], flag_counts=diagnostic['counts'],
            status_counts=dict(Counter(map(str, diagnostic['status']))),
            max_contact_error_cm=float(diagnostic['max_contact_error_cm'].max()),
            median_frame_max_contact_error_cm=float(np.median(diagnostic['max_contact_error_cm'])),
            max_segment_length_error_cm=diagnostic['max_segment_length_error_cm'],
            max_angle_step_deg=float(diagnostic['angle_step_deg'].max()),
            max_hip_step_cm=float(diagnostic['hip_step_cm'].max()),
            settings=config, candidate_run_id=inputs['geometry']['candidate_run_id'],
            reference_frame=inputs['geometry']['reference_frame_number'],
            model_threshold_coverage=inputs['prediction']['accepted_frame_fraction'],
            contact_source='Original decoded per-frame predictions; no reclassification or retraining.',
            floor_support='Not measured. Free feet remain ground/wall/air unresolved.',
            calibration_scope='Existing white-wall transform; H053 is a low-weight flagged extrapolation. No expanded calibration.',
            interpretation='Full-duration best-effort 2D adaptation. Warning frames are exported for diagnosis, not accepted IK or climbing success.',
            limitations=['Contact centers and timing are unreviewed model predictions.',
                'A joint global objective balances contact gaps, projected references, velocity and acceleration.',
                'The 0.1 cm diagnostic threshold is numerical, not measurement accuracy.',
                'Body lengths are fixed; infeasible contacts remain residuals instead of stretching limbs.',
                'Ground/wall/air support cannot be distinguished from free labels alone.',
                'No force, friction, balance, collision, grasp shape, depth, or alternative-beta planner.',
                'Future frames are used; this is an offline animation.'],
            output_sha256={f.name:file_sha256(f) for f in files})
        _json(stage/'summary.json', summary)
        files.append(stage/'summary.json')
        if input_provenance(project) != before:
            raise RuntimeError('Inputs changed during the fit; no outputs were installed.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(exist_ok=False)
        for f in files:
            (destination/f.name).write_bytes(f.read_bytes())
    if progress: progress(f'Full sequence saved: {destination}')
    return {**summary, 'output_dir':str(destination), 'video_file':str(destination/video_name)}
