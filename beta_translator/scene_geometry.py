"""Local scale and vanishing-point diagnostics; no inferred global metric camera."""

import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .inverse_kinematics import RATIOS, make_body
from .project_io import file_sha256, read_frame


def positive(value, name):
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive.')
    return float(value)


def image_points(value, image_size):
    points = np.asarray(value, dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError('Each segment needs two finite image points.')
    width, height = image_size
    if (points < 0).any() or (points >= [width, height]).any():
        raise ValueError('Segment endpoints lie outside the reference image.')
    if np.linalg.norm(points[1] - points[0]) < 2:
        raise ValueError('Segment endpoints must be distinct.')
    return points


def local_scale_samples(pairs, spacing_cm, image_size):
    """Directional, local pixel scales only; never average into a global IK scale."""
    spacing_cm = positive(spacing_cm, 'Hole spacing')
    rows = []
    seen = set()
    for pair in pairs:
        if pair['id'] in seen:
            raise ValueError('Duplicate hole-pair ID.')
        seen.add(pair['id'])
        intervals = pair['intervals']
        if type(intervals) is not int or intervals <= 0:
            raise ValueError('Count positive whole intervals, not the number of holes.')
        points = image_points(pair['endpoints_px'], image_size)
        pixels = float(np.linalg.norm(points[1] - points[0]))
        length = intervals * spacing_cm
        rows.append(dict(sample_id=pair['id'], panel_id=pair['panel_id'],
                         direction=pair['direction'], intervals=intervals,
                         assumed_length_cm=length, image_distance_px=pixels,
                         local_px_per_cm=pixels / length,
                         review_status=pair['review_status']))
    if not rows:
        raise ValueError('Select at least one adjacent-hole pair.')
    return rows


def parallel_line_vanishing_point(segments, image_size):
    """Intersect image lines known to be parallel in the scene, including infinity."""
    if len(segments) != 2:
        raise ValueError('Provide exactly two lines for one scene direction.')
    lines = []
    for segment in segments:
        points = image_points(segment, image_size)
        line = np.cross(np.r_[points[0], 1], np.r_[points[1], 1])
        lines.append(line / np.linalg.norm(line[:2]))
    point = np.cross(*lines)
    if np.linalg.norm(point) < 1e-9:
        raise ValueError('Coincident lines do not determine a vanishing point.')
    if abs(point[2]) < 1e-8:
        point /= np.linalg.norm(point[:2])
        return dict(homogeneous=point.tolist(), pixel_xy=None, at_infinity=True)
    point /= point[2]
    return dict(homogeneous=point.tolist(), pixel_xy=point[:2].tolist(), at_infinity=False)


def analyze_geometry(observations, spacing_cm, source_height_cm, target_height_cm,
                     target_ratios=None):
    source_height_cm = positive(source_height_cm, 'Source climber height')
    target_height_cm = positive(target_height_cm, 'Target user height')
    rows = local_scale_samples(observations['hole_pairs'], spacing_cm, observations['image_size'])
    floor = observations['floor_lines']
    floor_result = dict(status='needs_line_review', vanishing_point=None)
    if floor['relation'] == 'parallel_in_world' and floor['image_lines_reviewed'] is True:
        floor_result = dict(status='one_direction_estimated', vanishing_point=
            parallel_line_vanishing_point(floor['line_endpoints_px'], observations['image_size']))
    return dict(
        schema_version=1,
        assumptions=dict(hole_spacing_cm=float(spacing_cm), spacing_status='user_assumed_not_measured',
                         spacing_definition='Center-to-center distance for each marked adjacent-hole interval; grid angles are not assumed.',
                         source_climber_height_cm=source_height_cm, source_height_status='user_reported',
                         target_user_height_cm=target_height_cm),
        body_priors=dict(source_segments_cm=make_body(source_height_cm),
                         target_segments_cm=make_body(target_height_cm, target_ratios),
                         source_ratios=dict(RATIOS),
                         note='Segment lengths use model ratios, not measured limb lengths. Height does not calibrate the crouched image silhouette.'),
        local_scale_samples=rows, floor=floor_result,
        global_px_per_cm=None, wall_floor_angle_deg=None, floor_horizon=None,
        metric_panel_transform=None, ready_for_metric_ik=False,
        limitations=[
            'Local directional scales are not a whole-image pixel-to-centimeter conversion.',
            'One floor vanishing point does not determine the floor horizon or plane normal.',
            'Wall angle needs additional camera/plane constraints; no angle is inferred here.',
            'Hole selections are assistant-reviewed draft samples on original frame 450.',
            'A second independent floor direction and calibrated camera intrinsics can constrain the floor normal.',
            'Wall grid layout and plane-specific calibration are required before metric pose fitting.',
        ])


def validate_reference(observations, video_file, reference_frame, image_size):
    if observations['video_sha256'] != file_sha256(video_file):
        raise ValueError('Geometry points belong to another video.')
    if observations['reference_frame'] != reference_frame:
        raise ValueError('Geometry points belong to another reference frame; annotate new points.')
    if list(observations['image_size']) != list(image_size):
        raise ValueError('Geometry image dimensions changed.')


def geometry_markdown(result):
    a = result['assumptions']
    vp = result['floor']['vanishing_point']
    point = ('not estimated' if vp is None else 'at infinity' if vp['at_infinity']
             else '(' + ', '.join(f'{v:.1f}' for v in vp['pixel_xy']) + ') pixels')
    grouped = pd.DataFrame(result['local_scale_samples']).groupby(['panel_id', 'direction']).local_px_per_cm
    table = '\n'.join(f'| {panel} | {direction} | {values.median():.3f} | {len(values)} |'
                      for (panel, direction), values in grouped)
    return f'''### Inputs and their meaning

- Adjacent-hole center spacing: **{a['hole_spacing_cm']:g} cm**, a user assumption, not a Movement specification or measurement.
- Filmed climber height: **{a['source_climber_height_cm']:g} cm**, reported by the user.
- Target user height: **{a['target_user_height_cm']:g} cm**, kept separate for later retargeting.
- Default body ratios give model priors only; the climber's bent image height is not used as a ruler.

### Local scale from visible holes

The overlay marks the exact adjacent-hole pairs on the original video frame, before multi-frame compositing.
Each sample uses `pixel distance / (number of intervals * assumed spacing)`.
Row and column are image-grid directions, not world-horizontal or world-vertical axes.

| Wall region | Grid direction | Median local px/cm | Pairs |
|---|---|---:|---:|
{table}

Perspective makes these scales vary with position and direction. No single value is assigned to `PX_PER_CM`.
The spacing assumption does not establish a square grid or angles between grid directions.

### What the parallel floor lines tell us

The user identified F1 and F2 as parallel in the scene. Their image-line extensions give one
vanishing point: **{point}**, in frame-{result['reference_frame']} image coordinates.
This is a directional perspective constraint, not the floor horizon and not a wall-angle estimate.
Endpoint selection, lens distortion and curved mat seams affect the estimate.

To estimate a wall-to-floor angle, we still need the floor normal and the wall normal in the same
calibrated camera coordinates. For example: add a second floor-line family in another direction,
camera intrinsics, and known wall-grid geometry. A known vertical direction with camera calibration
is another way to constrain the floor normal. One known human height does not supply these by itself.

### Pipeline behavior

Section 4C saves assumptions, body priors, local measurements and the floor constraint. Section 8
shows them before the existing IK gate; Section 10 exports them. Contact predictions continue in
image coordinates. Metric IK remains pending until panel geometry and motion contacts are reviewed.
No wall angle, 3D wall reconstruction, physical success probability or contact-accuracy improvement is claimed.

Method references: [Single View Metrology](https://ora.ox.ac.uk/bookmarks/uuid%3A9b99df78-df75-49a9-a1b8-1b38e1b173a8),
[OpenCV planar homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html).
'''


def save_geometry_run(video_file, output_dir, observation_file, reference_frame,
                      spacing_cm, source_height_cm, target_height_cm, target_ratios=None):
    """Persist a reproducible diagnostic run without changing contact/IK inputs."""
    out = Path(output_dir)
    observation_file = Path(observation_file)
    observations = json.loads(observation_file.read_text())
    frame = read_frame(video_file, reference_frame)
    height, width = frame.shape[:2]
    validate_reference(observations, video_file, reference_frame, [width, height])
    result = analyze_geometry(observations, spacing_cm, source_height_cm, target_height_cm, target_ratios)
    result['reference_frame'] = reference_frame
    result['provenance'] = dict(video_sha256=observations['video_sha256'],
        observation_sha256=file_sha256(observation_file), module_sha256=file_sha256(__file__),
        frame_pixels_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
        pixel_source='Original decoded video frame, not the reconstructed wall composite.')
    result['notebook_markdown'] = geometry_markdown(result)
    for folder in ('state', 'tables', 'images', 'reports'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    state = out / 'state/geometry_calibration.json'
    state.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (out / 'state/geometry_observations.json').write_text(json.dumps(observations, indent=2) + '\n')
    pd.DataFrame(result['local_scale_samples']).to_csv(out / 'tables/geometry_local_scale.csv', index=False)

    overlay = frame.copy()
    colors = {'row': (0, 170, 255), 'column': (255, 170, 0)}
    for pair in observations['hole_pairs']:
        points = np.rint(pair['endpoints_px']).astype(int)
        color = colors[pair['direction']]
        cv2.line(overlay, tuple(points[0]), tuple(points[1]), color, 2)
        for point in points:
            cv2.circle(overlay, tuple(point), 5, color, 1)
    for i, segment in enumerate(observations['floor_lines']['line_endpoints_px'], 1):
        p, q = np.rint(segment).astype(int)
        cv2.line(overlay, tuple(p), tuple(q), (0, 190, 255), 3)
        cv2.putText(overlay, f'F{i}', tuple(p - [0, 15]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(overlay, f'F{i}', tuple(p - [0, 15]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 190, 255), 2)
    cv2.imwrite(str(out / 'images/geometry_reference_points.jpg'), overlay)

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    panels = sorted({p['panel_id'] for p in observations['hole_pairs']})
    fig = Figure(figsize=(12, 5)); FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(panels), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        pairs = [p for p in observations['hole_pairs'] if p['panel_id'] == panel]
        points = np.concatenate([p['endpoints_px'] for p in pairs])
        lo = np.floor(points.min(0) - 22).astype(int); hi = np.ceil(points.max(0) + 22).astype(int)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for p in pairs:
            xy = np.asarray(p['endpoints_px'])
            ax.plot(xy[:, 0], xy[:, 1], 'o-', ms=5, mfc='none', lw=1,
                    color='darkorange' if p['direction'] == 'row' else 'deepskyblue')
        ax.set(xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), title=panel, xlabel='Original-frame x (px)', ylabel='y (px)')
    fig.suptitle(f'Each marked interval = {spacing_cm:g} cm assumed | Orange: row; blue: column')
    fig.tight_layout(); fig.savefig(out / 'images/geometry_hole_details.png', dpi=160)

    fig = Figure(figsize=(12, 7)); FigureCanvasAgg(fig); ax = fig.subplots()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    vp = result['floor']['vanishing_point']
    for i, line in enumerate(observations['floor_lines']['line_endpoints_px'], 1):
        xy = np.asarray(line)
        ax.plot(xy[:, 0], xy[:, 1], lw=3, label=f'F{i}: user-confirmed parallel line')
        if vp is not None and not vp['at_infinity']:
            v = np.array(vp['pixel_xy'])
            ax.plot([xy[0, 0], v[0]], [xy[0, 1], v[1]], '--', lw=1)
    if vp is not None and not vp['at_infinity']:
        x, y = vp['pixel_xy']; ax.plot(x, y, 'r+', ms=14, mew=2)
        ax.annotate('One direction vanishing point\nNot a horizon or wall angle', (x, y), xytext=(x + 40, y - 40), color='darkred', bbox=dict(fc='white', alpha=.85))
    ax.set(xlim=(0, width), ylim=(height, 0), title='Parallel floor seams: one direction constraint')
    ax.legend(loc='upper left'); fig.tight_layout(); fig.savefig(out / 'images/floor_vanishing_point.png', dpi=150)
    # Store a standalone, local report alongside the notebook outputs.
    report = '<!doctype html><meta charset="utf-8"><title>Trial2 geometry assumptions</title>'
    report += '<style>body{max-width:1100px;margin:40px auto;font:17px/1.6 sans-serif}pre{white-space:pre-wrap}img{max-width:100%}</style>'
    report += '<h1>Trial2 geometry assumptions and constraints</h1><pre>' + html.escape(result['notebook_markdown']) + '</pre>'
    for name in ('geometry_reference_points.jpg', 'geometry_hole_details.png', 'floor_vanishing_point.png'):
        report += f'<img src="../images/{name}" alt="{name}">'
    (out / 'reports/geometry_calibration.html').write_text(report)
    return result


def load_geometry_run(output_dir, video_file, observation_file, reference_frame,
                      spacing_cm, source_height_cm, target_height_cm, target_ratios=None):
    """Reject stale geometry before notebook IK review or export."""
    path = Path(output_dir) / 'state/geometry_calibration.json'
    result = json.loads(path.read_text())
    observations = json.loads(Path(observation_file).read_text())
    validate_reference(observations, video_file, reference_frame, observations['image_size'])
    expected = analyze_geometry(observations, spacing_cm, source_height_cm, target_height_cm, target_ratios)
    for key in expected:
        if result.get(key) != expected[key]:
            raise ValueError('Geometry settings changed. Rerun Section 4C.')
    p = result['provenance']
    if (p['observation_sha256'] != file_sha256(observation_file) or
            p['module_sha256'] != file_sha256(__file__) or
            p['video_sha256'] != file_sha256(video_file) or result['reference_frame'] != reference_frame):
        raise ValueError('Geometry inputs changed. Rerun Section 4C.')
    return result


def geometry_artifacts(output_dir):
    out = Path(output_dir)
    names = ['state/geometry_calibration.json', 'state/geometry_observations.json',
             'tables/geometry_local_scale.csv', 'images/geometry_reference_points.jpg',
             'images/geometry_hole_details.png', 'images/floor_vanishing_point.png',
             'reports/geometry_calibration.html']
    paths = [out / name for name in names]
    if not all(p.is_file() for p in paths):
        raise ValueError('Geometry artifacts are incomplete. Rerun Section 4C.')
    return paths
