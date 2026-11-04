"""Compare reviewed endpoint IK with static balance, without changing saved runs."""

from pathlib import Path
import hashlib
import json
import tempfile

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .inverse_kinematics import (
    LIMBS, RATIOS, GRAVITY, MASS_FRACTIONS, CENTER_OF_MASS_FRACTIONS,
    GravityIKResult, make_body, make_mass_model, forward_kinematics,
    center_of_mass, static_balance_error, solve_inverse_kinematics,
    solve_inverse_kinematics_with_gravity,
)
from .pose_adapter import (
    LIMB_LANDMARKS, to_frame0, image_to_wall_cm, pose_row_to_vector,
    angle_visibility_weights,
)
from .project_io import (
    file_sha256, load_reviewed_holds, load_alignment, load_annotations,
    validate_reference_alignment,
)


def evaluate_fixed_pose_balance(body, targets, pose, mass_kg, *, foot_friction=0.8,
                                max_force_n=None, balance_weight=30.0,
                                force_weight=0.05, balance_tolerance=0.01):
    """Fit forces only; keep the ordinary IK pose exactly fixed for a fair check.

    The force objective, component bounds and foot approximation match gravity IK.
    These forces are inferred, not measured. A failed fit is reported, not accepted.
    """
    masses = make_mass_model(mass_kg)
    if not np.isfinite(foot_friction) or foot_friction < 0:
        raise ValueError('Foot friction must be non-negative and finite.')
    if max_force_n is not None and (not np.isfinite(max_force_n) or max_force_n <= 0):
        raise ValueError('Force limit must be positive and finite.')
    contacted = [limb for limb in LIMBS if targets.get(limb) is not None]
    if not contacted:
        raise ValueError('At least one known contact is required.')
    joints = forward_kinematics(pose, body)
    ends = np.array([joints.limbs[limb].end for limb in contacted])
    com = center_of_mass(pose, body, masses)
    weight = mass_kg * GRAVITY
    limit = 1.0 if max_force_n is None else max_force_n / weight
    feet = np.array([limb.endswith('foot') for limb in contacted])
    lever = (ends - com) / body['torso']

    def residuals(flat):
        forces = flat.reshape(-1, 2)
        force_error = forces.sum(axis=0) - [0., 1.]
        moment = np.sum(lever[:, 0] * forces[:, 1] - lever[:, 1] * forces[:, 0])
        friction = np.maximum(0., np.abs(forces[feet, 0]) - foot_friction * forces[feet, 1])
        return np.r_[balance_weight * force_error, balance_weight * moment,
                     balance_weight * friction, force_weight * flat]

    lower = np.full((len(contacted), 2), -limit)
    lower[feet, 1] = 0.
    initial = np.zeros_like(lower)
    initial[:, 1] = min(1. / len(contacted), limit)
    solution = least_squares(residuals, initial.ravel(),
                             bounds=(lower.ravel(), limit), x_scale='jac')
    if not solution.success:
        raise RuntimeError(f'Fixed-pose force fit did not converge: {solution.message}')
    forces = solution.x.reshape(-1, 2) * weight
    net_force, moment = static_balance_error(com, ends, forces, mass_kg)
    excess = np.maximum(0., np.abs(forces[feet, 0]) - foot_friction * forces[feet, 1])
    tolerance = balance_tolerance * weight
    balanced = bool(np.linalg.norm(net_force) <= tolerance
                    and abs(moment) <= tolerance * body['torso']
                    and (not excess.size or excess.max() <= tolerance))
    return GravityIKResult(np.asarray(pose).copy(),
        {limb: float(np.linalg.norm(ends[i] - targets[limb])) for i, limb in enumerate(contacted)},
        {limb: forces[i].copy() for i, limb in enumerate(contacted)}, com,
        net_force, moment, balanced)


def comparison_metrics(result, reference, body, foot_friction, contact_tolerance_cm):
    """Keep contact validity separate from the solver's equilibrium-only flag."""
    errors = list(result.contact_errors_cm.values())
    friction = [max(0., abs(f[0]) - foot_friction * f[1])
                for limb, f in result.contact_forces_n.items() if limb.endswith('foot')]
    maximum = float(max(errors))
    return dict(max_contact_error_cm=maximum, mean_contact_error_cm=float(np.mean(errors)),
        net_force_n=float(np.linalg.norm(result.net_force_n)),
        abs_net_moment_nm=float(abs(result.net_moment_ncm) / 100.),
        max_foot_friction_excess_n=float(max(friction, default=0.)),
        com_x_cm=float(result.center_of_mass_cm[0]), com_y_cm=float(result.center_of_mass_cm[1]),
        hip_shift_from_reference_cm=float(np.linalg.norm(result.pose_vector[:2] - reference[:2])),
        max_angle_change_from_reference_deg=float(np.rad2deg(np.max(np.abs(result.pose_vector[2:] - reference[2:])))),
        contacts_within_tolerance=bool(maximum <= contact_tolerance_cm),
        balanced=bool(result.balanced),
        model_checks_pass=bool(maximum <= contact_tolerance_cm and result.balanced))


def compare_endpoint(body, targets, reference, visibility, mass_kg, *,
                     foot_friction=0.8, max_force_n=None, contact_tolerance_cm=0.1):
    """Solve both methods independently from the same observed reference."""
    fits = {}
    for method in ('ordinary', 'gravity'):
        try:
            if method == 'ordinary':
                pose, _ = solve_inverse_kinematics(
                    body, targets, reference, angle_weight=0.3 * visibility)
                result = evaluate_fixed_pose_balance(body, targets, pose, mass_kg,
                    foot_friction=foot_friction, max_force_n=max_force_n)
            else:
                result = solve_inverse_kinematics_with_gravity(body, targets, reference, mass_kg,
                    angle_weight=0.3 * visibility, foot_friction=foot_friction, max_force_n=max_force_n)
            fits[method] = dict(converged=True, error=None,
                metrics=comparison_metrics(result, reference, body, foot_friction, contact_tolerance_cm),
                pose_vector=result.pose_vector.tolist(),
                center_of_mass_cm=result.center_of_mass_cm.tolist(),
                contact_errors_cm=result.contact_errors_cm,
                contact_forces_n={limb: f.tolist() for limb, f in result.contact_forces_n.items()},
                net_force_vector_n=result.net_force_n.tolist(), net_moment_ncm=result.net_moment_ncm)
        except RuntimeError as error:
            fits[method] = dict(converged=False, error=str(error), metrics=dict(
                model_checks_pass=False, balanced=None, contacts_within_tolerance=None))
    return fits


def _plot_comparison(endpoints, body, mass_kg, masks, records, scale, image_height):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D

    figure = Figure(figsize=(12, 10), facecolor='white')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    records = {record['hold_id']: record for record in records}
    all_points = []
    for entry in endpoints:
        all_points.extend(entry['targets_cm'].values())
        for fit in entry['fits'].values():
            if fit['converged']:
                joints = forward_kinematics(fit['pose_vector'], body)
                all_points.extend(np.asarray(list(joints.limbs.values())).reshape(-1, 2))
                all_points.append(np.asarray(fit['center_of_mass_cm']) + [0, -35])
    points = np.asarray(all_points)
    lo, hi = points.min(axis=0) - [35, 65], points.max(axis=0) + [35, 35]
    span = max(hi - lo)
    middle = (lo + hi) / 2
    lo, hi = middle - span / 2, middle + span / 2
    force_scale = 60. / (mass_kg * GRAVITY)  # One body weight has a 60 cm display length.
    for row, entry in enumerate(endpoints):
        for col, method in enumerate(('ordinary', 'gravity')):
            ax, fit = axes[row, col], entry['fits'][method]
            for hold_id in set(entry['contacts'].values()):
                contours, _ = cv2.findContours(masks[hold_id].astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour = image_to_wall_cm(contour.reshape(-1, 2), scale, image_height)
                    ax.fill(*contour.T, color='#e2e6eb', edgecolor='#aab3bd', linewidth=0.8)
                record = records[hold_id]
                target = image_to_wall_cm([[record['cx_px'], record['cy_px']]], scale, image_height)[0]
                ax.plot(*target, 'x', color='#c6333b', markersize=7)
                ax.annotate(hold_id, target, xytext=(8, 5), textcoords='offset points', fontsize=9)
            label = 'Ordinary IK + fixed-pose force check' if method == 'ordinary' else 'Gravity IK: pose + forces'
            ax.set_title(f"{entry['phase'].upper()} | frame {entry['frame']}\n{label}", fontsize=11, pad=10)
            if fit['converged']:
                color = '#2563a5' if method == 'ordinary' else '#16845c'
                joints = forward_kinematics(fit['pose_vector'], body)
                roots = np.array([joints.limbs[l].root for l in ('left_hand', 'right_hand', 'right_foot', 'left_foot', 'left_hand')])
                ax.plot(*roots.T, 'o-', color=color, linewidth=2, markersize=3)
                for limb, chain in joints.limbs.items():
                    ax.plot(*np.array(chain).T, 'o-', color=color, linewidth=2, markersize=4)
                    target = entry['targets_cm'][limb]
                    ax.plot([chain.end[0], target[0]], [chain.end[1], target[1]], ':', color='#c6333b')
                    force = np.asarray(fit['contact_forces_n'][limb]) * force_scale
                    ax.quiver(*chain.end, *force, angles='xy', scale_units='xy', scale=1,
                        color='#a75b05', width=.006, zorder=6)
                com = np.asarray(fit['center_of_mass_cm'])
                ax.plot(*com, marker='D', color='#7c3aa5', markersize=7)
                ax.quiver(*com, 0, -60, angles='xy', scale_units='xy', scale=1,
                    color='#7c3aa5', width=.006, zorder=6)
                m = fit['metrics']
                text = (f"Max contact gap: {m['max_contact_error_cm']:.2e} cm\n"
                        f"Net force: {m['net_force_n']:.2e} N | moment: {m['abs_net_moment_nm']:.2e} N m\n"
                        f"Contact check: {'PASS' if m['contacts_within_tolerance'] else 'FAIL'} | "
                        f"Balance check: {'PASS' if m['balanced'] else 'FAIL'}")
            else:
                text = 'NOT SOLVED\n' + fit['error']
            ax.text(.02, .02, text, transform=ax.transAxes, fontsize=8,
                va='bottom', bbox=dict(facecolor='white', alpha=.92, edgecolor='#d4d8dd'))
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect='equal',
                   xlabel='Scaled wall x (cm)', ylabel='Scaled wall y (cm)')
            ax.grid(alpha=.15)
    figure.legend(handles=[Line2D([], [], color='#a75b05', label='Inferred contact force'),
        Line2D([], [], color='#7c3aa5', marker='D', label='COM and gravity'),
        Line2D([], [], color='#c6333b', marker='x', linestyle='None', label='Hold-center target')],
        loc='lower center', bbox_to_anchor=(.5, .055), ncol=3, frameon=False)
    figure.text(.5, .022, 'Static 2D approximation; arrows: 1 body weight = 60 cm display length.\n'
        'Numerical checks do not establish real climbing feasibility. Both methods may already balance.',
        ha='center', fontsize=9, color='#444444')
    figure.subplots_adjust(top=.91, bottom=.14, hspace=.34, wspace=.15)
    return figure


def run_gravity_comparison(video_file, output_dir, reference_path, *, height_cm,
                           ratios=None, mass_kg=70., foot_friction=0.8,
                           max_force_n=None, contact_tolerance_cm=0.1,
                           reference_frame_number=0, result_root=None):
    """Read the confirmed example, export a new comparison, preserve all old outputs.

    This does not load/save the live annotation form or run the motion pipeline.
    Identity/scale checks prevent reuse on a different video, map, or reference.
    """
    video_file, output_dir, reference_path = map(Path, (video_file, output_dir, reference_path))
    ratios = dict(RATIOS if ratios is None else ratios)
    body = make_body(height_cm, ratios)
    make_mass_model(mass_kg)
    if reference_frame_number != 0:
        raise ValueError('The confirmed H44 → H28 scale is valid only at reference frame 0.')
    if not np.isfinite(contact_tolerance_cm) or contact_tolerance_cm <= 0:
        raise ValueError('Contact tolerance must be positive and finite.')
    if not np.isfinite(foot_friction) or foot_friction < 0:
        raise ValueError('Foot friction must be non-negative and finite.')
    if max_force_n is not None and (not np.isfinite(max_force_n) or max_force_n <= 0):
        raise ValueError('Force limit must be positive and finite.')
    paths = dict(video=video_file, reference=reference_path,
        holds=output_dir/'state/reviewed_hold_map.npz',
        alignment=output_dir/'state/alignment_cache.npz', poses=output_dir/'tables/pose_coordinates.csv')

    def input_hashes():
        return {name: file_sha256(path) for name, path in paths.items()}

    before_hashes = input_hashes()
    records, masks, run_id, hold_ids = load_reviewed_holds(paths['holds'], video_file, reference_frame_number=0)
    if validate_reference_alignment(paths['holds'], paths['alignment']) != 0:
        raise ValueError('The confirmed example requires reference-frame-0 alignment.')
    capture = cv2.VideoCapture(str(video_file))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    annotation = load_annotations(reference_path, video_file, run_id, hold_ids, frame_count)
    expected = {'before': dict(zip(LIMBS, ('H44', 'H33', 'H62', 'H65'))),
                'after': dict(zip(LIMBS, ('H28', 'H33', 'H62', 'H65')))}
    if annotation['frames'] != dict(before=107, after=129) or annotation['contacts'] != expected:
        raise ValueError('Expected the confirmed H44 → H28 endpoint example, frames 107 and 129.')
    scale = annotation['body']['px_per_cm']
    if scale is None or not np.isclose(scale, np.hypot(80, -21) / 46, rtol=0, atol=1e-10):
        raise ValueError('The example needs its confirmed 46 cm reference-image scale.')
    with np.load(paths['alignment'], allow_pickle=False) as saved:
        settings = json.loads(str(saved['settings'].item()))
    transforms = load_alignment(paths['alignment'], video_file, paths['poses'], settings)
    if transforms is None:
        raise ValueError('Alignment is stale for this video/pose table. Rerun Section 5.')
    poses = pd.read_csv(paths['poses']).set_index('frame')
    if not poses.index.is_unique:
        raise ValueError('Duplicate pose frames.')
    configuration = dict(height_cm=float(height_cm), ratios=ratios, mass_kg=float(mass_kg),
        foot_friction=float(foot_friction), max_force_n=max_force_n,
        contact_tolerance_cm=float(contact_tolerance_cm), balance_tolerance=0.01,
        gravity=GRAVITY, contact_weight=10., angle_weight=0.3, hip_position_weight=0.02,
        balance_weight=30., force_weight=0.05, max_bend_deg=150., px_per_cm=scale,
        mass_fractions=MASS_FRACTIONS, com_fractions=CENTER_OF_MASS_FRACTIONS)
    modules = ('gravity_comparison.py', 'inverse_kinematics.py', 'pose_adapter.py', 'project_io.py')
    provenance = dict(inputs=before_hashes, modules={name: file_sha256(Path(__file__).with_name(name)) for name in modules})
    key = hashlib.sha256(json.dumps(dict(configuration=configuration, provenance=provenance), sort_keys=True).encode()).hexdigest()[:12]
    root = Path(result_root) if result_root is not None else output_dir/'gravity_comparison'
    destination = root/f'H44_H28_{height_cm:g}cm_{mass_kg:g}kg_{key}'
    if destination.exists():
        summary = json.loads((destination/'summary.json').read_text())
        if summary['provenance'] != provenance or summary['configuration'] != configuration:
            raise ValueError('Existing comparison provenance does not match. Preserve it and choose another result_root.')
        for name, digest in summary['output_sha256'].items():
            path = destination/name
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError('Existing comparison was changed. Preserve it and choose another result_root.')
        return dict(summary, output_dir=str(destination))
    height = next(iter(masks.values())).shape[0]
    hold_targets = {r['hold_id']: image_to_wall_cm([[r['cx_px'], r['cy_px']]], scale, height)[0] for r in records}
    names = list(dict.fromkeys(n for chain in LIMB_LANDMARKS.values() for n in chain))
    endpoints, metrics = [], []
    for phase, frame in annotation['frames'].items():
        if frame not in poses.index or frame not in transforms or transforms[frame][2] != 'ok':
            raise ValueError(f'Frame {frame} needs a pose and verified alignment.')
        row = poses.loc[frame].copy()
        points = to_frame0([[row[n+'_x_px'], row[n+'_y_px']] for n in names], transforms, frame)
        if not np.isfinite(points).all():
            raise ValueError(f'Invalid aligned landmarks at frame {frame}.')
        for name, point in zip(names, points):
            row[name+'_x_px'], row[name+'_y_px'] = point
        reference = pose_row_to_vector(row, scale, height)
        visibility = angle_visibility_weights(row)
        if not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)):
            raise ValueError(f'Invalid visibility at frame {frame}.')
        targets = {limb: hold_targets[hold] for limb, hold in annotation['contacts'][phase].items()}
        fits = compare_endpoint(body, targets, reference, visibility, mass_kg,
            foot_friction=foot_friction, max_force_n=max_force_n, contact_tolerance_cm=contact_tolerance_cm)
        endpoints.append(dict(phase=phase, frame=frame, contacts=annotation['contacts'][phase],
            reference_pose=reference.tolist(), angle_visibility=visibility.tolist(),
            targets_cm={limb: target.tolist() for limb, target in targets.items()}, fits=fits))
        for method, fit in fits.items():
            metrics.append(dict(phase=phase, frame=frame, method=method,
                converged=fit['converged'], error=fit['error'], **fit['metrics']))
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.staging-', dir=root) as temp:
        stage = Path(temp)
        figure = _plot_comparison(endpoints, body, mass_kg, masks, records, scale, height)
        figure.suptitle(f'H44 → H28 | target {height_cm:g} cm | assumed mass {mass_kg:g} kg', fontsize=16)
        figure.savefig(stage/'comparison.png', dpi=160)
        figure.clear()
        pd.DataFrame(metrics).to_csv(stage/'metrics.csv', index=False)
        summary = dict(schema_version=1, configuration=configuration, provenance=provenance,
            reference_annotation=annotation, endpoints=endpoints, metrics=metrics,
            interpretation='Independent endpoint comparison, not a motion or whole-video gravity result.',
            limitations=[
                'Mass is a user-supplied or demonstration assumption, not measured from the video.',
                'Image-up gravity and uniform image scale approximate the wall; perspective/depth are not recovered.',
                'Contact forces are inferred; hands have arbitrary in-plane direction within component limits.',
                'Feet use fy >= 0 and abs(fx) <= mu*fy as a simplified hold-support model.',
                'No hold normals, grip strength, joint torque, collision or dynamic balance model.',
                'Fixed-pose ordinary IK force fitting does not change its pose; both methods may pass balance.',
                'Contact and balance checks are separate numerical tests, not real feasibility.',
                'With default force limits proportional to weight, mass alone does not change the fitted pose.',
                'Existing annotation forms, motion videos, and full-sequence outputs are unchanged.'],
            output_sha256={name: file_sha256(stage/name) for name in ('comparison.png', 'metrics.csv')})
        (stage/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        if input_hashes() != before_hashes:
            raise RuntimeError('Inputs changed during the comparison; no result installed.')
        stage.rename(destination)
    return dict(summary, output_dir=str(destination))
