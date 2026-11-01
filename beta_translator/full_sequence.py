"""Whole-sequence 2D retargeting with variable contacts and explicit residuals.

This diagnostic solver exports finite best-effort trajectories, including frames
that fail contact tolerance. It must not be used as the strict single-move gate.
"""
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, diags, kron, vstack

from .inverse_kinematics import LIMBS, make_body, forward_kinematics
from .motion_ik import batch_joints


DEFAULT_SETTINGS = dict(
    contact_scale_cm=1.0,
    outside_hull_weight=0.15,
    reference_angle_weight=0.6,
    reference_hip_scale_cm=35.0,
    free_endpoint_scale_cm=30.0,
    velocity_weight=0.08,
    acceleration_weight=0.03,
    contact_tolerance_cm=0.1,
    large_error_cm=5.0,
    angle_step_warning_deg=15.0,
    hip_step_warning_cm=5.0,
    max_nfev=1000,
)


def build_schedule(contacts, frames, geometry, accept_threshold=0.8):
    """Keep decoded labels intact. `free` does not identify ground/wall/air support."""
    frames = np.asarray(frames)
    if (frames.dtype.kind not in 'iu' or frames.ndim != 1 or len(frames) < 3
            or not np.all(np.diff(frames) == 1)):
        raise ValueError('Use every consecutive frame, with at least three frames.')
    if not np.isfinite(accept_threshold) or not 0 <= accept_threshold <= 1:
        raise ValueError('Invalid confidence threshold.')
    table = contacts.set_index(['frame', 'limb'])
    if not table.index.is_unique:
        raise ValueError('Duplicate contact frame/limb rows.')
    expected = {(int(f), limb) for f in frames for limb in LIMBS}
    if set(table.index) != expected:
        raise ValueError('Contact rows must match all requested frames and four limbs exactly.')
    shape = (len(frames), 4)
    labels = np.empty(shape, dtype='U64')
    states = np.empty(shape, dtype='U40')
    targets = np.zeros(shape + (2,))
    confidence = np.zeros(shape)
    known = np.zeros(shape, bool)
    supported = np.zeros(shape, bool)
    supported_ids = set(geometry['supported_hold_ids'])
    positions = geometry['hold_positions_cm']
    for t, frame in enumerate(frames):
        for i, limb in enumerate(LIMBS):
            row = table.loc[(int(frame), limb)]
            label = str(row['candidate_hold'])
            if len(label) > 64:
                raise ValueError('Contact label is too long.')
            labels[t, i] = label
            confidence[t, i] = float(row['confidence'])
            if label in positions:
                target = np.asarray(positions[label], float)
                if target.shape != (2,) or not np.isfinite(target).all():
                    raise ValueError('Nonfinite hold target.')
                targets[t, i] = target
                known[t, i] = True
                supported[t, i] = label in supported_ids
                states[t, i] = 'hold_in_coverage' if supported[t, i] else 'hold_extrapolated'
            elif label == 'free':
                states[t, i] = 'free_support_unresolved'
            elif label == 'wall_contact':
                states[t, i] = 'wall_contact_location_unknown'
            elif label in ('ground_contact', 'airborne'):
                # The deployed model does not produce these classes. Do not silently
                # accept a future label without a ground/depth observation model.
                raise ValueError('Ground/air labels require an explicit observation model.')
            else:
                states[t, i] = 'unknown_or_uncalibrated'
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('Contact confidence must be finite and in [0,1].')
    return dict(frames=frames.copy(), labels=labels, states=states, targets=targets,
                known=known, supported=supported, confidence=confidence,
                low_confidence=confidence < accept_threshold)


def endpoint_jacobian(poses, body):
    """Derivative of four endpoints with respect to each 11D pose; (T,4,2,11)."""
    q = np.asarray(poses)
    jac = np.zeros((len(q), 4, 2, 11), dtype=q.dtype)
    jac[:, :, 0, 0] = 1
    jac[:, :, 1, 1] = 1
    up = np.stack([-np.sin(q[:, 2]), np.cos(q[:, 2])], axis=-1)
    right = np.stack([np.cos(q[:, 2]), np.sin(q[:, 2])], axis=-1)
    for i in range(4):
        sign = -1 if i in (0, 2) else 1
        offset = body['shoulder_w'] / 2 if i < 2 else body['hip_w'] / 2
        jac[:, i, :, 2] = sign * offset * up - (body['torso'] * right if i < 2 else 0)
        a, b = q[:, 3 + 2*i], q[:, 4 + 2*i]
        first = body['upper_arm'] if i < 2 else body['thigh']
        second = body['forearm'] if i < 2 else body['shank']
        lower = second * np.stack([np.cos(a+b), np.sin(a+b)], axis=-1)
        jac[:, i, :, 3 + 2*i] = first * np.stack([np.cos(a), np.sin(a)], axis=-1) + lower
        jac[:, i, :, 4 + 2*i] = lower
    return jac


def solve_full_sequence(reference, visibility, reference_endpoints, schedule, body,
                        fps, settings=None, progress=None):
    """One sparse optimization over all frames; no move slicing or window stitching."""
    config = {**DEFAULT_SETTINGS, **(settings or {})}
    if set(config) != set(DEFAULT_SETTINGS):
        raise ValueError('Unknown full-sequence setting.')
    if (any(not np.isfinite(v) or v <= 0 for v in config.values())
            or type(config['max_nfev']) is not int or config['outside_hull_weight'] > 1):
        raise ValueError('Settings must be finite and positive; max_nfev must be an integer.')
    frames = schedule['frames']
    count = len(frames)
    if count < 3 or not np.all(np.diff(frames) == 1):
        raise ValueError('All consecutive source frames are required.')
    ref = np.asarray(reference, float).copy()
    vis = np.asarray(visibility, float)
    ends_ref = np.asarray(reference_endpoints, float)
    if (ref.shape != (count, 11) or vis.shape != (count, 9)
            or ends_ref.shape != (count, 4, 2)
            or not all(np.isfinite(a).all() for a in (ref, vis, ends_ref))
            or np.any((vis < 0) | (vis > 1)) or not np.isfinite(fps) or fps <= 0):
        raise ValueError('Invalid references, visibility, or FPS.')
    for key, shape in [('targets', (count, 4, 2)), ('known', (count, 4)),
                       ('supported', (count, 4)), ('confidence', (count, 4))]:
        if np.asarray(schedule[key]).shape != shape or not np.isfinite(schedule[key]).all():
            raise ValueError('Invalid contact schedule arrays.')
    lengths = make_body(body['height_cm'], body['ratios'])
    # Unwrap orientation angles. Joint bends remain signed within +/-150 degrees.
    orientation = [2, 3, 5, 7, 9]
    ref[:, orientation] = np.unwrap(ref[:, orientation], axis=0)
    lower, upper = np.full(11, -np.inf), np.full(11, np.inf)
    lower[4::2], upper[4::2] = -np.deg2rad(150), np.deg2rad(150)
    conf = schedule['confidence']
    reliability = np.where(schedule['supported'], 1.0, config['outside_hull_weight'])
    contact_weight = np.sqrt(np.maximum(conf, 0.05) * reliability) * schedule['known'] / config['contact_scale_cm']
    angle_weight = config['reference_angle_weight'] * np.sqrt(vis)
    free_weight = (~schedule['known']) / config['free_endpoint_scale_cm']
    scales = np.array([50.0, 50.0] + [1.0]*9)

    def local(q, indices):
        ends = batch_joints(q, lengths)[:, :, 2]
        return np.concatenate([
            ((ends - schedule['targets'][indices]) * contact_weight[indices, :, None]).reshape(len(q), 8),
            angle_weight[indices] * 2 * np.sin((q[:, 2:] - ref[indices, 2:]) / 2),
            (q[:, :2] - ref[indices, :2]) / config['reference_hip_scale_cm'],
            ((ends - ends_ref[indices]) * free_weight[indices, :, None]).reshape(len(q), 8),
        ], axis=1)

    def local_jac(q, indices):
        endpoint = endpoint_jacobian(q, lengths)
        jac = np.zeros((len(q), 27, 11))
        jac[:, :8] = (endpoint * contact_weight[indices, :, None, None]).reshape(len(q), 8, 11)
        jac[:, 8+np.arange(9), 2+np.arange(9)] = angle_weight[indices] * np.cos((q[:, 2:] - ref[indices, 2:]) / 2)
        jac[:, 17, 0] = jac[:, 18, 1] = 1 / config['reference_hip_scale_cm']
        jac[:, 19:] = (endpoint * free_weight[indices, :, None, None]).reshape(len(q), 8, 11)
        return jac

    # Sequential initialization supplies a branch-continuous starting guess only.
    # Every frame is subsequently optimized jointly, including both endpoints.
    seed = np.clip(ref[0], lower, upper)
    seeds = []
    for t in range(count):
        previous = seed.copy()
        def initial_residual(q):
            return np.r_[local(q[None], slice(t, t+1)).ravel(), 0.2*(q-previous)/scales]
        def initial_jac(q):
            return np.vstack([local_jac(q[None], slice(t, t+1))[0], np.diag(0.2/scales)])
        fit = least_squares(initial_residual, seed, jac=initial_jac,
                            bounds=(lower, upper), max_nfev=100, ftol=1e-6)
        if not np.isfinite(fit.x).all():
            raise RuntimeError(f'Nonfinite initialization at frame {frames[t]}.')
        seed = fit.x
        seeds.append(seed.copy())
        if progress and (t % 100 == 0 or t == count-1):
            progress(f'Initialized {t+1}/{count} frames')

    first = diags([-np.ones(count-1), np.ones(count-1)], [0, 1], shape=(count-1, count))
    second = diags([np.ones(count-2), -2*np.ones(count-2), np.ones(count-2)], [0, 1, 2], shape=(count-2, count))
    smooth = vstack([
        kron(first, diags(config['velocity_weight']*fps/scales)),
        kron(second, diags(config['acceleration_weight']*fps**2/scales)),
    ], format='csr')
    row = np.broadcast_to(np.arange(count*27).reshape(count, 27, 1), (count, 27, 11)).ravel()
    col = np.broadcast_to((np.arange(count)[:, None, None]*11 + np.arange(11)), (count, 27, 11)).ravel()
    evaluations = [0]
    def residual(flat):
        evaluations[0] += 1
        if progress and evaluations[0] % 25 == 0:
            progress(f'Global trajectory evaluation {evaluations[0]} ({count} jointly fitted frames)')
        return np.r_[local(flat.reshape(count, 11), slice(None)).ravel(), smooth @ flat]
    def jacobian(flat):
        data = local_jac(flat.reshape(count, 11), slice(None)).ravel()
        sparse = coo_matrix((data, (row, col)), shape=(count*27, count*11)).tocsr()
        sparse.eliminate_zeros()
        return vstack([sparse, smooth], format='csr')
    solution = least_squares(residual, np.asarray(seeds).ravel(), jac=jacobian,
        bounds=(np.tile(lower, count), np.tile(upper, count)), x_scale='jac',
        tr_options={'atol': 1e-7, 'btol': 1e-7, 'maxiter': 500},
        max_nfev=config['max_nfev'], ftol=1e-6, xtol=1e-7, gtol=1e-6)
    if not np.isfinite(solution.x).all():
        raise RuntimeError('Nonfinite full-sequence output; no video can be exported.')
    poses = solution.x.reshape(count, 11)
    return dict(frames=frames, poses=poses, joints=batch_joints(poses, lengths),
        reference=ref, visibility=vis, reference_endpoints=ends_ref, schedule=schedule,
        body=body, lengths=lengths, settings=config,
        optimizer=dict(success=bool(solution.success), message=str(solution.message),
                       nfev=int(solution.nfev), cost=float(solution.cost),
                       optimality=float(solution.optimality)),
        method='One joint optimization over every source frame; variable per-limb contacts.')


def diagnose_sequence(result, reference_outside=None):
    """Separate geometry residuals, input uncertainty, and optimizer convergence."""
    q, joints, schedule, config = (result[k] for k in ('poses', 'joints', 'schedule', 'settings'))
    count = len(q)
    known, supported = schedule['known'], schedule['supported']
    errors = np.linalg.norm(joints[:, :, 2] - schedule['targets'], axis=-1)
    errors = np.where(known, errors, np.nan)
    max_error = np.max(np.where(known, errors, 0), axis=1)
    length_errors = []
    for i in range(4):
        names = ('upper_arm', 'forearm') if i < 2 else ('thigh', 'shank')
        for j, name in enumerate(names):
            length_errors.append(np.abs(np.linalg.norm(joints[:, i, j+1]-joints[:, i, j], axis=1)-result['lengths'][name]))
    roots = forward_kinematics(np.zeros(11), result['lengths']).limbs
    reaches = [result['lengths']['upper_arm']+result['lengths']['forearm']]*2 + [result['lengths']['thigh']+result['lengths']['shank']]*2
    span_excess = np.zeros(count)
    span_pairs = [[] for _ in range(count)]
    for a, b in combinations(range(4), 2):
        bound = reaches[a]+reaches[b]+np.linalg.norm(roots[LIMBS[a]].root-roots[LIMBS[b]].root)
        excess = np.linalg.norm(schedule['targets'][:, a]-schedule['targets'][:, b], axis=1)-bound
        active = supported[:, a] & supported[:, b]
        span_excess = np.maximum(span_excess, np.where(active, excess, 0))
        for t in np.flatnonzero(active & (excess > 2*config['contact_tolerance_cm'])):
            span_pairs[t].append(f'{LIMBS[a]}/{LIMBS[b]}:+{excess[t]:.2f}cm')
    angle_steps = np.r_[0., np.rad2deg(np.abs(np.diff(q[:, 2:], axis=0))).max(axis=1)]
    hip_steps = np.r_[0., np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1)]
    flags = {
        'contact_residual': max_error > config['contact_tolerance_cm'],
        'large_contact_residual': max_error > config['large_error_cm'],
        'span_conflict': span_excess > 2*config['contact_tolerance_cm'],
        'hold_extrapolation': np.any(known & ~supported, axis=1),
        'unknown_contact': np.any(schedule['states'] == 'unknown_or_uncalibrated', axis=1),
        'unresolved_foot_support': np.any(~known[:, 2:], axis=1),
        'wall_contact_unlocated': np.any(schedule['labels'] == 'wall_contact', axis=1),
        'low_contact_confidence': np.any(schedule['low_confidence'], axis=1),
        'low_pose_visibility': np.any(result['visibility'] < 0.5, axis=1),
        'reference_extrapolation': np.zeros(count, bool) if reference_outside is None else np.asarray(reference_outside, bool),
        'rapid_pose_change': (angle_steps > config['angle_step_warning_deg']) | (hip_steps > config['hip_step_warning_cm']),
        'optimizer_unconverged': np.full(count, not result['optimizer']['success']),
    }
    if flags['reference_extrapolation'].shape != (count,):
        raise ValueError('Reference-coverage flags must have one value per frame.')
    issues = [';'.join(k for k, values in flags.items() if values[t]) for t in range(count)]
    constraint = flags['contact_residual'] | flags['span_conflict'] | flags['optimizer_unconverged']
    uncertain = np.array([bool(s) for s in issues])
    status = np.where(constraint, 'constraint_warning', np.where(uncertain, 'uncertain', 'within_numeric_tolerance'))
    return dict(contact_errors_cm=errors, max_contact_error_cm=max_error, flags=flags,
        issues=issues, status=status, span_excess_cm=span_excess, span_pairs=span_pairs,
        angle_step_deg=angle_steps, hip_step_cm=hip_steps,
        max_segment_length_error_cm=float(np.max(length_errors)),
        counts={k:int(v.sum()) for k,v in flags.items()})
