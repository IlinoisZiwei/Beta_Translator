"""Contact-aware 2D trajectory fitting. This is kinematics, not a dynamics model."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .inverse_kinematics import LIMBS, make_body, solve_inverse_kinematics


DEFAULT_SETTINGS = dict(
    contact_scale_cm=0.05,
    contact_tolerance_cm=0.1,
    reference_angle_weight=0.3,
    reference_hip_weight=0.5,
    swing_weight=0.7,
    swing_scale_cm=20.0,
    velocity_weight=0.04,
    acceleration_weight=0.004,
    max_nfev=500,
)


def validate_motion_annotation(annotation):
    """Release is the first free frame; arrival is the first new-contact frame."""
    motion = annotation.get('motion')
    if not isinstance(motion, dict):
        raise ValueError('Save release/arrival frames and confirm the other three contacts first.')
    before, after = (annotation['frames'][k] for k in ('before', 'after'))
    release, arrival = (motion.get(k) for k in ('release_frame', 'arrival_frame'))
    if any(type(v) is not int for v in (before, release, arrival, after)):
        raise ValueError('Motion endpoints and event frames must be integers.')
    if not before < release < arrival <= after:
        raise ValueError('Motion timing requires BEFORE < RELEASE < ARRIVAL <= AFTER.')
    if motion.get('support_contacts_maintained') is not True:
        raise ValueError('Confirm that the other three limbs remain in contact throughout.')
    moving = annotation['moving_limb']
    if moving not in LIMBS:
        raise ValueError('Unknown moving limb.')
    contacts = annotation['contacts']
    allowed = set(annotation['selected_hold_ids'])
    for phase in ('before', 'after'):
        if set(contacts[phase]) != set(LIMBS) or any(c not in allowed for c in contacts[phase].values()):
            raise ValueError('This first motion model requires four known hold IDs at each endpoint.')
    for limb in LIMBS:
        if limb != moving and contacts['before'][limb] != contacts['after'][limb]:
            raise ValueError('Only the selected moving limb may change holds in this motion model.')
    if contacts['before'][moving] == contacts['after'][moving]:
        raise ValueError('The moving limb must have different start and destination holds.')
    return motion


def contact_schedule(annotation):
    motion = validate_motion_annotation(annotation)
    frames = np.arange(annotation['frames']['before'], annotation['frames']['after'] + 1)
    labels = []
    moving = annotation['moving_limb']
    for frame in frames:
        row = dict(annotation['contacts']['before'])
        if motion['release_frame'] <= frame < motion['arrival_frame']:
            row[moving] = 'free'
        elif frame >= motion['arrival_frame']:
            row[moving] = annotation['contacts']['after'][moving]
        labels.append(row)
    return frames, labels


def batch_joints(poses, body):
    """Vectorized equivalent of forward_kinematics; shape (time, limb, joint, xy)."""
    q = np.asarray(poses)
    angle = q[:, 2]
    up = np.stack([-np.sin(angle), np.cos(angle)], axis=-1)
    right = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    hip = q[:, :2]
    shoulder = hip + body['torso'] * up
    roots = np.stack([
        shoulder - body['shoulder_w'] / 2 * right,
        shoulder + body['shoulder_w'] / 2 * right,
        hip - body['hip_w'] / 2 * right,
        hip + body['hip_w'] / 2 * right,
    ], axis=1)
    angles = q[:, 3:].reshape(-1, 4, 2)
    upper, lower = angles[:, :, 0], angles.sum(axis=-1)
    lengths1 = np.array([body['upper_arm']] * 2 + [body['thigh']] * 2)
    lengths2 = np.array([body['forearm']] * 2 + [body['shank']] * 2)
    middle = roots + lengths1[None, :, None] * np.stack([np.sin(upper), -np.cos(upper)], axis=-1)
    ends = middle + lengths2[None, :, None] * np.stack([np.sin(lower), -np.cos(lower)], axis=-1)
    return np.stack([roots, middle, ends], axis=2)


def solve_motion_sequence(reference_poses, angle_weights, frames, hold_positions_cm,
                          annotation, fps, settings=None):
    """Fit all frames together; reject unverified timing, gaps, or excess contact error.

    Active contacts use strong soft penalties and a separate residual acceptance check.
    The free limb has a minimum-jerk straight-line guide between its contact targets.
    That guide is a modeling assumption, not an observed hand trajectory.
    """
    config = dict(DEFAULT_SETTINGS)
    if settings:
        if set(settings) - set(config):
            raise ValueError('Unknown motion solver setting.')
        config.update(settings)
    if any(not np.isfinite(v) or v <= 0 for v in config.values()):
        raise ValueError('Motion solver settings must be finite and positive.')
    if type(config['max_nfev']) is not int:
        raise ValueError('max_nfev must be an integer.')
    expected_frames, labels = contact_schedule(annotation)
    frames = np.asarray(frames)
    if frames.dtype.kind not in 'iu' or not np.array_equal(frames, expected_frames):
        raise ValueError('Provide every consecutive source frame from BEFORE to AFTER.')
    reference = np.asarray(reference_poses, dtype=float)
    weights = np.asarray(angle_weights, dtype=float)
    count = len(frames)
    if (reference.shape != (count, 11) or weights.shape != (count, 9)
            or not np.isfinite(reference).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or (weights > 1).any()):
        raise ValueError('References must be finite (T,11), and visibility weights (T,9) in [0,1].')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive and finite.')
    reference = reference.copy()
    reference[:, 2:] = np.unwrap(reference[:, 2:], axis=0)
    body = make_body(annotation['body']['height_cm'], annotation['body']['ratios'])
    targets = np.zeros((count, 4, 2))
    active = np.zeros((count, 4), dtype=bool)
    for t, row in enumerate(labels):
        for limb_index, limb in enumerate(LIMBS):
            if row[limb] != 'free':
                target = np.asarray(hold_positions_cm.get(row[limb]), dtype=float)
                if target.shape != (2,) or not np.isfinite(target).all():
                    raise ValueError(f'Missing finite hold position: {row[limb]}')
                targets[t, limb_index] = target
                active[t, limb_index] = True
    moving_index = LIMBS.index(annotation['moving_limb'])
    release = annotation['motion']['release_frame']
    arrival = annotation['motion']['arrival_frame']
    u = np.clip((frames - (release - 1)) / (arrival - (release - 1)), 0, 1)
    blend = 10*u**3 - 15*u**4 + 6*u**5
    guide = targets[0, moving_index] + blend[:, None] * (
        targets[-1, moving_index] - targets[0, moving_index])
    free = ~active[:, moving_index]
    scales = np.array([50., 50.] + [1.] * 9)
    dt = 1.0 / fps
    lower = np.full(11, -np.inf)
    upper = np.full(11, np.inf)
    lower[4::2], upper[4::2] = -np.deg2rad(150), np.deg2rad(150)

    def local_residual(q, t):
        ends = batch_joints(np.atleast_2d(q), body)[0, :, 2]
        return np.concatenate([
            ((ends - targets[t]) * active[t, :, None] / config['contact_scale_cm']).ravel(),
            config['reference_angle_weight'] * np.sqrt(weights[t]) *
            2 * np.sin((q[2:] - reference[t, 2:]) / 2),
            config['reference_hip_weight'] * (q[:2] - reference[t, :2]) / scales[:2],
            (ends[moving_index] - guide[t]) * free[t] *
            config['swing_weight'] / config['swing_scale_cm'],
        ])

    first_targets = {limb: targets[0, i] for i, limb in enumerate(LIMBS)}
    seed, _ = solve_inverse_kinematics(body, first_targets, reference[0],
                                      angle_weight=0.3 * weights[0])
    seeds = []
    for t in range(count):
        previous = seed.copy()
        def objective(q):
            return np.concatenate([local_residual(q, t), 0.1 * (q - previous) / scales])
        initial = least_squares(objective, np.clip(seed, lower, upper), bounds=(lower, upper),
                                jac='cs', x_scale='jac', max_nfev=250)
        if not np.isfinite(initial.x).all():
            raise RuntimeError(f'Non-finite initialization at frame {frames[t]}.')
        seed = initial.x
        seeds.append(seed.copy())

    def residual(flat):
        q = flat.reshape(count, 11)
        ends = batch_joints(q, body)[:, :, 2]
        per_frame = np.concatenate([
            ((ends - targets) * active[:, :, None] / config['contact_scale_cm']).reshape(count, 8),
            config['reference_angle_weight'] * np.sqrt(weights) *
            2 * np.sin((q[:, 2:] - reference[:, 2:]) / 2),
            config['reference_hip_weight'] * (q[:, :2] - reference[:, :2]) / scales[:2],
            (ends[:, moving_index] - guide) * free[:, None] *
            config['swing_weight'] / config['swing_scale_cm'],
        ], axis=1)
        velocity = config['velocity_weight'] * np.diff(q / scales, axis=0) / dt
        acceleration = config['acceleration_weight'] * np.diff(q / scales, n=2, axis=0) / dt**2
        return np.concatenate([per_frame.ravel(), velocity.ravel(), acceleration.ravel()])

    local_size = 21
    sparsity = lil_matrix((count*local_size + (count-1)*11 + (count-2)*11, count*11), dtype=int)
    for t in range(count):
        sparsity[t*local_size:(t+1)*local_size, t*11:(t+1)*11] = 1
    offset = count * local_size
    for t in range(count-1):
        sparsity[offset+t*11:offset+(t+1)*11, t*11:(t+2)*11] = 1
    offset += (count-1)*11
    for t in range(count-2):
        sparsity[offset+t*11:offset+(t+1)*11, t*11:(t+3)*11] = 1
    solution = least_squares(residual, np.asarray(seeds).ravel(), jac='cs',
        jac_sparsity=sparsity.tocsr(), bounds=(np.tile(lower, count), np.tile(upper, count)),
        x_scale='jac', tr_options={'atol': 1e-10, 'btol': 1e-10, 'maxiter': 1000},
        max_nfev=config['max_nfev'], ftol=1e-7, xtol=1e-7, gtol=1e-7)
    if not solution.success or not np.isfinite(solution.x).all():
        raise RuntimeError(f'Motion optimization did not converge: {solution.message}; '
                           f'cost={solution.cost:.6g}, optimality={solution.optimality:.6g}')
    fitted = solution.x.reshape(count, 11)
    joints = batch_joints(fitted, body)
    contact_error = np.linalg.norm(joints[:, :, 2] - targets, axis=-1)
    max_error = float(contact_error[active].max())
    if max_error > config['contact_tolerance_cm']:
        raise RuntimeError(f'Motion contact error {max_error:.6g} cm exceeds the numerical tolerance '
                           f"{config['contact_tolerance_cm']:g} cm. This does not prove physical impossibility.")
    support_ids = [i for i in range(4) if i != moving_index]
    support_drift = max(float(np.linalg.norm(joints[:, i, 2] - joints[0, i, 2], axis=-1).max())
                        for i in support_ids)
    angle_steps = np.rad2deg(np.abs(np.diff(fitted[:, 2:], axis=0)))
    # Ignore sign noise near a straight joint when reporting changes of bending direction.
    bends = np.rad2deg(fitted[:, 4::2])
    branch_changes = {limb: bool((bends[:, i] > 5).any() and (bends[:, i] < -5).any())
                      for i, limb in enumerate(LIMBS)}
    metrics = dict(max_contact_error_cm=max_error,
        rms_contact_error_cm=float(np.sqrt(np.mean(contact_error[active]**2))),
        max_support_drift_cm=support_drift,
        max_angle_step_deg=float(angle_steps.max()),
        max_hip_step_cm=float(np.linalg.norm(np.diff(fitted[:, :2], axis=0), axis=1).max()),
        bend_direction_changes=branch_changes,
        low_visibility_angle_observations=int((weights < 0.5).sum()),
        angle_observations=int(weights.size),
        optimizer=dict(success=bool(solution.success), message=str(solution.message),
                       nfev=int(solution.nfev), cost=float(solution.cost)),
        settings=config)
    return dict(frames=frames, poses=fitted, reference_poses=reference, weights=weights,
                joints=joints, contacts=labels, active=active, targets=targets,
                contact_errors_cm=contact_error, swing_guide_cm=guide, metrics=metrics,
                body=body)
