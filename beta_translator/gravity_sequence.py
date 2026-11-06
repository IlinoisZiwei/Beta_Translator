"""Joint full-sequence pose/force fitting under a planar quasi-static assumption.

All source frames share one optimization. No inferred ground forces, no stitching,
and no claim that numerical equilibrium establishes physical climbing feasibility.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, diags, kron, vstack

from .full_sequence import DEFAULT_SETTINGS as GEOMETRY_SETTINGS, endpoint_jacobian, diagnose_sequence
from .inverse_kinematics import (
    GRAVITY, MASS_FRACTIONS, CENTER_OF_MASS_FRACTIONS, make_body, make_mass_model,
    validate_com_fractions,
)
from .motion_ik import batch_joints
from .contact_geometry import contact_points, support_ids, validate_contact_geometry


DEFAULT_SETTINGS = {**GEOMETRY_SETTINGS, 'balance_weight': 30., 'force_weight': .05,
    'force_velocity_weight': .02, 'balance_tolerance': .01,
    'force_step_warning_bw': .5}


def batch_com_and_jacobian(poses, lengths, com_fractions=None):
    """COM and exact pose derivatives; optional fractions permit profile sensitivity.

    Sequence optimizers retain the legacy default unless explicitly wired to a
    different profile throughout both optimization and audit.
    """
    q = np.asarray(poses)
    n = len(q)
    joints = batch_joints(q, lengths)
    end_jac = endpoint_jacobian(q, lengths)
    hip = q[:, :2]
    hip_jac = np.zeros((n, 2, 11), dtype=q.dtype)
    hip_jac[:, 0, 0] = hip_jac[:, 1, 1] = 1
    up = np.stack([-np.sin(q[:, 2]), np.cos(q[:, 2])], axis=-1)
    right = np.stack([np.cos(q[:, 2]), np.sin(q[:, 2])], axis=-1)
    f = MASS_FRACTIONS
    c = CENTER_OF_MASS_FRACTIONS if com_fractions is None else com_fractions
    validate_com_fractions(c)
    torso_factor = f['head'] * (1 + c['head']) + f['torso'] * c['torso']
    com = (f['head'] + f['torso']) * hip + torso_factor * lengths['torso'] * up
    jac = (f['head'] + f['torso']) * hip_jac
    jac[:, :, 2] -= torso_factor * lengths['torso'] * right
    for i in range(4):
        upper, lower = ('upper_arm', 'forearm') if i < 2 else ('thigh', 'shank')
        sign = -1 if i in (0, 2) else 1
        offset = lengths['shoulder_w' if i < 2 else 'hip_w'] / 2
        root_jac = hip_jac.copy()
        root_jac[:, :, 2] += sign * offset * up - (lengths['torso'] * right if i < 2 else 0)
        middle_jac = root_jac.copy()
        angle = q[:, 3 + 2*i]
        middle_jac[:, :, 3+2*i] += lengths[upper] * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        root_w = f[upper] * (1 - c[upper])
        middle_w = f[upper] * c[upper] + f[lower] * (1 - c[lower])
        end_w = f[lower] * c[lower]
        com += root_w * joints[:, i, 0] + middle_w * joints[:, i, 1] + end_w * joints[:, i, 2]
        jac += root_w * root_jac + middle_w * middle_jac + end_w * end_jac[:, i]
    return com, jac


class GravityProblem:
    """Local residuals in 19 variables: 11 pose values and 8 force components/BW."""
    def __init__(self, reference, visibility, endpoints, schedule, body, fps,
                 mass_kg, foot_friction, max_force_n, settings):
        self.config = {**DEFAULT_SETTINGS, **(settings or {})}
        if set(self.config) != set(DEFAULT_SETTINGS):
            raise ValueError('Unknown gravity-sequence setting.')
        if (any(not np.isfinite(v) or v <= 0 for v in self.config.values())
                or type(self.config['max_nfev']) is not int or self.config['outside_hull_weight'] > 1):
            raise ValueError('Settings must be finite and positive, with integer max_nfev.')
        make_mass_model(mass_kg)
        if not np.isfinite(foot_friction) or foot_friction < 0:
            raise ValueError('Foot friction must be nonnegative and finite.')
        if max_force_n is not None and (not np.isfinite(max_force_n) or max_force_n <= 0):
            raise ValueError('Force component limit must be positive and finite.')
        self.frames = np.asarray(schedule['frames'])
        n = len(self.frames)
        if (self.frames.dtype.kind not in 'iu' or n < 3
                or not np.all(np.diff(self.frames) == 1)):
            raise ValueError('All consecutive source frames are required.')
        self.reference = np.asarray(reference, float).copy()
        self.visibility = np.asarray(visibility, float)
        self.endpoints = np.asarray(endpoints, float)
        if (self.reference.shape != (n, 11) or self.visibility.shape != (n, 9)
                or self.endpoints.shape != (n, 4, 2)
                or not all(np.isfinite(a).all() for a in (self.reference, self.visibility, self.endpoints))
                or np.any((self.visibility < 0) | (self.visibility > 1))
                or not np.isfinite(fps) or fps <= 0):
            raise ValueError('Invalid reference data, visibility or FPS.')
        for name, shape in [('known', (n, 4)), ('supported', (n, 4)),
                            ('confidence', (n, 4)), ('targets', (n, 4, 2))]:
            a = np.asarray(schedule[name])
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError('Invalid contact schedule.')
        if (np.asarray(schedule['labels']).shape != (n,4)
                or np.asarray(schedule['known']).dtype != bool
                or np.asarray(schedule['supported']).dtype != bool
                or np.any(schedule['supported'] & ~schedule['known'])
                or np.any((schedule['confidence'] < 0) | (schedule['confidence'] > 1))):
            raise ValueError('Invalid contact masks, labels or confidence.')
        validate_contact_geometry(schedule, n)
        self.reference[:, [2,3,5,7,9]] = np.unwrap(self.reference[:, [2,3,5,7,9]], axis=0)
        self.lengths = make_body(body['height_cm'], body['ratios'])
        self.schedule, self.fps = schedule, fps
        self.mass_kg, self.mu = mass_kg, foot_friction
        self.weight_n = mass_kg * GRAVITY
        self.limit_bw = 1. if max_force_n is None else max_force_n / self.weight_n
        self.max_force_n = self.limit_bw * self.weight_n
        c = self.config
        reliability = np.where(schedule['supported'], 1., c['outside_hull_weight'])
        self.contact_weight = np.sqrt(np.maximum(schedule.get('fit_weight', schedule['confidence']), .05) * reliability) * schedule['known'] / c['contact_scale_cm']
        self.angle_weight = c['reference_angle_weight'] * np.sqrt(self.visibility)
        self.free_weight = (~schedule['known']) / c['free_endpoint_scale_cm']
        self.lower, self.upper = np.full(19, -np.inf), np.full(19, np.inf)
        self.lower[[4,6,8,10]], self.upper[[4,6,8,10]] = -np.deg2rad(150), np.deg2rad(150)
        self.lower[11:], self.upper[11:] = -self.limit_bw, self.limit_bw
        self.lower[[16,18]] = 0.  # Foot fy may only push upward in this simplified model.

    def local(self, x, indices):
        q, raw = x[:, :11], x[:, 11:].reshape(-1, 4, 2)
        c, s = self.config, self.schedule
        known = s['known'][indices]
        force = raw * known[:, :, None]  # Unobserved support never receives a force.
        ends = contact_points(batch_joints(q, self.lengths)[:, :, 2], s, indices)
        com, _ = batch_com_and_jacobian(q, self.lengths)
        lever = (ends - com[:, None]) / self.lengths['torso']
        moment = np.sum(lever[:, :, 0] * force[:, :, 1] - lever[:, :, 1] * force[:, :, 0], axis=1)
        friction = np.maximum(0., np.abs(force[:, 2:, 0]) - self.mu * force[:, 2:, 1])
        return np.concatenate([
            ((ends - s['targets'][indices]) * self.contact_weight[indices, :, None]).reshape(-1, 8),
            self.angle_weight[indices] * 2 * np.sin((q[:, 2:] - self.reference[indices, 2:])/2),
            (q[:, :2] - self.reference[indices, :2]) / c['reference_hip_scale_cm'],
            ((ends - self.endpoints[indices]) * self.free_weight[indices, :, None]).reshape(-1, 8),
            c['balance_weight'] * (force.sum(axis=1) - [0., 1.]),
            c['balance_weight'] * moment[:, None], c['balance_weight'] * friction,
            c['force_weight'] * raw.reshape(-1, 8),
        ], axis=1)

    def local_jacobian(self, x, indices):
        q, raw = x[:, :11], x[:, 11:].reshape(-1,4,2)
        n, c, s = len(x), self.config, self.schedule
        known = s['known'][indices]
        force = raw * known[:, :, None]
        ends = contact_points(batch_joints(q, self.lengths)[:, :, 2], s, indices)
        endpoint = endpoint_jacobian(q, self.lengths)
        com, com_jac = batch_com_and_jacobian(q, self.lengths)
        lever = (ends - com[:, None]) / self.lengths['torso']
        lever_jac = (endpoint - com_jac[:, None]) / self.lengths['torso']
        jac = np.zeros((n,40,19))
        jac[:, :8, :11] = (endpoint * self.contact_weight[indices, :, None, None]).reshape(n,8,11)
        jac[:, 8+np.arange(9), 2+np.arange(9)] = self.angle_weight[indices] * np.cos((q[:,2:] - self.reference[indices,2:])/2)
        jac[:,17,0] = jac[:,18,1] = 1 / c['reference_hip_scale_cm']
        jac[:,19:27,:11] = (endpoint * self.free_weight[indices, :, None, None]).reshape(n,8,11)
        jac[:,29,:11] = c['balance_weight'] * np.sum(
            lever_jac[:,:,0,:] * force[:,:,1,None] - lever_jac[:,:,1,:] * force[:,:,0,None], axis=1)
        for i in range(4):
            fx, fy = 11+2*i, 12+2*i
            scale = c['balance_weight'] * known[:,i]
            jac[:,27,fx], jac[:,28,fy] = scale, scale
            jac[:,29,fx], jac[:,29,fy] = -scale * lever[:,i,1], scale * lever[:,i,0]
            if i >= 2:
                active = np.abs(force[:,i,0]) > self.mu * force[:,i,1]
                jac[:,30+i-2,fx] = scale * active * np.sign(force[:,i,0])
                jac[:,30+i-2,fy] = -scale * active * self.mu
        jac[:,32+np.arange(8),11+np.arange(8)] = c['force_weight']
        return jac

    def smooth_matrix(self):
        n, c = len(self.frames), self.config
        first = diags([-np.ones(n-1), np.ones(n-1)], [0,1], shape=(n-1,n))
        second = diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0,1,2], shape=(n-2,n))
        scales = np.array([50.,50.] + [1.]*9 + [np.inf]*8)
        pose_smooth = [kron(first, diags(c['velocity_weight']*self.fps/scales)),
                       kron(second, diags(c['acceleration_weight']*self.fps**2/scales))]
        s = self.schedule
        same = s['known'][1:] & s['known'][:-1] & (support_ids(s)[1:] == support_ids(s)[:-1])
        t, limb = np.where(same)
        columns = ((t[:,None]*19 + 11 + limb[:,None]*2) + np.arange(2)).ravel()
        row = np.arange(len(columns))
        weight = c['force_velocity_weight'] * self.fps
        force_smooth = coo_matrix((np.r_[np.full(len(row),-weight), np.full(len(row),weight)],
            (np.r_[row,row], np.r_[columns, columns+19])), shape=(len(row),n*19))
        return vstack([*pose_smooth, force_smooth], format='csr')


def solve_gravity_sequence(reference, visibility, reference_endpoints, schedule, body, fps,
                           mass_kg=70., foot_friction=.8, max_force_n=None, settings=None,
                           progress=None):
    p = GravityProblem(reference, visibility, reference_endpoints, schedule, body, fps,
                       mass_kg, foot_friction, max_force_n, settings)
    n, config = len(p.frames), p.config
    seed = np.r_[np.clip(p.reference[0],p.lower[:11],p.upper[:11]), np.zeros(8)]
    seeds = []
    scales = np.array([50.,50.] + [1.]*9 + [10.]*8)
    for t in range(n):
        known = schedule['known'][t]
        force = np.zeros((4,2))
        force[known,1] = min(1. / max(known.sum(),1), p.limit_bw)
        seed[11:] = force.ravel()
        previous = seed.copy()
        fit = least_squares(lambda x: np.r_[p.local(x[None],slice(t,t+1)).ravel(), .2*(x-previous)/scales],
            seed, jac=lambda x: np.vstack([p.local_jacobian(x[None],slice(t,t+1))[0],np.diag(.2/scales)]),
            bounds=(p.lower,p.upper), max_nfev=100, ftol=1e-6)
        if not np.isfinite(fit.x).all():
            raise RuntimeError(f'Nonfinite initializer at frame {p.frames[t]}.')
        seed = fit.x
        seeds.append(seed.copy())
        if progress and (t % 100 == 0 or t == n-1):
            progress(f'Initialized pose/forces: {t+1}/{n} frames')
    smooth = p.smooth_matrix()
    row = np.broadcast_to(np.arange(n*40).reshape(n,40,1),(n,40,19)).ravel()
    col = np.broadcast_to(np.arange(n)[:,None,None]*19 + np.arange(19),(n,40,19)).ravel()
    evaluations = [0]
    def residual(flat):
        evaluations[0] += 1
        if progress and evaluations[0] % 25 == 0:
            progress(f'Global gravity evaluation {evaluations[0]}: {n} poses and contact forces together')
        return np.r_[p.local(flat.reshape(n,19),slice(None)).ravel(), smooth @ flat]
    def jacobian(flat):
        data = p.local_jacobian(flat.reshape(n,19),slice(None)).ravel()
        local = coo_matrix((data,(row,col)),shape=(n*40,n*19)).tocsr()
        local.eliminate_zeros()
        return vstack([local,smooth],format='csr')
    solution = least_squares(residual,np.asarray(seeds).ravel(),jac=jacobian,
        bounds=(np.tile(p.lower,n),np.tile(p.upper,n)),x_scale='jac',
        tr_options={'atol':1e-7,'btol':1e-7,'maxiter':500},
        max_nfev=config['max_nfev'],ftol=1e-6,xtol=1e-7,gtol=1e-6)
    if not np.isfinite(solution.x).all():
        raise RuntimeError('Nonfinite gravity trajectory; no export is possible.')
    x = solution.x.reshape(n,19)
    forces = x[:,11:].reshape(n,4,2) * schedule['known'][:,:,None] * p.weight_n
    com, _ = batch_com_and_jacobian(x[:,:11],p.lengths)
    return dict(frames=p.frames,poses=x[:,:11],joints=batch_joints(x[:,:11],p.lengths),
        reference=p.reference,visibility=p.visibility,reference_endpoints=p.endpoints,
        schedule=schedule,body=body,lengths=p.lengths,settings=config,
        contact_forces_n=forces,center_of_mass_cm=com,mass_kg=float(mass_kg),
        foot_friction=float(foot_friction),max_force_component_n=float(p.max_force_n),
        optimizer=dict(success=bool(solution.success),message=str(solution.message),nfev=int(solution.nfev),
                       cost=float(solution.cost),optimality=float(solution.optimality)),
        method='One joint optimization of all source-frame poses and contact forces, with temporal regularization; quasi-static 2D balance.')


def diagnose_gravity_sequence(result, reference_outside=None):
    d = diagnose_sequence(result,reference_outside)
    forces, com = result['contact_forces_n'], result['center_of_mass_cm']
    s, c = result['schedule'], result['settings']
    weight = result['mass_kg'] * GRAVITY
    ends = contact_points(result['joints'][:,:,2], s)
    net = forces.sum(axis=1) - [0.,weight]
    lever = ends - com[:,None]
    moment = np.sum(lever[:,:,0]*forces[:,:,1] - lever[:,:,1]*forces[:,:,0],axis=1)
    friction = np.maximum(0.,np.abs(forces[:,2:,0])-result['foot_friction']*forces[:,2:,1])
    tolerance = c['balance_tolerance'] * weight
    force_norm = np.linalg.norm(net,axis=1)
    balance = (force_norm <= tolerance) & (np.abs(moment) <= tolerance*result['lengths']['torso']) & (friction.max(axis=1) <= tolerance)
    force_steps = np.r_[0.,np.linalg.norm(np.diff(forces,axis=0),axis=2).max(axis=1)/weight]
    force_on_gap = np.any((d['contact_errors_cm'] > c['contact_tolerance_cm']) & (np.linalg.norm(forces,axis=2) > tolerance),axis=1)
    d['flags'].update(
        force_imbalance=force_norm > tolerance,
        moment_imbalance=np.abs(moment) > tolerance*result['lengths']['torso'],
        foot_friction_violation=friction.max(axis=1) > tolerance,
        no_known_support=~s['known'].any(axis=1),
        force_at_missed_contact=force_on_gap,
        force_on_uncertain_contact=np.any((s['known'] & (~s['supported'] | s['low_confidence'])) & (np.linalg.norm(forces,axis=2)>tolerance),axis=1),
        rapid_force_change=force_steps > c['force_step_warning_bw'],
    )
    d['issues'] = [';'.join(k for k,v in d['flags'].items() if v[t]) for t in range(len(com))]
    numerical = (d['max_contact_error_cm'] <= c['contact_tolerance_cm']) & balance
    numerical &= ~d['flags']['optimizer_unconverged'] & ~d['flags']['span_conflict']
    d['status'] = np.where(~numerical,'constraint_warning',
        np.where([bool(s) for s in d['issues']],'uncertain','within_numeric_tolerance'))
    d['counts'] = {k:int(v.sum()) for k,v in d['flags'].items()}
    d.update(net_force_n=net,net_force_norm_n=force_norm,net_moment_nm=moment/100.,
        foot_friction_excess_n=friction,balanced=balance,numerical_checks_pass=numerical,
        force_step_bw=force_steps)
    return d
