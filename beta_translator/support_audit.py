"""Fixed-pose, quasi-static support audits; never a physical-success certificate.

Forces are solved at located hold centres, independently of the candidate IK
forces. Confidence/calibration eligibility is a model filter, not ground truth.
"""
import numpy as np
from scipy.optimize import linprog

from .inverse_kinematics import GRAVITY, LIMBS
from .contact_geometry import support_ids

DEFAULT_AUDIT_SETTINGS = dict(com_perturbation_cm=2., margin_warning=.1,
                              unload_lookahead_frames=6)


def audit_settings(settings=None):
    c = {**DEFAULT_AUDIT_SETTINGS, **(settings or {})}
    if (set(c) != set(DEFAULT_AUDIT_SETTINGS)
            or not np.isfinite(c['com_perturbation_cm']) or c['com_perturbation_cm'] < 0
            or not np.isfinite(c['margin_warning']) or not 0 <= c['margin_warning'] < 1
            or type(c['unload_lookahead_frames']) is not int or c['unload_lookahead_frames'] < 1):
        raise ValueError('Invalid support audit settings.')
    return c


def force_reserve(targets, active, com, weight_n, limit_n, mu, torso_cm):
    """Minimize peak component utilization rho; reserve = 1-rho.

    rho is unbounded above, so negative reserve quantifies a capacity deficit.
    Structural infeasibility and solver failure have no numeric reserve.
    Feet push upward with |fx| <= mu*fy; hands have arbitrary planar forces.
    """
    ids = np.flatnonzero(active)
    answer = dict(status='no_support', margin=None, utilization=None,
                  forces_n=np.zeros((4, 2)), solver_message='No eligible located contacts.')
    if not len(ids):
        return answer
    data = np.r_[np.asarray(targets)[ids].ravel(), com, weight_n, limit_n, mu, torso_cm]
    if not np.isfinite(data).all() or min(weight_n, limit_n, torso_cm) <= 0 or mu < 0:
        return {**answer, 'status':'not_evaluated', 'solver_message':'Invalid audit geometry or physics.'}
    m = len(ids)
    eq = np.zeros((3, 2*m+1))
    eq[0, 0:2*m:2] = 1.; eq[1, 1:2*m:2] = 1.
    lever = (np.asarray(targets)[ids] - com) / torso_cm
    eq[2, 0:2*m:2] = -lever[:, 1]; eq[2, 1:2*m:2] = lever[:, 0]
    rows = []
    bounds = []
    for k, limb in enumerate(ids):
        for j in (2*k, 2*k+1):
            for sign in (-1., 1.):
                row = np.zeros(2*m+1); row[j] = sign; row[-1] = -limit_n/weight_n
                rows.append(row)
        bounds.extend([(None, None), (0., None) if limb >= 2 else (None, None)])
        if limb >= 2:
            for sign in (-1., 1.):
                row = np.zeros(2*m+1); row[2*k] = sign; row[2*k+1] = -mu
                rows.append(row)
    bounds.append((0., None))
    ub = np.asarray(rows)
    objective = np.zeros(2*m+1); objective[-1] = 1.
    try:
        fit = linprog(objective, A_ub=ub, b_ub=np.zeros(len(rows)),
                      A_eq=eq, b_eq=[0., 1., 0.], bounds=bounds, method='highs')
    except (ValueError, RuntimeError) as exc:
        return {**answer, 'status':'not_evaluated', 'solver_message':str(exc)}
    answer['solver_message'] = str(fit.message)
    if fit.status == 2:
        return {**answer, 'status':'structurally_infeasible'}
    if not fit.success or not np.isfinite(fit.x).all():
        return {**answer, 'status':'not_evaluated'}
    if (np.max(np.abs(eq @ fit.x - [0., 1., 0.])) > 1e-7
            or np.max(ub @ fit.x) > 1e-7
            or any(fit.x[2*k+1] < -1e-7 for k, i in enumerate(ids) if i >= 2)):
        return {**answer, 'status':'not_evaluated', 'solver_message':'LP residual validation failed.'}
    answer['forces_n'][ids] = fit.x[:-1].reshape(m, 2)*weight_n
    rho = float(fit.x[-1])
    return {**answer, 'status':'feasible' if rho <= 1.+1e-7 else 'capacity_exceeded',
            'utilization':rho, 'margin':1.-rho}


def reserve_sensitivity(result, t, active, config):
    """Nominal and horizontal COM perturbations; perturbation is an assumption."""
    trials = []
    for dx in (0., -config['com_perturbation_cm'], config['com_perturbation_cm']):
        trials.append(force_reserve(result['schedule']['targets'][t], active,
            result['center_of_mass_cm'][t] + [dx, 0.], result['mass_kg']*GRAVITY,
            result['max_force_component_n'], result['foot_friction'], result['lengths']['torso']))
    margins = [v['margin'] for v in trials]
    return dict(nominal=trials[0], perturbed=trials[1:],
                worst_margin=min(margins) if all(x is not None for x in margins) else None,
                robust=all(v['status'] == 'feasible' for v in trials),
                evaluated=all(v['status'] != 'not_evaluated' for v in trials))


def classify_balance(eligible, conditional, warning):
    # Failure of the superset establishes failure of all located usable support.
    if conditional['nominal']['status'] == 'not_evaluated':
        return 'not_evaluated'
    if conditional['nominal']['status'] != 'feasible':
        return 'fails_located_support_model'
    if eligible['nominal']['status'] == 'not_evaluated':
        return 'not_evaluated'
    if eligible['nominal']['status'] != 'feasible':
        return 'uncertain_support'
    if not eligible['evaluated']:
        return 'not_evaluated'
    if not eligible['robust'] or eligible['worst_margin'] < warning:
        return 'low_margin'
    return 'passes_model_checks'


def release_events(schedule, frames):
    """Group simultaneous departures; direct jumps do not invent free frames."""
    ids = support_ids(schedule)
    changed = schedule['known'][:-1] & ((ids[:-1] != ids[1:]) | ~schedule['known'][1:])
    events = []
    for previous in np.flatnonzero(changed.any(axis=1)):
        moving = np.flatnonzero(changed[previous]).tolist()
        t = int(previous+1)
        direct = bool(schedule['known'][t, moving].any())
        unknown = any(str(schedule['labels'][t, i]) not in ('free',) and not schedule['known'][t, i] for i in moving)
        ground_departure = any(str(ids[previous,i]).startswith('ground_') for i in moving)
        ground_uncertain = bool(schedule.get('ground_timing_uncertain', np.zeros_like(schedule['known']))[t,moving].any())
        events.append(dict(event_id=f'R{len(events)+1:03d}', release_index=t,
            release_frame=int(frames[t]), moving_indices=moving,
            moving_limbs=';'.join(LIMBS[i] for i in moving),
            from_labels=';'.join(map(str,ids[previous,moving])),
            to_labels=';'.join(map(str,np.where(ids[t,moving] != '', ids[t,moving], schedule['labels'][t,moving]))),
            timing_uncertain=direct or unknown or ground_uncertain,
            timing_reason=('ground_observation_ends_boundary_uncertain' if ground_uncertain else
                ('reviewed_ground_liftoff' if ground_departure and not direct and not unknown else
                 ('direct_hold_change' if direct else ('unlocated_destination' if unknown else 'predicted_first_free_frame'))))))
    return events


def _flat_reserve(prefix, v):
    return {f'{prefix}_status':v['nominal']['status'], f'{prefix}_margin':v['nominal']['margin'],
            f'{prefix}_worst_margin':v['worst_margin'], f'{prefix}_robust':v['robust'],
            f'{prefix}_minus_status':v['perturbed'][0]['status'],
            f'{prefix}_plus_status':v['perturbed'][1]['status'],
            f'{prefix}_solver_message':v['nominal']['solver_message']}


def audit_support_sequence(result, diagnostic, settings=None):
    c = audit_settings(settings)
    s = result['schedule']; n = len(result['frames'])
    conditional = s['known'] & np.isfinite(diagnostic['contact_errors_cm']) & (diagnostic['contact_errors_cm'] <= result['settings']['contact_tolerance_cm'])
    eligible = conditional & s['supported'] & ~s['low_confidence']
    state = np.full((n,4), 'no_located_support', dtype='<U40')
    state[s['known']] = 'invalid_gap'
    state[conditional] = 'eligible_under_model'
    state[conditional & s['low_confidence']] = 'uncertain_confidence'
    state[conditional & ~s['supported']] = 'uncertain_geometry'
    if 'contact_type' in s:
        ground = s['contact_type'] == 'ground'
        state[conditional & ground] = 'conditional_ground_geometry'
        candidate = (s['ground_evidence'] == 'ground_candidate') & ~s['known']
        state[candidate] = 'ground_candidate_no_force'
    rows = []; forces = {k:np.zeros((n,4,2)) for k in ('eligible','conditional')}
    for t, frame in enumerate(result['frames']):
        a = reserve_sensitivity(result,t,eligible[t],c)
        b = a if np.array_equal(eligible[t],conditional[t]) else reserve_sensitivity(result,t,conditional[t],c)
        status = classify_balance(a,b,c['margin_warning'])
        rows.append(dict(frame=int(frame), eligible_support_count=int(eligible[t].sum()),
            conditional_support_count=int(conditional[t].sum()), balance_audit_status=status,
            **_flat_reserve('eligible',a), **_flat_reserve('conditional',b)))
        forces['eligible'][t] = a['nominal']['forces_n']; forces['conditional'][t] = b['nominal']['forces_n']
    events = release_events(s,result['frames']); unload_rows = []
    severity = {'not_applicable':0,'passes_model_checks':1,'low_margin':2,'uncertain_support':3,
                'uncertain_timing':3,'not_evaluated':4,'fails_located_support_model':5}
    unloading = np.full(n,'not_applicable',dtype='<U40'); frame_events = [[] for _ in range(n)]
    for event in events:
        release = event['release_index']; moving = event['moving_indices']
        preframes = []
        for t in range(release-1,max(-1,release-c['unload_lookahead_frames']-1),-1):
            if not np.array_equal(support_ids(s)[t,moving],support_ids(s)[release-1,moving]):break
            preframes.append(t)
        for t in sorted(preframes):
            ea = eligible[t].copy(); ca = conditional[t].copy()
            ea[moving] = False; ca[moving] = False
            a = reserve_sensitivity(result,t,ea,c)
            b = a if np.array_equal(ea,ca) else reserve_sensitivity(result,t,ca,c)
            support_status = classify_balance(a,b,c['margin_warning'])
            status = 'uncertain_timing' if event['timing_uncertain'] and support_status in ('passes_model_checks','low_margin') else support_status
            unload_rows.append(dict(event_id=event['event_id'],frame=int(result['frames'][t]),
                release_frame=event['release_frame'],moving_limbs=event['moving_limbs'],
                from_labels=event['from_labels'],to_labels=event['to_labels'],
                timing_uncertain=event['timing_uncertain'],timing_reason=event['timing_reason'],
                status=status,support_status=support_status,
                remaining_eligible_count=int(ea.sum()),remaining_conditional_count=int(ca.sum()),
                remaining_eligible_limbs=';'.join(LIMBS[i] for i in np.flatnonzero(ea)),
                remaining_conditional_limbs=';'.join(LIMBS[i] for i in np.flatnonzero(ca)),
                departing_candidate_load_bw=float(np.linalg.norm(result['contact_forces_n'][t,moving],axis=1).sum()/(result['mass_kg']*GRAVITY)),
                **_flat_reserve('eligible',a), **_flat_reserve('conditional',b)))
            frame_events[t].append(event['event_id'])
            if severity[status] > severity[unloading[t]]:unloading[t] = status
    for t,row in enumerate(rows):
        row.update(unload_audit_status=str(unloading[t]),upcoming_release_events=';'.join(frame_events[t]))
    return dict(settings=c,eligible=eligible,conditional=conditional,contact_state=state,
                frames=rows,events=events,unloading=unload_rows,forces_n=forces)


def attach_audit(diagnostic, audit):
    """Preserve old flags and add audit outcomes without laundering uncertainty."""
    d = diagnostic
    balance = np.array([r['balance_audit_status'] for r in audit['frames']])
    unload = np.array([r['unload_audit_status'] for r in audit['frames']])
    d['legacy_numerical_checks_pass'] = d['numerical_checks_pass'].copy()
    uncertain_timing_frames = {r['frame'] for r in audit['unloading'] if r['timing_uncertain']}
    d['flags'].update(
        support_balance_failed=balance == 'fails_located_support_model',
        support_balance_uncertain=balance == 'uncertain_support',
        support_margin_low=balance == 'low_margin',
        preunload_failed=unload == 'fails_located_support_model',
        preunload_uncertain=np.isin(unload,['uncertain_support','uncertain_timing']),
        preunload_timing_uncertain=np.array([r['frame'] in uncertain_timing_frames for r in audit['frames']]),
        preunload_margin_low=unload == 'low_margin',
        support_audit_not_evaluated=(balance == 'not_evaluated') | (unload == 'not_evaluated'))
    d['numerical_checks_pass'] &= (balance == 'passes_model_checks') & np.isin(unload,['passes_model_checks','not_applicable'])
    fail = (~d['legacy_numerical_checks_pass'] | d['flags']['support_balance_failed'] | d['flags']['preunload_failed'])
    d['issues'] = [';'.join(k for k,v in d['flags'].items() if v[t]) for t in range(len(balance))]
    d['status'] = np.where(fail,'constraint_warning',np.where([bool(x) for x in d['issues']],'uncertain','within_numeric_tolerance'))
    d['counts'] = {k:int(v.sum()) for k,v in d['flags'].items()}
    d['support_audit'] = audit
    return d
