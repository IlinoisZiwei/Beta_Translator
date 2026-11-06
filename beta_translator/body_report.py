"""Common constraint outcomes and candidate selection, independent of a trial."""
import numpy as np
from .body_constraints import CONSTRAINTS, DEFAULT_LIMITS, require_complete
from .body_optimizer import selection_row


def constraint_results(fit, metrics, data, scene, regions, extra_checks=None):
    m=metrics;L=DEFAULT_LIMITS;known=data['known'];qdim=fit['anatomical_q'].shape[1];forces=fit['values'][:,qdim:qdim+12].reshape(-1,4,3)
    departing=data['unload']>=1.-1e-9
    load=np.linalg.norm(forces,axis=-1)
    release_failed=np.any(departing&(load>L.departing_force_bw),axis=1)
    m['departing_force_failed']=release_failed
    m['departing_force_bw']=np.max(np.where(departing,load,0),axis=1)
    m.loc[release_failed,'issues']=m.loc[release_failed,'issues'].astype(str)+';departing_force_remaining'
    uv=fit['episode_uv'];in_region=bool(np.all(uv>=-1e-7) and np.all(uv<=1+1e-7))
    checks={
        'fixed_geometry':not bool((m.bone_error_cm>L.bone_cm).any()),
        'arm_rom':not bool(m.anatomy_bounds_failed.any()),
        **({'lower_limb_rom':not bool(m.lower_limb_rom_failed.any()),'lower_limb_chain':True} if qdim==32 else {}),
        'coordinate_rates':not bool((m.rate_failed|m.angular_acceleration_failed).any()),
        'shared_contact_regions':in_region,
        'physical_patch_contact':not bool(m.contact_failed.any()),
        'free_contact_force':bool(np.all(abs(forces[~known])<1e-10)),
        'friction_and_support':bool(m.conditional_balance.all()),
        'release':not bool(m.preunload.eq(False).any() or release_failed.any()),
        'trusted_support':bool(m.eligible_balance.all()),
        'balance_reserve':bool(m.robust_2cm.all()),
        'provenance':True,
        'candidate_selection':True,
    }
    rows=[]
    for c in CONSTRAINTS:
        status=('pass' if checks[c.id] else 'fail') if c.id in checks else 'unknown'
        if c.id=='release' and not np.any(data['unload']):status='not_applicable'
        rows.append(dict(id=c.id,specification_section=c.section,enforcement=c.enforcement,status=status,
                         reason=('Quasi-static perturbation diagnostic only; dynamic reserve is not certified' if c.id=='balance_reserve' and 'advanced_checks' in fit else c.missing or c.evidence)))
    # Optional advanced audit results do not erase unimplemented mechanisms.
    extras={r['id']:r for r in fit.get('advanced_checks',[])}
    if extra_checks is not None:extras.update({r['id']:r for r in extra_checks})
    for row in rows:
        if row['id'] in extras:
            if row['enforcement'] not in ('parameterized_audit','profile_soft_and_audit'):raise ValueError('Cannot override core/model coverage with an external pass')
            row.update(extras[row['id']])
    return rows


def candidate_row(name, fit, metrics, results, fps):
    row=selection_row(name,metrics,fit['solver'])
    extra=metrics.departing_force_failed
    if 'advanced_failure' in metrics:
        extra=extra|metrics.advanced_failure
        for col in ['shoulder_envelope','self_collision','torque_capacity','fatigue','dynamic_balance']:
            row[col+'_failure_frames']=int(metrics[col+'_failed'].sum())
        row['advanced_failure_frames']=int(metrics.advanced_failure.sum())
        row['worst_failure_fraction']=max(row['worst_failure_fraction'],float(metrics.advanced_failure.mean()))
        row['total_failed_frames']=int((metrics.advanced_failure|metrics.anatomy_bounds_failed|metrics.rate_failed|metrics.angular_acceleration_failed|metrics.geometric_failed|~metrics.conditional_balance|metrics.preunload.eq(False)|metrics.departing_force_failed|metrics.grip_proxy_failed|metrics.motion_warning).sum())
    row['departing_force_failure_frames']=int(metrics.departing_force_failed.sum())
    row['numerically_feasible']=bool(row['numerically_feasible'] and not extra.any())
    acceleration=np.diff(fit['joints'],n=2,axis=0)*fps**2
    row['joint_acceleration_rms_cm_s2']=float(np.sqrt(np.mean(acceleration**2)))
    row['specification_status']=require_complete(results)
    row['trusted_support_failure_frames']=int((~metrics.eligible_balance).sum())
    row['reserve_failure_frames']=int((~metrics.robust_2cm).sum())
    applicable=metrics.preunload.notna()
    row['preunload_applicable_frames']=int(applicable.sum())
    row['preunload_failure_fraction_applicable']=float(metrics.preunload.eq(False).sum()/max(1,applicable.sum()))
    row['optimized_hand_load_integral_bw_s']=float(np.linalg.norm(fit['values'][:,fit['anatomical_q'].shape[1]:fit['anatomical_q'].shape[1]+6].reshape(-1,2,3),axis=-1).sum()/fps)
    if 'lower_limb_rom_failed' in metrics:
        row['lower_limb_rom_failure_frames']=int(metrics.lower_limb_rom_failed.sum())
        row['max_knee_coupling_displacement_cm']=float(metrics.knee_coupling_displacement_cm.max())
    row['unknown_constraints']=sum(r['status']=='unknown' for r in results)
    row['uncertainty']='Reduced anatomy; inferred hand/foot and scene geometry; full specification coverage is explicit in constraints.csv.'
    return row


def rank_candidates(rows):
    """Do not let sub-tolerance contact noise outweigh a much smoother plan."""
    def key(r):
        return (not r['numerically_feasible'],r['anatomy_temporal_failure_frames'],
                r['worst_failure_fraction'],r.get('advanced_failure_frames',0),r['total_failed_frames'],r['departing_force_failure_frames'],
                r.get('trusted_support_failure_frames',0),r.get('reserve_failure_frames',0),
                r['unknown_constraints'],not r['converged'],
                max(0.,r['max_contact_gap_cm']-DEFAULT_LIMITS.contact_cm),
                r.get('optimized_hand_load_integral_bw_s',0.) if r['numerically_feasible'] else 0.,
                r['joint_acceleration_rms_cm_s2'])
    names=['numerical gate','anatomy/rates','worst failure fraction','active mechanics failures','failed frames','departing force',
           'trusted support','support reserve','unknown coverage','convergence','contact excess above tolerance','hand load among feasible candidates','Cartesian acceleration']
    ranked=sorted(rows,key=key);winner=ranked[0]
    for i,row in enumerate(ranked):
        row['rank']=i+1
        row['selection_status']='conditional_numerical_pass' if row['numerically_feasible'] else 'diagnostic_only_not_feasible'
        if i==0:row['selection_reason']='Selected among complete tested candidates; required failures precede motion, and unknown physiology remains unknown.'
        else:
            k=next((i for i,(a,b) in enumerate(zip(key(row),key(winner))) if a!=b),None)
            row['selection_reason']='Deterministic tie.' if k is None else f'First difference: {names[k]} ({key(row)[k]} versus {key(winner)[k]}).'
    return ranked
