"""Complete Trial3 contact-geometry candidate comparison, with preserved history."""
from pathlib import Path
import hashlib,json,time
import cv2,numpy as np,pandas as pd
from .trial3_3d_pipeline import load_inputs,identity,write_json,verified_experiment,BASELINE
from .contact_regions3d import build_regions,contacts_from_parameters,homography
from .region_ik3d import RegionProblem,PROFILES
from .ik3d import force_feasible
from .scene3d import project
from .inverse_kinematics import make_body,LIMBS,RATIOS
from .project_io import file_sha256


def audit_regions(result,data,regions,tilt_deg,height_cm=180.,ratios=None):
    """Wrench acts at surface contact, while bone lengths end at wrist/ankle."""
    j=result['joints'];c=result['contacts'];off=result['offsets'];n=len(j)
    gaps=np.linalg.norm(j[:,:,2]-off-c,axis=-1)
    pixels,z=project(j,data['K'],data['R'],data['t'])
    lengths=make_body(height_cm,ratios)
    expected=np.array([[lengths['upper_arm'],lengths['forearm']]]*2+[[lengths['thigh'],lengths['shank']]]*2)
    segment=np.linalg.norm(np.diff(j,axis=2),axis=-1)
    ref_px=homography(c[...,:2],np.linalg.inv(regions['H']))
    polygon=np.asarray(data['old']['calibration']['plane_polygon_xy'],np.float32)
    rows=[];limb_rows=[]
    step=np.r_[0.,np.linalg.norm(np.diff(j,axis=0),axis=-1).max(axis=(1,2))]
    acceleration=np.r_[0.,0.,np.linalg.norm(np.diff(j,n=2,axis=0),axis=-1).max(axis=(1,2))] if n>2 else np.zeros(n)
    for t,frame in enumerate(data['frames']):
        known=data['known'][t];inside=np.ones(4,bool);within_grid=np.zeros(4,bool)
        for i in range(4):
            if not known[i]:continue
            hold=str(data['labels'][t,i]);mask=data['old']['masks'][hold]
            x,y=ref_px[t,i];xx,yy=int(round(x)),int(round(y))
            inside[i]=bool(0<=xx<mask.shape[1] and 0<=yy<mask.shape[0] and mask[yy,xx])
            within_grid[i]=cv2.pointPolygonTest(polygon,(float(x),float(y)),False)>=0
            ep=regions['episodes'][regions['mapping'][t,i]]
            limb_rows.append(dict(frame=int(frame),limb=LIMBS[i],hold=hold,episode=ep['episode'],
                contact_gap_cm=float(gaps[t,i]),inside_mask=bool(inside[i]),within_grid=bool(within_grid[i]),
                offset_length_cm=float(np.linalg.norm(off[t,i])),offset_measured=False,
                evidence_frames=ep['evidence_frames'],expert_confirmed=False,
                contact_x_cm=float(c[t,i,0]),contact_y_cm=float(c[t,i,1]),contact_z_cm=float(c[t,i,2]),
                wrist_ankle_x_cm=float(j[t,i,2,0]),wrist_ankle_y_cm=float(j[t,i,2,1]),wrist_ankle_z_cm=float(j[t,i,2,2])))
        matched=known&(gaps[t]<=.1)&inside
        eligible=matched&data['supported'][t]&within_grid&~data['low_confidence'][t]
        balance=force_feasible(c[t],result['com'][t],matched,tilt_deg)
        eligible_balance=force_feasible(c[t],result['com'][t],eligible,tilt_deg)
        up_balance=force_feasible(c[t],result['com'][t],matched,tilt_deg,foot_normal='world_up')
        robust=balance and all(force_feasible(c[t],result['com'][t]+v,matched,tilt_deg)
            for v in ([2,0,0],[-2,0,0],[0,0,2],[0,0,-2]))
        departing=data['unload'][t]>0
        unload=force_feasible(c[t],result['com'][t],matched&~departing,tilt_deg) if departing.any() else None
        cosine=((j[t,:,1]-j[t,:,0])*(j[t,:,2]-j[t,:,1])).sum(-1)/np.prod(expected,axis=1)
        flex=float(np.rad2deg(np.arccos(np.clip(cosine,-1,1))).max())
        offset_max=np.array([.06,.06,.10,.10])*height_cm
        offset_fail=bool((np.linalg.norm(off[t],axis=1)>offset_max+1e-9).any())
        issues=[]
        if (known&~matched).any():issues.append('contact_failed')
        if not inside[known].all():issues.append('outside_mask')
        if j[t,:,:,2].min()<-.1:issues.append('wall_penetration')
        if flex>155.1:issues.append('bend_limit')
        if step[t]>8:issues.append('joint_step_over_8cm')
        if acceleration[t]>4:issues.append('joint_acceleration_over_4cm_per_frame2')
        if offset_fail:issues.append('offset_limit')
        if not balance:issues.append('conditional_balance_failed')
        if unload is False:issues.append('conditional_preunload_failed')
        if not result['solver']['success']:issues.append('optimizer_unconverged')
        if np.asarray(result.get('direct_switch',np.zeros((n,4),bool)))[t].any():issues.append('direct_hold_switch_timing_unreviewed')
        rows.append(dict(frame=int(frame),max_contact_gap_cm=float(gaps[t,known].max()) if known.any() else 0.,
            missed_contacts=int((known&~matched).sum()),outside_mask_contacts=int((known&~inside).sum()),
            max_segment_error_cm=float(abs(segment[t]-expected).max()),
            min_wall_clearance_cm=float(j[t,:,:,2].min()),max_joint_flex_deg=flex,
            joint_step_cm=float(step[t]),joint_acceleration_cm_per_frame2=float(acceleration[t]),
            hip_depth_cm=float(result['values'][t,2]),source_guide_rmse_px=float(np.sqrt(np.mean((pixels[t]-data['observed_px'][t])**2))),
            conditional_balance=balance,eligible_balance=eligible_balance,conditional_balance_upward_footholds=up_balance,
            conditional_robust_2cm=robust,preunload_without_departing=unload,preunload_timing_uncertain=bool(departing.any()),
            low_confidence_contacts=int((known&data['low_confidence'][t]).sum()),
            extrapolated_contacts=int((known&~within_grid).sum()),
            optimizer_converged=bool(result['solver']['success']),offset_limit_failed=offset_fail,
            all_geometry_uncertain=True,contact_region_unreviewed=True,offsets_unmeasured=True,
            status='constraint_warning' if issues else 'assumption_uncertain',issues=';'.join(issues)))
    return pd.DataFrame(rows),pd.DataFrame(limb_rows),gaps


def summarize(profile,fit,metrics,regions):
    active=regions['mapping']>=0;c=fit['contacts'];prior=np.array([e['center_cm'] for e in regions['episodes']])[np.maximum(regions['mapping'],0)]
    geometric=(metrics.missed_contacts>0)|(metrics.min_wall_clearance_cm<-.1)|(metrics.max_joint_flex_deg>155.1)|metrics.offset_limit_failed
    temporal=(metrics.joint_step_cm>8)|(metrics.joint_acceleration_cm_per_frame2>4)
    return dict(profile=profile,frames=len(metrics),converged=bool(fit['solver']['success']),
        projected_gradient=float(fit['solver']['optimality']),solve_seconds=float(fit['solver']['seconds']),
        geometric_failure_frames=int(geometric.sum()),contact_failure_frames=int((metrics.missed_contacts>0).sum()),
        max_contact_gap_cm=float(metrics.max_contact_gap_cm.max()),mask_failure_frames=int((metrics.outside_mask_contacts>0).sum()),
        temporal_warning_frames=int(temporal.sum()),max_joint_step_cm=float(metrics.joint_step_cm.max()),
        max_segment_error_cm=float(metrics.max_segment_error_cm.max()),
        conditional_balance_frames=int(metrics.conditional_balance.sum()),eligible_balance_frames=int(metrics.eligible_balance.sum()),
        conditional_robust_frames=int(metrics.conditional_robust_2cm.sum()),
        preunload_failure_frames=int((metrics.preunload_without_departing==False).sum()),
        preunload_checked_frames=int(metrics.preunload_without_departing.notna().sum()),
        mean_center_shift_cm=float(np.linalg.norm(c-prior,axis=-1)[active].mean()),
        mean_offset_cm=float(np.linalg.norm(fit['offsets'],axis=-1)[active].mean()),
        evidence_uncertainty_frames=len(metrics),tilt_measured=False)


def rank_candidates(rows):
    """Select the least-failing whole candidate; never call a failing winner valid."""
    fields=['geometric_failure_frames','temporal_warning_frames','not_converged',
            'balance_failure_frames','preunload_failure_frames','not_robust_frames',
            'max_contact_gap_cm','mean_offset_cm']
    records=[]
    for row in rows:
        x=dict(row);x['not_converged']=int(not x['converged']);x['balance_failure_frames']=x['frames']-x['conditional_balance_frames']
        x['not_robust_frames']=x['frames']-x['conditional_robust_frames'];records.append(x)
    ranked=sorted(records,key=lambda r:tuple(r[k] for k in fields))
    best=ranked[0]
    for rank,r in enumerate(ranked,1):
        r['rank']=rank
        if rank==1:r['selection_reason']='Selected as the least-failing complete trajectory under the declared ranking; not a physically validated route.'
        else:
            diff=next((k for k in fields if r[k]!=best[k]),None)
            r['selection_reason']=(f'First ranking difference: {diff} = {r[diff]:.6g}; selected = {best[diff]:.6g}.' if diff else 'Numerically tied; stable profile order is the tie break.')
    return ranked,fields


def region_identity(project,config):
    p=Path(project);x=identity(p,config)
    for name in ['contact_regions3d.py','region_ik3d.py','trial3_region_pipeline.py','region3d_render.py']:
        x['inputs_sha256']['beta_translator/'+name]=file_sha256(p/'beta_translator'/name)
    for name in ['outputs/trial3/automatic/wholebody.npz','outputs/trial3/state/alignment_cache.npz']:
        x['inputs_sha256'][name]=file_sha256(p/name)
    return x


def run_region_experiment(project,height_cm=180.,tilt_deg=15.,depth_cm=25.,max_iterations=800,stride=4,ratios=None,profiles=PROFILES,progress=print):
    p=Path(project).resolve();ratios=dict(RATIOS if ratios is None else ratios);make_body(height_cm,ratios)
    config=dict(height_cm=float(height_cm),tilt_deg=float(tilt_deg),depth_cm=float(depth_cm),
        max_iterations=int(max_iterations),knot_stride=int(stride),ratios=ratios,profiles=list(profiles),version=2)
    provenance=region_identity(p,config);key=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:12]
    out=p/'outputs/trial3/contact_region_plans'/f'target_{height_cm:g}cm_{key}'
    if (out/'summary.json').exists():return verified_experiment(out,provenance)
    if out.exists():raise FileExistsError('Incomplete run preserved: '+str(out))
    out.mkdir(parents=True);start=time.monotonic();data=load_inputs(p);regions=build_regions(p,data,height_cm)
    # A complete prior 2D trajectory provides only the starting guess. Every
    # 3D joint/parameter is fitted again against all native frames.
    data['initial_reference']=np.load(p/BASELINE/'trajectory.npz')['poses']
    write_json(out/'contact_episodes.json',regions['episodes']);pd.DataFrame(regions['evidence']).to_csv(out/'source_evidence.csv',index=False)
    write_json(out/'camera.json',data['camera']);cases=[];rows=[]
    for profile in profiles:
        if progress:progress(f'{profile}: jointly fitting all {len(data["frames"])} frames',flush=True)
        problem=RegionProblem(data,regions,profile,height_cm,tilt_deg,depth_cm,stride,ratios)
        fit=problem.solve(max_iterations=max_iterations,progress=None if progress is None else lambda s:progress(profile+': '+s,flush=True))
        metrics,limbs,gaps=audit_regions(fit,data,regions,tilt_deg,height_cm,ratios)
        case=out/profile;case.mkdir();cases.append(case)
        np.savez_compressed(case/'trajectory.npz',frames=data['frames'],parameters=fit['parameters'],values=fit['values'],
            joints_cm=fit['joints'],center_of_mass_cm=fit['com'],head_cm=fit['head'],
            contact_gaps_cm=gaps,contacts_cm=fit['contacts'],targets_cm=fit['targets'],offsets_cm=fit['offsets'],
            known=data['known'],labels=data['labels'],confidence=data['confidence'],knots=fit['knots'],
            direct_switch=fit['direct_switch'],
            K=data['K'],R=data['R'],t_cm=data['t'],basis_helper=fit['basis_helper'])
        metrics.to_csv(case/'metrics.csv',index=False);limbs.to_csv(case/'limb_metrics.csv',index=False)
        solver=dict(fit['solver']);history=solver.pop('history',[]);pd.DataFrame(history).to_csv(case/'solver_history.csv',index=False)
        write_json(case/'summary.json',dict(name=profile,height_cm=height_cm,tilt_deg=tilt_deg,depth_prior_cm=depth_cm,
            solver=solver,frame_count=len(data['frames']),first_frame=int(data['frames'][0]),last_frame=int(data['frames'][-1]),
            pose_knot_count=len(fit['knots']),all_native_frames_in_objective=True,global_trajectory_solve=True,
            offsets='Rigid unmeasured episode connector from surface to wrist/ankle; no hand/foot articulation or torque limits.',
            physical_validation=False))
        row=summarize(profile,fit,metrics,regions);row['tilt_deg']=tilt_deg;rows.append(row)
        if progress:progress(f'{profile}: contact failures {row["contact_failure_frames"]}, gap {row["max_contact_gap_cm"]:.3f} cm, converged {row["converged"]}',flush=True)
    ranked,fields=rank_candidates(rows);pd.DataFrame(ranked).to_csv(out/'candidate_comparison.csv',index=False)
    selected=ranked[0]['profile'];selected_case=out/selected
    with np.load(selected_case/'trajectory.npz') as z:
        chosen=dict(values=z['values'],joints=z['joints_cm'],com=z['center_of_mass_cm'],head=z['head_cm'],
            contacts=z['contacts_cm'],targets=z['targets_cm'],offsets=z['offsets_cm'],solver=json.loads((selected_case/'summary.json').read_text())['solver'])
        chosen['direct_switch']=z['direct_switch']
    sensitivity=[]
    for angle in sorted(set([0.,float(tilt_deg),30.])):
        mt,_,_=audit_regions(chosen,data,regions,angle,height_cm,ratios)
        mt.to_csv(out/f'selected_audit_tilt{angle:g}.csv',index=False)
        sensitivity.append(dict(tilt_deg=angle,conditional_balance_frames=int(mt.conditional_balance.sum()),
            eligible_balance_frames=int(mt.eligible_balance.sum()),robust_frames=int(mt.conditional_robust_2cm.sum()),
            preunload_failure_frames=int((mt.preunload_without_departing==False).sum()),same_geometry=True,tilt_measured=False))
    pd.DataFrame(sensitivity).to_csv(out/'angle_sensitivity.csv',index=False)
    explanations=[];mt=pd.read_csv(selected_case/'metrics.csv').fillna('not_applicable')
    for t,f in enumerate(data['frames']):
        moving=[LIMBS[i] for i in range(4) if not data['known'][t,i]]
        departing=[LIMBS[i] for i in range(4) if data['unload'][t,i]>0]
        phase=('Pre-release support check: '+', '.join(departing)) if departing else ('Transfer / unresolved limb: '+', '.join(moving) if moving else 'Maintain contacts and reposition body')
        explanations.append(dict(frame=int(f),phase=phase,selected_candidate=selected,reason=ranked[0]['selection_reason'],
            contact_gap_cm=float(mt.iloc[t].max_contact_gap_cm),conditional_balance=bool(mt.iloc[t].conditional_balance),
            issues=mt.iloc[t].issues,all_geometry_uncertain=True))
    pd.DataFrame(explanations).to_csv(out/'phase_explanations.csv',index=False)
    from .region3d_render import render_region_outputs
    render_region_outputs(p,data,regions,out,cases,selected)
    if region_identity(p,config)!=provenance:raise RuntimeError('Inputs changed during run.')
    summary=dict(provenance=provenance,selected_candidate=selected,ranking_fields=fields,comparison=ranked,
        native_frames=len(data['frames']),start_frame=int(data['frames'][0]),end_frame=int(data['frames'][-1]),fps=float(data['fps']),
        seconds=time.monotonic()-start,physical_validation=False,
        interpretation='Selection is among complete candidate trajectories under unmeasured wall/camera/contact/offset assumptions. A selected candidate can still fail.',
        output_sha256={str(f.relative_to(out)):file_sha256(f) for f in sorted(out.rglob('*')) if f.is_file()})
    write_json(out/'summary.json',summary)
    return {**summary,'output_dir':str(out)}
