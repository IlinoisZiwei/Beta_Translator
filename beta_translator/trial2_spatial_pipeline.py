"""Trial2 complete multi-panel candidates, independent audits and preserved outputs."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,time
import numpy as np
import pandas as pd
from .project_io import file_sha256
from .inverse_kinematics import RATIOS
from .trial2_scene3d import Trial2Scene
from .trial2_spatial_data import load_trial2,episode_regions
from .trial2_spatial_optimizer import Trial2Problem,audit_plan,rank_plans


def write_json(path,value):
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value,indent=2,default=default,allow_nan=False)+'\n')


def load_fit(directory):
    with np.load(Path(directory)/'trajectory.npz',allow_pickle=False) as z:fit={k:z[k] for k in z.files}
    fit['solver']=json.loads((Path(directory)/'summary.json').read_text())['solver']
    return fit


def case_row(name,config,fit,metrics):
    valid=metrics.hand_load_l1_bw.dropna()
    return dict(candidate=name,middle_deg=config['middle_deg'],upper_deg=config['upper_deg'],
        contact_profile=config['profile'],support_optimized=config['support'],technique=config['technique'],
        native_frames=len(metrics),converged=bool(fit['solver']['success']),nonconverged=int(not fit['solver']['success']),
        contact_failure_frames=int(metrics.contact_failed.sum()),geometric_failure_frames=int(metrics.geometric_failed.sum()),
        motion_warning_frames=int(metrics.motion_warning.sum()),support_failure_frames=int((~metrics.conditional_balance).sum()),
        conditional_balance_frames=int(metrics.conditional_balance.sum()),eligible_balance_frames=int(metrics.eligible_balance.sum()),
        preunload_check_frames=int(metrics.preunload.notna().sum()),preunload_failure_frames=int((metrics.preunload==False).sum()),
        low_margin_frames=int((~metrics.robust_2cm).sum()),max_contact_gap_cm=float(metrics.max_contact_gap_cm.max()),
        maximum_bone_error_cm=float(metrics.bone_error_cm.max()),max_step_cm=float(metrics.step_cm.max()),
        hand_load_integral_bw_s=float(valid.sum()/30) if len(valid) else 1e12,hand_load_evaluated_frames=len(valid),
        hand_load_peak_bw=float(valid.max()) if len(valid) else None,
        solve_seconds=float(fit['solver']['seconds']),all_geometry_uncertain=True)


def common_load_comparison(out,rows):
    """Compare load proxies only on identical feasible frames across these cases."""
    tables=[pd.read_csv(Path(out)/r['candidate']/'metrics.csv') for r in rows]
    common=np.logical_and.reduce([t.conditional_balance.to_numpy(bool)&t.hand_load_l1_bw.notna().to_numpy() for t in tables])
    for row,t in zip(rows,tables):
        # Recompute from its own feasible frames: this function is also called
        # after a refinement is appended to an already compared candidate set.
        row['own_feasible_hand_integral_bw_s']=float(t.hand_load_l1_bw.dropna().sum()/30)
        row['common_load_frames']=int(common.sum())
        row['hand_load_integral_bw_s']=float(t.loc[common,'hand_load_l1_bw'].sum()/30) if common.any() else 0.
        row['hand_load_proxy_informative']=bool(common.any())
    return rows


def run_case(project,data,out,name,config,initial=None,progress=print):
    dest=out/name
    if dest.exists():raise FileExistsError('Existing candidate is preserved: '+str(dest))
    dest.mkdir(parents=True)
    scene=Trial2Scene(project,config['middle_deg'],config['upper_deg'],config['spacing_cm'])
    regions=episode_regions(data,scene,config['height_cm'])
    if progress:progress(f'{name}: {len(data["frames"])} native frames, middle {scene.angles[0]:g} / upper {scene.angles[1]:g} degrees',flush=True)
    problem=Trial2Problem(data,scene,regions,config['height_cm'],config['ratios'],config['profile'],
        config['support'],config['technique'],config['stride'],initial,config.get('analytic_contacts',True))
    fit=problem.solve(config['max_iterations'],None if progress is None else lambda x:progress(name+': '+x,flush=True))
    metrics,gaps,forces=audit_plan(fit,data,scene,regions,config['height_cm'],config['ratios'])
    solver=dict(fit['solver']);history=solver.pop('history',[])
    np.savez_compressed(dest/'trajectory.npz',**{k:v for k,v in fit.items() if isinstance(v,np.ndarray)},
        frames=data['frames'],raw_labels=data['raw_labels'],labels=data['labels'],confidence=data['confidence'],
        known=data['known'],ground=data['ground'],contact_gaps_cm=gaps,audit_forces_bw=forces)
    metrics.to_csv(dest/'metrics.csv',index=False);pd.DataFrame(history).to_csv(dest/'solver_history.csv',index=False)
    write_json(dest/'scene.json',scene.report());write_json(dest/'contact_episodes.json',regions['episodes'])
    pd.DataFrame(regions['evidence']).to_csv(dest/'contact_evidence.csv',index=False)
    row=case_row(name,config,fit,metrics)
    write_json(dest/'summary.json',dict(config=config,solver=solver,metrics=row,physical_validation=False,
        optimization='One global sparse solve; every native frame contributes. One surface point shared per contact episode.',
        uncertainty='All panel angles, intrinsics, ground depth, grid orthogonality and joint offsets are conditional.',
        all_native_frames_in_objective=True,source_height_cm=173.,target_height_cm=config['height_cm']))
    if progress:progress(f'{name} finished: contact failures {row["contact_failure_frames"]}, geometric failures {row["geometric_failure_frames"]}, motion warnings {row["motion_warning_frames"]}, balance {row["conditional_balance_frames"]}/{len(metrics)}, converged={row["converged"]}',flush=True)
    return row,fit


def input_identity(project,config):
    p=Path(project)
    files=['videos/trial2.mp4','data/trial2_geometry_reference.json','data/trial2_spatial_hypotheses.json',
        'outputs/trial2/state/reviewed_hold_map.npz','outputs/trial2/state/alignment_cache.npz',
        'outputs/trial2/state/identity_guard.npz','outputs/trial2/automatic/wholebody.npz',
        'outputs/trial2/automatic/predictions/limb_contacts.csv','outputs/trial2/tables/pose_coordinates.csv']
    files+=['beta_translator/'+n+'.py' for n in ['trial2_scene3d','trial2_spatial_data','trial2_spatial_optimizer','trial2_spatial_pipeline','trial2_spatial_render','ik3d','scene3d','contact_regions3d','support_refinement','inverse_kinematics']]
    return dict(config=config,input_sha256={n:file_sha256(p/n) for n in files})


def run_trial2_spatial(project,height_cm=180.,ratios=None,angles=None,spacing_cm=15.,max_iterations=350,stride=4,progress=print,workers=2):
    from .trial2_spatial_render import render_outputs
    p=Path(project).resolve();started=time.monotonic()
    if angles is None:angles=json.loads((p/'data/trial2_spatial_hypotheses.json').read_text())['angle_scenarios']
    config=dict(height_cm=float(height_cm),ratios=dict(RATIOS if ratios is None else ratios),angles=[list(a) for a in angles],
        spacing_cm=float(spacing_cm),max_iterations=int(max_iterations),stride=int(stride))
    identity=input_identity(p,config);key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    out=p/'outputs/trial2/spatial_plans'/f'target_{height_cm:g}cm_{key}'
    if (out/'summary.json').exists():
        summary=json.loads((out/'summary.json').read_text())
        if summary['provenance']!=identity:raise ValueError('Cache provenance mismatch')
        for name,digest in summary['output_sha256'].items():
            if not (out/name).is_file() or file_sha256(out/name)!=digest:raise ValueError('Changed or missing cached artifact: '+name)
        return {**summary,'output_dir':str(out)}
    if out.exists():raise FileExistsError('Incomplete experiment preserved: '+str(out))
    out.mkdir(parents=True);data=load_trial2(p);rows=[]
    pd.DataFrame(data['ground_evidence']).to_csv(out/'ground_evidence.csv',index=False)
    base={k:v for k,v in config.items() if k!='angles'}
    def fit_batch(cases,initial=None):
        if workers<=1:
            return [run_case(p,data,out,name,cfg,initial,progress)[0] for name,cfg in cases]
        # Candidates are independent complete solves. Collect in declared order
        # so parallel scheduling cannot change tie-breaking or the candidate set.
        with ProcessPoolExecutor(max_workers=min(int(workers),len(cases))) as pool:
            jobs=[pool.submit(run_case,p,data,out,name,cfg,initial,print if progress else None) for name,cfg in cases]
            return [job.result()[0] for job in jobs]
    cases=[]
    for middle,upper in angles:
        name=f'region_m{middle:g}_u{upper:g}'
        cfg=dict(base,middle_deg=float(middle),upper_deg=float(upper),profile='region_offset',support=False,technique='none')
        cases.append((name,cfg))
    rows.extend(fit_batch(cases))
    angle_rank,_=rank_plans(common_load_comparison(out,[dict(x) for x in rows]));chosen=angle_rank[0]
    # Selecting a low-failure hypothesis does not estimate the physical wall angle.
    selected_angle=(chosen['middle_deg'],chosen['upper_deg']);seed=load_fit(out/chosen['candidate'])
    cases=[]
    for name,profile,support,technique in [('center_control','center',False,'none'),('support','region_offset',True,'none'),
            ('leg_drive','region_offset',True,'leg_drive'),('flag_assist','region_offset',True,'flag_assist'),('arm_economy','region_offset',True,'arm_economy')]:
        cfg=dict(base,middle_deg=selected_angle[0],upper_deg=selected_angle[1],profile=profile,support=support,technique=technique)
        cases.append((name,cfg))
    rows.extend(fit_batch(cases,seed))
    ranked,fields=rank_plans(common_load_comparison(out,rows));selected=ranked[0]['candidate']
    pd.DataFrame(ranked).to_csv(out/'candidate_comparison.csv',index=False)
    pd.DataFrame(angle_rank).to_csv(out/'angle_comparison.csv',index=False)
    mt=pd.read_csv(out/selected/'metrics.csv');phases=[]
    for t,f in enumerate(data['frames']):
        ground=bool(data['ground'][t].any());upper=mt.iloc[t].upper_contacts;middle=mt.iloc[t].middle_contacts
        phase='Ground departure (proposed)' if ground else ('Mixed support across panel hinge' if upper and middle else ('Upper-panel movement' if upper else 'Middle-panel movement / unresolved support'))
        if (data['unload'][t]>0).any():phase+='; pre-release check'
        phases.append(dict(frame=int(f),phase=phase,selected_candidate=selected,selection_reason=ranked[0]['selection_reason'],
            issues=str(mt.iloc[t].issues),geometry_uncertain=True))
    pd.DataFrame(phases).to_csv(out/'phase_explanations.csv',index=False)
    render_outputs(p,data,out,ranked,selected)
    for name,digest in identity['input_sha256'].items():
        if file_sha256(p/name)!=digest:raise RuntimeError('Input changed during experiment: '+name)
    summary=dict(provenance=identity,selected_candidate=selected,comparison=ranked,ranking_fields=fields,
        frame_count=len(data['frames']),first_frame=0,last_frame=int(data['frames'][-1]),fps=30.,seconds=time.monotonic()-started,
        all_geometry_uncertain=True,physical_validation=False,
        interpretation='Lowest-failure tested full trajectory under conditional multi-panel geometry; not a measured angle or validated climb.',
        output_sha256={str(f.relative_to(out)):file_sha256(f) for f in sorted(out.rglob('*')) if f.is_file()})
    write_json(out/'summary.json',summary)
    return {**summary,'output_dir':str(out)}
