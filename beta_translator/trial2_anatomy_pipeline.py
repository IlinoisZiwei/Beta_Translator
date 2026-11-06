"""Versioned short-clip comparisons followed by independent full-video fitting."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from .project_io import file_sha256
from .inverse_kinematics import make_body,RATIOS
from .trial2_scene3d import Trial2Scene
from .trial2_spatial_data import load_trial2,episode_regions
from .trial2_spatial_pipeline import load_fit,write_json
from .trial2_anatomy_optimizer import AnatomicalProblem,audit,selection_row,rank_candidates
from . import anatomical_arms as arms

LEGACY='outputs/trial2/spatial_plans/target_180cm_5f9e7fcd448b_native/native_refinement'
CLIPS=[('hinge_transition',475,515),('right_elbow',550,595),('left_elbow',615,650)]
PROFILES={'balanced':1.,'steady':2.5}


def slice_arrays(data,indices,n):
    return {k:(v[indices] if isinstance(v,np.ndarray) and v.ndim and len(v)==n else v) for k,v in data.items()}


def provenance(project,config):
    paths=[LEGACY+'/trajectory.npz',LEGACY+'/summary.json','data/trial2_arm_model_sources.json',
           'data/trial2_spatial_hypotheses.json','videos/trial2.mp4',
           'outputs/trial2/state/alignment_cache.npz','outputs/trial2/state/reviewed_hold_map.npz',
           'outputs/trial2/state/identity_guard.npz','outputs/trial2/tables/pose_coordinates.csv',
           'outputs/trial2/automatic/wholebody.npz','outputs/trial2/automatic/predictions/limb_contacts.csv']
    paths+=['beta_translator/'+s+'.py' for s in ['anatomical_arms','trial2_anatomy_optimizer','trial2_anatomy_pipeline',
        'constrained_sequence_solver','restored_sequence_solver','trial2_spatial_data','trial2_spatial_optimizer','trial2_scene3d','contact_regions3d','inverse_kinematics','ik3d','scene3d','support_refinement']]
    return dict(config=config,input_sha256={s:file_sha256(project/s) for s in paths})


def save_case(dest,problem,config,iterations,progress):
    if (dest/'summary.json').exists():
        s=json.loads((dest/'summary.json').read_text())
        if s['config']!=config:raise ValueError('Candidate settings changed')
        for name,digest in s['output_sha256'].items():
            if file_sha256(dest/name)!=digest:raise ValueError('Candidate artifact changed: '+name)
        return s['metrics']
    if dest.exists():raise FileExistsError('Incomplete candidate preserved: '+str(dest))
    dest.mkdir(parents=True);progress(f'Fitting {dest.name}: {problem.n} frames',flush=True)
    fit=problem.solve(iterations,lambda s:progress(dest.name+': '+s,flush=True))
    m,gaps,forces=audit(fit,problem.data,problem.scene,problem.regions,problem.height)
    solver=dict(fit['solver']);history=solver.pop('history',[])
    np.savez_compressed(dest/'trajectory.npz',**{k:v for k,v in fit.items() if isinstance(v,np.ndarray)},
        frames=problem.data['frames'],known=problem.data['known'],ground=problem.data['ground'],
        labels=problem.data['labels'],raw_labels=problem.data['raw_labels'],contact_gaps_cm=gaps,audit_forces_bw=forces)
    m.to_csv(dest/'metrics.csv',index=False);pd.DataFrame(history).to_csv(dest/'solver_history.csv',index=False)
    write_json(dest/'scene.json',problem.scene.report());write_json(dest/'contact_episodes.json',problem.regions['episodes'])
    row=selection_row(dest.name,m,solver)
    write_json(dest/'summary.json',dict(config=config,solver=solver,metrics=row,physical_validation=False,
        output_sha256={f.name:file_sha256(f) for f in dest.iterdir() if f.is_file()}))
    progress(f'{dest.name}: contact {row["contact_failure_frames"]}/{problem.n}, anatomy/time {row["anatomy_temporal_failure_frames"]}, elbow swivel max {row["max_elbow_swivel_deg_s"]:.1f} deg/s',flush=True)
    return row


def comparative_metrics(legacy,new,old_metrics,new_metrics,fps):
    rows=[]
    for name,fit,m in [('legacy',legacy,old_metrics),('anatomical',new,new_metrics)]:
        j=fit['joints'];sw=arms.elbow_swivel_rates(j,fps);sw=sw[np.isfinite(sw)]
        acc=np.linalg.norm(np.diff(j,n=2,axis=0),axis=-1)*fps**2
        hip=j[:,2:,0].mean(axis=1);ha=np.linalg.norm(np.diff(hip,n=2,axis=0),axis=1)*fps**2
        torso=j[:,:2,0].mean(axis=1)-hip;torso/=np.linalg.norm(torso,axis=1,keepdims=True)
        tr=np.rad2deg(np.arccos(np.clip(np.sum(torso[:-1]*torso[1:],axis=1),-1,1)))*fps
        rows.append(dict(model=name,frames=len(j),elbow_swivel_max_deg_s=float(sw.max()) if len(sw) else None,
            elbow_swivel_p95_deg_s=float(np.quantile(sw,.95)) if len(sw) else None,
            joint_acceleration_rms_cm_s2=float(np.sqrt(np.mean(acc**2))),hip_acceleration_rms_cm_s2=float(np.sqrt(np.mean(ha**2))),
            torso_axis_rotation_p95_deg_s=float(np.quantile(tr,.95)),contact_failure_frames=int(m.contact_failed.sum()),
            maximum_contact_gap_cm=float(m.max_contact_gap_cm.max()),support_failure_frames=int((~m.conditional_balance).sum()),
            preunload_failure_frames=int(m.preunload.eq(False).sum()),motion_warning_frames=int(m.motion_warning.sum()),
            maximum_bone_error_cm=float(m.bone_error_cm.max())))
    return rows


def run_anatomical_trial2(project,height_cm=180.,clip_iterations=150,full_iterations=350,stage='all',progress=print):
    p=Path(project).resolve()
    if stage not in ('clips','full','all'):raise ValueError('Unknown stage')
    if height_cm!=180.:raise ValueError('This controlled comparison currently requires 180 cm, matching its saved baseline.')
    config=dict(height_cm=float(height_cm),fps=30.,ratios=RATIOS,clip_iterations=int(clip_iterations),
        full_iterations=int(full_iterations),clips=CLIPS,profiles=PROFILES,angles_deg=[15.,0.],
        velocity_limits_status='Project assumptions, not measured human limits or MoBL limits',
        arm_model='reduced_MoBL_inspired',support_optimized=True,
        initialization='wrist_priority_with_elbow_hint_and_adaptive_multistart')
    # Normalize tuples exactly as they will be stored in JSON.
    config=json.loads(json.dumps(config));identity=provenance(p,config)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    out=p/'outputs/trial2/anatomy_plans'/f'target_180cm_{key}'
    out.mkdir(parents=True,exist_ok=True)
    final=out/'summary.json'
    if final.exists():
        s=json.loads(final.read_text())
        if s['provenance']!=identity:raise ValueError('Provenance mismatch')
        for name,digest in s['output_sha256'].items():
            if file_sha256(out/name)!=digest:raise ValueError('Changed anatomical artifact: '+name)
        return {**s,'output_dir':str(out)}
    write_json(out/'provenance.json',identity)
    data=load_trial2(p);n=len(data['frames']);legacy=load_fit(p/LEGACY)
    old_metrics=pd.read_csv(p/LEGACY/'metrics.csv');scene=Trial2Scene(p,15,0,15);regions=episode_regions(data,scene,height_cm)
    seed_key=hashlib.sha256((identity['input_sha256']['beta_translator/anatomical_arms.py']+
        identity['input_sha256'][LEGACY+'/trajectory.npz']).encode()).hexdigest()[:12]
    seedpath=out.parent/f'initial_180cm_{seed_key}.npz'
    if seedpath.exists():
        with np.load(seedpath,allow_pickle=False) as z:q=z['q']
    else:
        q=arms.initialize(legacy,make_body(height_cm),height_cm,progress);np.savez_compressed(seedpath,q=q)
    np.savez_compressed(out/'initialization.npz',q=q)
    clip_rows=[]
    for label,start,end in CLIPS:
        ix=np.arange(start,end+1);d=slice_arrays(data,ix,n);r={**regions,'mapping':regions['mapping'][ix]}
        old=slice_arrays(legacy,ix,n);rows=[]
        for name,gain in PROFILES.items():
            case=f'{label}_{name}';cfg={**config,'frames':[start,end],'smoothness':gain}
            problem=AnatomicalProblem(d,scene,r,old,q[ix],height_cm,smoothness=gain)
            row=save_case(out/'clips'/case,problem,cfg,clip_iterations,progress);rows.append(row)
        ranked=rank_candidates(rows);best=ranked[0]['candidate'];dest=out/'clips'/best
        new=load_fit(dest);m=pd.read_csv(dest/'metrics.csv')
        comparison=comparative_metrics(old,new,old_metrics.iloc[ix],m,30.)
        write_json(out/'clips'/f'{label}_comparison.json',dict(frames=[start,end],ranking=ranked,comparison=comparison))
        clip_rows.extend(dict(clip=label,**row) for row in ranked)
    pd.DataFrame(clip_rows).to_csv(out/'clip_candidate_comparison.csv',index=False)
    write_json(out/'clip_summary.json',dict(provenance=identity,comparison=clip_rows,
        conclusion='Anatomy/rate/physics gates evaluated on predefined abnormal clips before the full independent solve.'))
    if stage=='clips':return dict(output_dir=str(out),stage='clips',comparison=clip_rows,provenance=identity)
    # Both complete candidates start from the SAME whole-video initialization.
    # Short-clip fitted poses are never pasted into these trajectories.
    full_rows=[]
    for name,gain in PROFILES.items():
        problem=AnatomicalProblem(data,scene,regions,legacy,q,height_cm,smoothness=gain)
        full_rows.append(save_case(out/name,problem,{**config,'frames':[0,n-1],'smoothness':gain},full_iterations,progress))
    ranked=rank_candidates(full_rows);selected=ranked[0]['candidate']
    pd.DataFrame(ranked).to_csv(out/'candidate_comparison.csv',index=False)
    new=load_fit(out/selected);m=pd.read_csv(out/selected/'metrics.csv')
    compare=comparative_metrics(legacy,new,old_metrics,m,30.);pd.DataFrame(compare).to_csv(out/'before_after.csv',index=False)
    explanations=[]
    for k,row in m.iterrows():
        phase='Mixed panel support' if row.upper_contacts and row.middle_contacts else ('Upper wall' if row.upper_contacts else 'Lower / middle wall')
        if data['ground'][k].any():phase+='; ground proposed'
        if data['unload'][k].any():phase+='; release preparation'
        explanations.append(dict(frame=int(row.frame),phase=phase,selected=selected,reason=ranked[0]['selection_reason'],issues=row.issues))
    pd.DataFrame(explanations).to_csv(out/'phase_explanations.csv',index=False)
    for name,digest in identity['input_sha256'].items():
        if file_sha256(p/name)!=digest:raise RuntimeError('Input changed during solve: '+name)
    result=dict(provenance=identity,selected_candidate=selected,selection_status=ranked[0]['selection_status'],
        comparison=ranked,before_after=compare,clip_comparison=clip_rows,frame_count=n,first_frame=0,last_frame=n-1,fps=30.,
        baseline=LEGACY,physical_validation=False,
        output_sha256={str(f.relative_to(out)):file_sha256(f) for f in out.rglob('*') if f.is_file()})
    write_json(final,result)
    return {**result,'output_dir':str(out)}
