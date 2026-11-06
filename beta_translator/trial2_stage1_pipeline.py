"""Versioned, controlled Stage 1 comparisons; old artifacts are read-only inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .project_io import file_sha256
from .trial2_spatial_pipeline import load_fit, write_json
from .trial2_anatomy_pipeline import slice_arrays, CLIPS
from .trial2_anatomy_optimizer import AnatomicalProblem, audit, selection_row, rank_candidates
from .trial2_spatial_optimizer import audit_plan
from .trial2_spatial_data import load_trial2, episode_regions
from .trial2_scene3d import Trial2Scene
from .trial2_stage1 import Stage1Problem, audit_stage1, foot_frames, verify_chain
from .anatomical_arms import elbow_swivel_rates

BASELINE = 'outputs/trial2/anatomy_plans/target_180cm_controls_db7b5a5de762'


def common_audit(fit, data, scene, regions, height):
    """Re-audit both models' actual patch positions with the SAME 0.1 cm test."""
    _, offset, _, _ = foot_frames(data, scene, regions, height)
    points = fit.get('contact_points', np.concatenate([
        fit['palm_contacts'], fit['joints'][:, 2:, 2]+offset], axis=1))
    physical = {**fit, 'contact_points':points, 'contact_targets':fit['contacts']}
    m, gaps, forces = audit_plan(physical, data, scene, regions, height)
    return m, gaps, forces, points


def run_stage1(project, height_cm=180., clip='right_elbow', iterations=250,
               output_root=None, progress=print):
    p = Path(project).resolve()
    if height_cm != 180.:raise ValueError('This controlled comparison requires its 180 cm baseline.')
    clips = {name:(a,b) for name,a,b in CLIPS}
    if clip not in clips:raise ValueError('Unknown abnormal clip')
    start,end = clips[clip]
    config = dict(height_cm=height_cm, clip=clip, frames=[start,end], iterations=iterations,
        angles_deg=[15,0], fps=30, experiment='stage1_contact_consistency_v1',
        candidates={'legacy_control':1., 'stage1_balanced':1., 'stage1_steady':2.5},
        foot_model='episode_surface_aligned_proxy; ankle ROM unchecked',
        hand_model='source_rotational_chain; translations/girdle omitted',
        contact_tolerance_cm=.1, contact_labels_changed=False)
    paths = [BASELINE+'/balanced/trajectory.npz', BASELINE+'/initialization.npz',
        'data/trial2_stage1_chain_reference.json', 'data/trial2_spatial_hypotheses.json',
        'videos/trial2.mp4', 'outputs/trial2/state/reviewed_hold_map.npz',
        'outputs/trial2/state/alignment_cache.npz', 'outputs/trial2/state/identity_guard.npz',
        'outputs/trial2/tables/pose_coordinates.csv', 'outputs/trial2/automatic/wholebody.npz',
        'outputs/trial2/automatic/predictions/limb_contacts.csv']
    paths += ['beta_translator/'+f+'.py' for f in ['trial2_stage1','trial2_stage1_pipeline',
        'anatomical_arms','trial2_anatomy_optimizer','trial2_spatial_optimizer','trial2_spatial_data',
        'trial2_scene3d','restored_sequence_solver','constrained_sequence_solver','inverse_kinematics',
        'ik3d','scene3d','contact_regions3d']]
    identity = dict(config=config, input_sha256={f:file_sha256(p/f) for f in paths})
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    root = Path(output_root) if output_root else p/'outputs/trial2/stage1_contacts'
    out = root/f'target_180cm_{clip}_{key}'
    if (out/'summary.json').exists():
        saved = json.loads((out/'summary.json').read_text())
        if saved['provenance'] != identity:raise ValueError('Provenance mismatch')
        for f,h in saved['output_sha256'].items():
            if file_sha256(out/f)!=h:raise ValueError('Saved artifact changed: '+f)
        return {**saved, 'output_dir':str(out)}
    if out.exists():raise FileExistsError('Incomplete experiment preserved: '+str(out))
    out.mkdir(parents=True)
    write_json(out/'provenance.json', identity)
    verification = verify_chain()
    write_json(out/'chain_verification.json', verification)
    if not verification['rotational_chain_pass']:raise RuntimeError('Source rotation verification failed')
    all_data = load_trial2(p); n = len(all_data['frames']); ix = np.arange(start,end+1)
    scene = Trial2Scene(p,15,0,15); all_regions = episode_regions(all_data,scene,height_cm)
    data = slice_arrays(all_data,ix,n); regions = {**all_regions,'mapping':all_regions['mapping'][ix]}
    legacy = slice_arrays(load_fit(p/BASELINE/'balanced'),ix,n)
    with np.load(p/BASELINE/'initialization.npz', allow_pickle=False) as z:q = z['q'][ix]
    np.savez_compressed(out/'initialization.npz',q=q,frames=data['frames'])
    comparisons=[]; rankings=[]
    for name,gain in config['candidates'].items():
        cls = AnatomicalProblem if name=='legacy_control' else Stage1Problem
        problem = cls(data,scene,regions,legacy,q,height_cm,smoothness=gain)
        dest=out/name;dest.mkdir()
        progress(f'{name}: frames {start}–{end}, {iterations} iterations',flush=True)
        fit=problem.solve(iterations,lambda message:progress(name+': '+message,flush=True))
        audit_fn = audit if name=='legacy_control' else audit_stage1
        m,gaps,forces=audit_fn(fit,data,scene,regions,height_cm)
        common,cg,cf,points=common_audit(fit,data,scene,regions,height_cm)
        history=fit['solver'].pop('history',[])
        values={k:v for k,v in fit.items() if isinstance(v,np.ndarray)}
        # Use the actual geometry-derived offset in new output; retain the old
        # offset separately to make the change auditable.
        values['legacy_offsets']=values['offsets'].copy()
        if name!='legacy_control':values['offsets']=fit['joints'][:,:,2]-points
        np.savez_compressed(dest/'trajectory.npz',**values, frames=data['frames'],known=data['known'],
            labels=data['labels'],ground=data['ground'],contact_gaps_cm=gaps,audit_forces_bw=forces,
            common_patch_points=points,common_patch_gaps_cm=cg)
        m.to_csv(dest/'metrics.csv',index=False);common.to_csv(dest/'common_contact_metrics.csv',index=False)
        pd.DataFrame(history).to_csv(dest/'solver_history.csv',index=False)
        write_json(dest/'scene.json',scene.report());write_json(dest/'contact_episodes.json',regions['episodes'])
        if name!='legacy_control':write_json(dest/'foot_frame_evidence.json',problem.foot_evidence)
        row=selection_row(name,m,fit['solver'])
        write_json(dest/'summary.json',dict(metrics=row,solver=fit['solver'],config=config))
        if name!='legacy_control':rankings.append(row)
        step=np.diff(points,axis=0);same=data['known'][1:]&data['known'][:-1]&(data['labels'][1:]==data['labels'][:-1])
        acc=np.linalg.norm(np.diff(fit['joints'],n=2,axis=0),axis=-1)*data['fps']**2
        sw=elbow_swivel_rates(fit['joints'],data['fps']);valid=sw[np.isfinite(sw)]
        comparisons.append(dict(candidate=name,frames=len(m),declared_contact_failure_frames=int(m.contact_failed.sum()),
            common_patch_failure_frames=int(common.contact_failed.sum()),
            common_hand_max_gap_cm=float(np.max(np.where(data['known'][:,:2],cg[:,:2],0))),
            common_foot_max_gap_cm=float(np.max(np.where(data['known'][:,2:],cg[:,2:],0))),
            common_support_failure_frames=int((~common.conditional_balance).sum()),
            common_preunload_failure_frames=int(common.preunload.eq(False).sum()),
            matched_contact_samples=int(np.sum(data['known']&(cg<=.1))),contact_samples=int(data['known'].sum()),
            stance_patch_step_max_cm=float(np.linalg.norm(step,axis=-1)[same].max()) if same.any() else None,
            joint_acceleration_rms_cm_s2=float(np.sqrt(np.mean(acc**2))),
            elbow_swivel_p95_deg_s=float(np.quantile(valid,.95)) if len(valid) else None,
            anatomical_temporal_failure_frames=row['anatomy_temporal_failure_frames'],
            bone_error_cm=float(m.bone_error_cm.max()),converged=bool(fit['solver']['success'])))
    ranking=rank_candidates(rankings)
    pd.DataFrame(comparisons).to_csv(out/'before_after.csv',index=False)
    pd.DataFrame(ranking).to_csv(out/'candidate_comparison.csv',index=False)
    for f,h in identity['input_sha256'].items():
        if file_sha256(p/f)!=h:raise RuntimeError('Input changed during run: '+f)
    result=dict(provenance=identity,selected_candidate=ranking[0]['candidate'],
        selection_status=ranking[0]['selection_status'],comparison=comparisons,ranking=ranking,
        first_frame=start,last_frame=end,frame_count=len(data['frames']),fps=30,
        physical_validation=False,scope='Short segment only; no claim of full-sequence validation',
        output_sha256={str(f.relative_to(out)):file_sha256(f) for f in out.rglob('*') if f.is_file()})
    write_json(out/'summary.json',result)
    return {**result,'output_dir':str(out)}
