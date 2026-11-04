"""Compare coarse complete candidates, then refine one with native-frame DOFs."""
from pathlib import Path
import json,hashlib,shutil,time
import numpy as np,pandas as pd
from .trial3_region_pipeline import run_region_experiment,audit_regions,summarize,rank_candidates
from .trial3_3d_pipeline import load_inputs,BASELINE,write_json,verified_experiment
from .contact_regions3d import build_regions
from .native_region3d import NativeRegionProblem
from .project_io import file_sha256
from .region3d_render import render_region_outputs
from .inverse_kinematics import LIMBS


def run_complete_region_plan(project,native_iterations=600,progress=print,**kwargs):
    p=Path(project).resolve();base=run_region_experiment(p,progress=progress,**kwargs)
    if 'region_offset' not in base['provenance']['config']['profiles']:return base
    parent=Path(base['output_dir']);cfg=dict(base['provenance']['config'])
    cfg['native_refinement_iterations']=int(native_iterations)
    hashes=dict(base['provenance']['inputs_sha256'])
    hashes[str((parent/'summary.json').relative_to(p))]=file_sha256(parent/'summary.json')
    for n in ['native_region3d.py','trial3_native_region_pipeline.py']:
        hashes['beta_translator/'+n]=file_sha256(p/'beta_translator'/n)
    provenance=dict(config=cfg,inputs_sha256=hashes)
    key=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:12]
    height=cfg['height_cm'];tilt=cfg['tilt_deg'];depth=cfg['depth_cm'];ratios=cfg['ratios']
    out=parent.parent/f'target_{height:g}cm_{key}_native'
    if (out/'summary.json').exists():return verified_experiment(out,provenance)
    if out.exists():raise FileExistsError('Incomplete native refinement is preserved: '+str(out))
    out.mkdir();started=time.monotonic();cases=[]
    for name in cfg['profiles']:
        shutil.copytree(parent/name,out/name);cases.append(out/name)
    for name in ['camera.json','contact_episodes.json','source_evidence.csv']:shutil.copy2(parent/name,out/name)
    data=load_inputs(p);data['initial_reference']=np.load(p/BASELINE/'trajectory.npz')['poses']
    regions=build_regions(p,data,height)
    with np.load(parent/'region_offset/trajectory.npz') as z:
        initial=dict(values=z['values'],joints=z['joints_cm'],parameters=z['parameters'])
    problem=NativeRegionProblem(data,regions,initial,height,tilt,depth,ratios)
    if progress:progress('Native refinement: all 503 poses and episode contacts optimized together.',flush=True)
    fit=problem.solve(max_iterations=native_iterations,progress=None if progress is None else lambda s:progress('native: '+s,flush=True))
    metrics,limbs,gaps=audit_regions(fit,data,regions,tilt,height,ratios)
    name='region_offset_native';case=out/name;case.mkdir();cases.append(case)
    np.savez_compressed(case/'trajectory.npz',frames=data['frames'],parameters=fit['parameters'],values=fit['values'],
        joints_cm=fit['joints'],center_of_mass_cm=fit['com'],head_cm=fit['head'],contact_gaps_cm=gaps,
        contacts_cm=fit['contacts'],targets_cm=fit['targets'],offsets_cm=fit['offsets'],known=data['known'],
        labels=data['labels'],confidence=data['confidence'],knots=fit['knots'],direct_switch=fit['direct_switch'],
        K=data['K'],R=data['R'],t_cm=data['t'],basis_helper=fit['basis_helper'])
    metrics.to_csv(case/'metrics.csv',index=False);limbs.to_csv(case/'limb_metrics.csv',index=False)
    solver=dict(fit['solver']);history=solver.pop('history',[]);pd.DataFrame(history).to_csv(case/'solver_history.csv',index=False)
    write_json(case/'summary.json',dict(name=name,height_cm=height,tilt_deg=tilt,depth_prior_cm=depth,solver=solver,
        first_frame=0,last_frame=502,frame_count=503,pose_knot_count=len(fit['knots']),all_native_frames_in_objective=True,
        global_trajectory_solve=True,initialization='Complete region_offset candidate, converted to segment directions.',
        comparison_caveat='Additional native-frame DOFs and solve budget; a refinement, not an equal-budget ablation.',
        physical_validation=False))
    row=summarize(name,fit,metrics,regions);row['tilt_deg']=tilt
    ranked,fields=rank_candidates([*base['comparison'],row]);pd.DataFrame(ranked).to_csv(out/'candidate_comparison.csv',index=False)
    selected=ranked[0]['profile'];chosen_case=out/selected
    with np.load(chosen_case/'trajectory.npz') as z:
        chosen=dict(values=z['values'],joints=z['joints_cm'],com=z['center_of_mass_cm'],head=z['head_cm'],
            contacts=z['contacts_cm'],targets=z['targets_cm'],offsets=z['offsets_cm'],direct_switch=z['direct_switch'],
            solver=json.loads((chosen_case/'summary.json').read_text())['solver'])
    sensitivity=[]
    for angle in sorted(set([0.,tilt,30.])):
        mt,_,_=audit_regions(chosen,data,regions,angle,height,ratios)
        mt.to_csv(out/f'selected_audit_tilt{angle:g}.csv',index=False)
        sensitivity.append(dict(tilt_deg=angle,conditional_balance_frames=int(mt.conditional_balance.sum()),
            eligible_balance_frames=int(mt.eligible_balance.sum()),robust_frames=int(mt.conditional_robust_2cm.sum()),
            preunload_failure_frames=int((mt.preunload_without_departing==False).sum()),same_geometry=True,tilt_measured=False))
    pd.DataFrame(sensitivity).to_csv(out/'angle_sensitivity.csv',index=False)
    mt=pd.read_csv(chosen_case/'metrics.csv').fillna('not_applicable');explanations=[]
    for t,f in enumerate(data['frames']):
        moving=[LIMBS[i] for i in range(4) if not data['known'][t,i]];departing=[LIMBS[i] for i in range(4) if data['unload'][t,i]>0]
        phase=('Pre-release support check: '+', '.join(departing)) if departing else ('Transfer / unresolved limb: '+', '.join(moving) if moving else 'Maintain contacts and reposition body')
        explanations.append(dict(frame=int(f),phase=phase,selected_candidate=selected,reason=ranked[0]['selection_reason'],
            contact_gap_cm=float(mt.iloc[t].max_contact_gap_cm),conditional_balance=bool(mt.iloc[t].conditional_balance),
            issues=mt.iloc[t].issues,all_geometry_uncertain=True))
    pd.DataFrame(explanations).to_csv(out/'phase_explanations.csv',index=False)
    render_region_outputs(p,data,regions,out,cases,selected)
    for relative,digest in hashes.items():
        if file_sha256(p/relative)!=digest:raise RuntimeError('Input changed during native refinement: '+relative)
    summary=dict(provenance=provenance,parent_experiment=str(parent.relative_to(p)),selected_candidate=selected,
        ranking_fields=fields,comparison=ranked,native_frames=503,start_frame=0,end_frame=502,fps=float(data['fps']),
        seconds=time.monotonic()-started,parent_solve_seconds=base['seconds'],physical_validation=False,
        interpretation='Five equal-setting contact-model ablations plus one full native-frame refinement with extra DOFs and compute. All geometry remains conditional.',
        output_sha256={str(f.relative_to(out)):file_sha256(f) for f in sorted(out.rglob('*')) if f.is_file()})
    write_json(out/'summary.json',summary)
    return {**summary,'output_dir':str(out)}
