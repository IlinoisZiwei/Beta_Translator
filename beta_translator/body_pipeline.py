"""One fit/audit/export path for every trial config; old outputs stay read-only."""
from pathlib import Path
import hashlib,json,time,platform,zipfile
from importlib.metadata import version
import numpy as np
import pandas as pd
from .body_articulation import BodyProblem,PatchProblem,audit_body
from . import body_legs
from .body_contacts import foot_frames
from .body_leg_reference import verify_leg_chain
from .body_reference import verify_chain
from .body_kinematics import initialize
from .body_report import constraint_results,candidate_row,rank_candidates
from .body_constraints import coverage_table
from .trials import load_trial,load_config
from .inverse_kinematics import make_body
from .project_io import file_sha256

PROFILES={'legacy_control':1.,'anatomical_control':1.,'balanced':1.,'steady':2.5}
CONTROLS=('legacy_control','anatomical_control')


def numpy_json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=numpy_json)+'\n')


def sliced(data, ix, n):
    return {k:(v[ix] if isinstance(v,np.ndarray) and v.ndim and len(v)==n else v) for k,v in data.items()}


def input_hashes(project, config):
    p=Path(project);root=p/config['input_root']
    files=[p/config['video'],p/config['baseline']]
    for folder in ['state','tables','automatic/predictions']:
        files.extend(f for f in (root/folder).iterdir() if f.is_file())
    files.extend([root/'automatic/wholebody.npz',p/'data/trial2_stage1_chain_reference.json'])
    files.extend(f for f in (p/'data').iterdir() if f.is_file() and f.suffix=='.json')
    files.extend(f for f in (p/'beta_translator').iterdir() if f.suffix=='.py')
    return {str(f.relative_to(p)):file_sha256(f) for f in sorted(set(files))}


def verified_run(out):
    out=Path(out);s=json.loads((out/'summary.json').read_text())
    for name,digest in s['output_sha256'].items():
        if file_sha256(out/name)!=digest:raise ValueError('Saved result changed: '+name)
    return {**s,'output_dir':str(out)}


def run(model, project, trial, frames, iterations, candidates, output_root, render, progress):
    p=Path(project).resolve();cfg=load_config(p,trial)
    if type(iterations) is not int or iterations<1:raise ValueError('Positive iteration budget required')
    if not candidates or not any(c not in CONTROLS for c in candidates) or len(set(candidates))!=len(candidates) or any(c not in PROFILES for c in candidates):raise ValueError('Invalid candidate profiles')
    if isinstance(frames,str):
        if frames not in ('verification','all'):raise ValueError('Use verification, all, or an inclusive [start,end] range')
        bounds=cfg['verification_frames'] if frames=='verification' else [0,cfg['frame_count']-1]
    else:bounds=list(frames)
    if len(bounds)!=2 or any(type(x) is not int for x in bounds) or not 0<=bounds[0]<bounds[1]<cfg['frame_count'] or bounds[1]-bounds[0]<2:
        raise ValueError('At least three consecutive native frames required')
    settings=dict(trial_config=cfg,height_cm=model.height_cm,ratios=model.ratios,frames=bounds,
                  iterations=iterations,candidates={c:PROFILES[c] for c in candidates},body_model='personalized_mechanics_v3',user_profile=model.user_profile,advanced_constraints=model.advanced_constraints)
    identity=dict(config=settings,input_sha256=input_hashes(p,cfg))
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    root=Path(output_root) if output_root else p/f"outputs/{cfg['trial']}/shared_body"
    out=root/f"target_{model.height_cm:g}cm_{bounds[0]:04d}_{bounds[1]:04d}_{digest}"
    if (out/'summary.json').exists():
        result=verified_run(out)
        if result['provenance']!=identity:raise ValueError('Saved provenance mismatch')
        if render:
            from .body_render import render_run
            result['presentation_dir']=str(render_run(out))
        return result
    if out.exists():raise FileExistsError('Incomplete run preserved: '+str(out))
    out.mkdir(parents=True);write_json(out/'provenance.json',identity);started=time.monotonic()
    write_json(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        packages={name:version(name) for name in ['numpy','scipy','pandas','matplotlib','opencv-python']}))
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in identity['input_sha256']:
            if name.startswith(('beta_translator/','data/')):archive.write(p/name,name)
    _,data,scene,regions,legacy=load_trial(p,cfg,model.height_cm)
    n=len(data['frames']);ix=np.arange(bounds[0],bounds[1]+1)
    input_report=dict(native_frames=n,known_contact_samples=int(data['known'].sum()),
        ground_samples=int(data['ground'].sum()),body_implementation='beta_translator.body_articulation.BodyProblem',
        native_time_axis_complete=True,source_visibility_preserved=True,scene=scene.report())
    write_json(out/'all_frame_input_validation.json',input_report)
    local=sliced(data,ix,n);old=sliced(legacy,ix,n);r={**regions,'mapping':regions['mapping'][ix]}
    lengths=make_body(model.height_cm,model.ratios)
    qlegacy=initialize(old,lengths,model.height_cm)
    q=body_legs.initialize(old,lengths,model.height_cm,foot_frames(local,scene,r,model.height_cm)[0])
    np.savez_compressed(out/'initialization.npz',q=q,legacy_q=qlegacy,frames=local['frames'])
    reference=verify_chain();write_json(out/'chain_verification.json',reference)
    if not reference['rotational_chain_pass']:raise RuntimeError('Anatomical chain verification failed')
    leg_reference=verify_leg_chain();write_json(out/'lower_limb_verification.json',leg_reference)
    if not leg_reference['passed']:raise RuntimeError('Lower-limb source verification failed')
    pd.DataFrame(coverage_table()).to_csv(out/'constraint_contract.csv',index=False)
    write_json(out/'scene.json',scene.report());write_json(out/'user_profile.json',model.user_profile);rows=[]
    for name in candidates:
        case=out/name;case.mkdir()
        progress(f"{cfg['trial']} / {name}: frames {bounds[0]}–{bounds[1]}",flush=True)
        from .body_personalized import PersonalizedProblem,audit_personalized
        problem_class=PatchProblem if name=='legacy_control' else PersonalizedProblem
        seed=qlegacy if name=='legacy_control' else q
        extra={} if name=='legacy_control' else dict(user_profile=model.user_profile,advanced=model.advanced_constraints and name not in CONTROLS)
        problem=problem_class(local,scene,r,old,seed,model.height_cm,ratios=model.ratios,smoothness=PROFILES[name],**extra)
        fit=problem.solve(iterations,lambda text:progress(name+': '+text,flush=True))
        m,gaps,forces=audit_body(fit,local,scene,r,model.height_cm,ratios=model.ratios)
        if name!='legacy_control':audit_personalized(fit,m,local,scene,r,model.user_profile)
        checks=constraint_results(fit,m,local,scene,r)
        history=fit['solver'].pop('history',[])
        row=candidate_row(name,fit,m,checks,local['fps']);rows.append(row)
        arrays={k:v for k,v in fit.items() if isinstance(v,np.ndarray)}
        arrays['offsets']=fit['joints'][:,:,2]-fit['contact_points']
        np.savez_compressed(case/'trajectory.npz',**arrays,frames=local['frames'],known=local['known'],
            labels=local['labels'],raw_labels=local['raw_labels'],ground=local['ground'],
            contact_gaps_cm=gaps,audit_forces_bw=forces)
        m.to_csv(case/'metrics.csv',index=False);pd.DataFrame(history).to_csv(case/'solver_history.csv',index=False)
        pd.DataFrame(checks).to_csv(case/'constraints.csv',index=False)
        write_json(case/'constraint_results.json',checks)
        write_json(case/'contact_episodes.json',r['episodes']);write_json(case/'foot_frame_evidence.json',problem.foot_evidence)
        write_json(case/'summary.json',dict(solver=fit['solver'],metrics=row))
    controls=[r for r in rows if r['candidate'] in CONTROLS]
    for row in controls:row.update(eligible_for_selection=False,selection_status='comparison_only',selection_reason='Comparison control; does not optimize all active personalized constraints.')
    ranking=rank_candidates([r for r in rows if r['candidate'] not in CONTROLS])
    for row in ranking:row['eligible_for_selection']=True
    pd.DataFrame([*ranking,*controls]).to_csv(out/'candidate_comparison.csv',index=False)
    for name,h in identity['input_sha256'].items():
        if file_sha256(p/name)!=h:raise RuntimeError('Input changed during solve: '+name)
    result=dict(provenance=identity,trial=cfg['trial'],height_cm=model.height_cm,first_frame=bounds[0],last_frame=bounds[1],
        frame_count=len(ix),fps=local['fps'],selected_candidate=ranking[0]['candidate'],
        selection_status=ranking[0]['selection_status'],specification_status=ranking[0]['specification_status'],
        ranking=ranking,controls=controls,physical_validation=False,seconds=time.monotonic()-started,
        output_sha256={str(f.relative_to(out)):file_sha256(f) for f in out.rglob('*') if f.is_file()})
    write_json(out/'summary.json',result)
    if render:
        from .body_render import render_run
        result['presentation_dir']=str(render_run(out))
    return {**result,'output_dir':str(out)}
