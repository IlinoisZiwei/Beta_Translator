"""Short native-frame tests followed by full cubic-coordinate trajectory fitting."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
from .trial2_anatomy_pipeline import (run_anatomical_trial2,LEGACY,PROFILES,save_case,
    comparative_metrics,rank_candidates,load_trial2,load_fit,Trial2Scene,episode_regions,write_json,file_sha256)
from .temporal_control_basis import ControlledProblem


def run_controlled_trial2(project,height_cm=180.,clip_iterations=150,full_iterations=160,progress=print):
    p=Path(project).resolve()
    # This stage preserves the exact native-frame short experiments. Their
    # fitted poses are never used as a source for the full trajectory.
    short=run_anatomical_trial2(p,height_cm,clip_iterations,350,'clips',progress)
    source=Path(short['output_dir']);qpath=source/'initialization.npz'
    config=dict(height_cm=float(height_cm),fps=30.,full_iterations=int(full_iterations),
        control_stride_frames=8,profiles=PROFILES,angles_deg=[15.,0.],
        short_clip_run=str(source.relative_to(p)),full_solver='restored_sparse_SQP')
    inputs=dict(short['provenance']['input_sha256'])
    for name in ['beta_translator/temporal_control_basis.py','beta_translator/restored_sequence_solver.py',
                 'beta_translator/trial2_anatomy_controls.py',str(qpath.relative_to(p)),
                 str((source/'clip_summary.json').relative_to(p)),'requirements.txt']:
        inputs[name]=file_sha256(p/name)
    clip_hashes={str(f.relative_to(source)):file_sha256(f) for f in (source/'clips').rglob('*') if f.is_file()}
    identity=dict(config=config,input_sha256=inputs,short_clip_provenance=short['provenance'],short_clip_sha256=clip_hashes)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    out=p/'outputs/trial2/anatomy_plans'/f'target_180cm_controls_{key}';out.mkdir(exist_ok=True,parents=True)
    if (out/'summary.json').exists():
        saved=json.loads((out/'summary.json').read_text())
        if saved['provenance']!=identity:raise ValueError('Controlled-run provenance changed')
        for name,h in saved['output_sha256'].items():
            if file_sha256(out/name)!=h:raise ValueError('Controlled artifact changed: '+name)
        return {**saved,'output_dir':str(out)}
    for name,h in inputs.items():
        if file_sha256(p/name)!=h:raise RuntimeError('Input changed before full solve: '+name)
    write_json(out/'provenance.json',identity)
    # Small clip artifacts are copied with explicit provenance, not re-fitted
    # or inserted into the full solution. Keeping them local makes ZIPs portable.
    for name,h in clip_hashes.items():
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and file_sha256(dest)!=h:raise RuntimeError('Clip copy changed: '+name)
        if not dest.exists():shutil.copy2(source/name,dest)
    shutil.copy2(source/'clip_candidate_comparison.csv',out/'clip_candidate_comparison.csv')
    shutil.copy2(qpath,out/'initialization.npz')
    data=load_trial2(p);n=len(data['frames']);legacy=load_fit(p/LEGACY)
    old_metrics=pd.read_csv(p/LEGACY/'metrics.csv');scene=Trial2Scene(p,15,0,15);regions=episode_regions(data,scene,height_cm)
    with np.load(qpath,allow_pickle=False) as z:q=z['q']
    full=[]
    for name,gain in PROFILES.items():
        problem=ControlledProblem(data,scene,regions,legacy,q,height_cm,smoothness=gain)
        full.append(save_case(out/name,problem,{**config,'frames':[0,n-1],'smoothness':gain},full_iterations,progress))
    ranked=rank_candidates(full);selected=ranked[0]['candidate']
    pd.DataFrame(ranked).to_csv(out/'candidate_comparison.csv',index=False)
    new=load_fit(out/selected);m=pd.read_csv(out/selected/'metrics.csv')
    comparison=comparative_metrics(legacy,new,old_metrics,m,30.)
    pd.DataFrame(comparison).to_csv(out/'before_after.csv',index=False)
    phases=[]
    for t,row in m.iterrows():
        phase='Mixed panel support' if row.upper_contacts and row.middle_contacts else ('Upper wall' if row.upper_contacts else 'Lower / middle wall')
        if data['ground'][t].any():phase+='; ground proposed'
        if data['unload'][t].any():phase+='; release preparation'
        phases.append(dict(frame=int(row.frame),phase=phase,selected=selected,reason=ranked[0]['selection_reason'],issues=row.issues))
    pd.DataFrame(phases).to_csv(out/'phase_explanations.csv',index=False)
    for name,h in inputs.items():
        if file_sha256(p/name)!=h:raise RuntimeError('Input changed during full solve: '+name)
    saved=dict(provenance=identity,selected_candidate=selected,selection_status=ranked[0]['selection_status'],
        comparison=ranked,before_after=comparison,clip_comparison=short['comparison'],baseline=LEGACY,
        frame_count=n,first_frame=0,last_frame=n-1,fps=30.,physical_validation=False,
        output_sha256={str(f.relative_to(out)):file_sha256(f) for f in out.rglob('*') if f.is_file()})
    write_json(out/'summary.json',saved)
    return {**saved,'output_dir':str(out)}
