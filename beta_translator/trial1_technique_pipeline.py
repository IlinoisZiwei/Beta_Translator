"""Reproducible Trial1 candidate comparison and a single selected full video."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from .trial1_refinement_pipeline import prepare,provenance as old_provenance
from .inverse_kinematics import make_body
from .project_io import file_sha256
from .technique_planner import PROFILES,templates,fit_candidate,assess,rank_key,episodes

BASELINE='outputs/trial1/refined_support/target_180cm_f1456408fd1d'


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_baseline(project,baseline_dir=None):
    root=Path(project).resolve();directory=root/(baseline_dir or BASELINE);summary=json.loads((directory/'summary.json').read_text())
    for rel in ('trajectory.npz','frame_diagnostics.csv'):
        if file_sha256(directory/rel)!=summary['output_sha256'][rel]:raise ValueError('Baseline artifact changed: '+rel)
    body=summary['body'];physics=summary['physics']
    inputs,_,_=prepare(root,body['height_cm'],physics['mass_kg'],physics['foot_friction'],physics['max_force_component_n'],
        body['ratios'],summary['settings'],summary['support_audit_settings'],summary['optimization_settings'],summary['refinement_settings'],None)
    current=old_provenance(root);differences={}
    for name,digest in summary['provenance']['inputs'].items():
        if current.get(name)!=digest:
            # Trial3's orchestration changed after this historical Trial1 fit.
            # Re-audit the fixed saved trajectory; never call it a new cache hit.
            if name!='beta_translator/full_sequence_pipeline.py':raise ValueError('Baseline provenance changed: '+name)
            differences[name]=dict(historical=digest,current=current.get(name),
                interpretation='Historical orchestration difference; saved trajectory is independently re-audited, not claimed to be refitted by current code.')
    inputs['baseline_code_differences']=differences
    with np.load(directory/'trajectory.npz',allow_pickle=False) as z:
        if not np.array_equal(z['decoded_labels'],inputs['schedule']['labels']):raise ValueError('Contact schedule mismatch.')
        if not np.array_equal(z['targets_cm'],inputs['schedule']['targets']):raise ValueError('Contact geometry mismatch.')
        result={k:z[v].copy() for k,v in dict(frames='frames',poses='poses',joints='joints_cm',
            contact_forces_n='contact_forces_n',center_of_mass_cm='center_of_mass_cm').items()}
    result.update(reference=inputs['reference'],visibility=inputs['visibility'],reference_endpoints=inputs['reference_endpoints'],
        schedule=inputs['schedule'],body=body,lengths=make_body(**body),settings=summary['settings'],
        mass_kg=physics['mass_kg'],foot_friction=physics['foot_friction'],max_force_component_n=physics['max_force_component_n'],
        optimizer=summary['optimizer'],trial1_context=inputs['trial1_context'],optimization_settings=summary['optimization_settings'],
        audit_settings=summary['support_audit_settings'],technique_profile='preserved_baseline')
    return inputs,result,directory


def identity(root,baseline,config):
    root=Path(root);files=old_provenance(root)
    for name in ['technique_planner.py','trial1_technique_pipeline.py','technique_render.py']:
        files['beta_translator/'+name]=file_sha256(root/'beta_translator'/name)
    for name in ['summary.json','trajectory.npz','frame_diagnostics.csv']:
        f=baseline/name;files[str(f.relative_to(root))]=file_sha256(f)
    return dict(config=config,inputs_sha256=files)


def verified(directory,expected=None):
    directory=Path(directory);p=directory/'summary.json'
    if not p.is_file():raise ValueError('Incomplete technique experiment; preserve it and choose a new configuration.')
    s=json.loads(p.read_text())
    if expected is not None and s['provenance']!=expected:raise ValueError('Stale technique experiment.')
    for f,digest in s['output_sha256'].items():
        if not (directory/f).is_file() or file_sha256(directory/f)!=digest:raise ValueError('Missing/modified technique artifact: '+f)
    return {**s,'output_dir':str(directory)}


def slice_result(result,lo,hi):
    n=len(result['frames']);keep=(result['frames']>=lo)&(result['frames']<=hi)
    def rec(x):
        if isinstance(x,np.ndarray) and x.ndim and len(x)==n:return x[keep].copy()
        if isinstance(x,dict):return {k:rec(v) for k,v in x.items()}
        return x
    return rec(result)


def source_evidence(inputs,baseline):
    """Measured image-plane changes; leave expert technique labels blank."""
    q=inputs['reference'];s=baseline['schedule'];rows=[]
    for k,e in enumerate(episodes(s)):
        lo,hi=e['release_index'],e['arrival_index'];limb=e['limb'];start=max(0,lo-6);end=min(len(q)-1,hi+12)
        # These are observations/proposals, not supervised technique classes.
        hip=q[end,:2]-q[start,:2];bends=np.rad2deg(abs(q[[start,end]][:,[8,10]]))
        rows.append(dict(event_id=f'T{k+1:02d}',moving_limb=['left_hand','right_hand','left_foot','right_foot'][limb],
            start_frame=int(s['frames'][start]),release_frame=e['release_frame'],arrival_frame=e['arrival_frame'],end_frame=int(s['frames'][end]),
            from_hold=e['from_hold'],to_hold=e['to_hold'],observed_hip_dx_projected_cm=float(hip[0]),
            observed_hip_dy_projected_cm=float(hip[1]),observed_left_knee_extension_deg=float(bends[0,0]-bends[1,0]),
            observed_right_knee_extension_deg=float(bends[0,1]-bends[1,1]),
            observation=('Unlocated interval / same-support recontact; do not infer a new move' if e['same_support_recontact'] else
                ('Reach with body repositioning' if limb<2 else 'Foot transfer; possible subsequent body rise')),
            preserved_approach=bool(e['arrival_frame']<103),
            source_height_known=False,observed_force_available=False,expert_technique_label='',review_status='needs_expert_review',
            flagging_confirmed=False,evidence_note='Free foot can be in transit or unlocated; no confirmed flagging label. Uniform image scale and occlusion affect observed posture.'))
    return rows


def save_candidate(directory,result,metrics,rows,forces,d):
    directory.mkdir();s=result['schedule'];np.savez_compressed(directory/'trajectory.npz',frames=result['frames'],poses=result['poses'],
        joints_cm=result['joints'],center_of_mass_cm=result['center_of_mass_cm'],contact_forces_n=result['contact_forces_n'],
        fair_allocation_forces_bw=forces,targets_cm=s['targets'],decoded_labels=s['labels'],known=s['known'],
        contact_errors_cm=d['contact_errors_cm'],contact_offsets_cm=s['contact_offsets_cm'],
        contact_type=s['contact_type'],supported=s['supported'],low_confidence=s['low_confidence'])
    pd.DataFrame(rows).to_csv(directory/'load_metrics.csv',index=False)
    table=pd.DataFrame(d['support_audit']['frames']);table['max_contact_error_cm']=d['max_contact_error_cm']
    table['hip_step_cm']=d['hip_step_cm'];table['angle_step_deg']=d['angle_step_deg'];table['issues']=d['issues']
    table.to_csv(directory/'frame_diagnostics.csv',index=False)
    pd.DataFrame(d['support_audit']['unloading']).to_csv(directory/'preunload_checks.csv',index=False)
    write_json(directory/'metrics.json',metrics)
    write_json(directory/'solver.json',dict(optimizer=result['optimizer'],history=result.get('technique_history',[])))


def run_trial1_techniques(project,height_cm=180.,baseline_dir=None,profiles=None,max_iterations=1600,max_outer=4,progress=print):
    root=Path(project).resolve();inputs,initial,baseline=load_baseline(root,baseline_dir)
    if not np.isclose(height_cm,initial['body']['height_cm']):
        raise ValueError('First run Trial1 Section 8 for the requested body, then pass its matching baseline_dir; never reuse another body silently.')
    profiles=list(PROFILES) if profiles is None else list(profiles)
    if not profiles or len(profiles)!=len(set(profiles)) or any(x not in PROFILES for x in profiles):raise ValueError('Invalid candidate profiles.')
    if type(max_iterations) is not int or max_iterations<1 or type(max_outer) is not int or max_outer<1:raise ValueError('Positive integer budgets required.')
    config=dict(height_cm=float(height_cm),profiles={x:PROFILES[x] for x in profiles},max_iterations=max_iterations,max_outer=max_outer,
        freeze_prefix_before_frame=103,freeze_endpoint_frames=2,contact_schedule='unchanged automatic labels + existing ground evidence',
        rank='feasibility, reliability, hand L1 integral/peak, distal arm-moment proxy, motion; numeric tie bands 0.005 BW*s / 0.005 BW / 0.05 BW*cm*s',
        force_allocation='same two-stage hand-L1 then distal-arm-moment LP for every pose; lower bound under assumed 2D contacts')
    ident=identity(root,baseline,config);digest=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:12]
    out=root/f'outputs/trial1/technique_plans/target_{height_cm:g}cm_{digest}'
    if out.exists():return verified(out,ident)
    out.mkdir(parents=True);write_json(out/'run_inputs.json',ident);started=time.monotonic()
    guide=templates(initial);pd.DataFrame(source_evidence(inputs,initial)).to_csv(out/'technique_review_queue.csv',index=False)
    write_json(out/'phase_events.json',guide['events']);metrics=[];evaluated={}
    names=['preserved_baseline',*profiles]
    for name in names:
        if progress:progress('Fitting/evaluating full 417-frame candidate: '+name,flush=True)
        start=time.monotonic();result=initial if name=='preserved_baseline' else fit_candidate(initial,inputs['fps'],name,guide,max_iterations,max_outer,progress)
        m,rows,forces,d=assess(result,inputs['fps'],inputs['reference_outside']);m['seconds']=time.monotonic()-start
        active=guide['active']>0;known=result['schedule']['known'][:,2:]
        m['mean_active_hip_change_cm']=float(np.mean((result['poses']-initial['poses'])[active,1]))
        m['mean_active_knee_extension_deg']=float(np.mean(np.rad2deg(abs(initial['poses'][active][:,[8,10]])-abs(result['poses'][active][:,[8,10]]))))
        m['max_free_foot_displacement_cm']=float(np.max(np.linalg.norm(result['joints'][:,2:,2]-initial['joints'][:,2:,2],axis=-1)*(~known)))
        m['max_segment_error_cm']=float(d['max_segment_length_error_cm'])
        if m['max_segment_error_cm']>1e-8:raise RuntimeError('Fixed-bone geometry violated; no candidate publication.')
        save_candidate(out/name,result,m,rows,forces,d);metrics.append(m);evaluated[name]=(result,rows,forces,d)
        if progress:progress(str(m),flush=True)
    ranked=sorted(metrics,key=rank_key);selected=ranked[0]['profile'];best=ranked[0];baseline_m=metrics[0]
    # The final candidate was optimized jointly over the entire native sequence.
    # No per-move winners are concatenated.
    criteria=['feasibility/convergence gate','failed checks','solver convergence','uncertain support frames',
        'low-margin frames','uncertain unloading frames','hand-load integral band','hand-load peak band',
        'arm-moment integral band','maximum angular step']
    for i,m in enumerate(ranked,1):
        m['rank']=i;m['selected']=m['profile']==selected
        if i==1:m['selection_reason']='Selected complete trajectory under feasibility/reliability-first ranking; numeric tie bands prevent tiny differences deciding technique choice.'
        else:
            differing=next((k for k,(a,b) in enumerate(zip(rank_key(m),rank_key(best))) if a!=b),None)
            m['selection_reason']=('Numeric tie; retained earlier candidate order.' if differing is None else
                f'Not selected: first ranking difference is {criteria[differing]} ({rank_key(m)[differing]} versus selected {rank_key(best)[differing]}). Earlier criteria tie.')
    pd.DataFrame(ranked).to_csv(out/'candidate_comparison.csv',index=False)
    result,rows,forces,d=evaluated[selected];explanations=[]
    for t,frame in enumerate(result['frames']):
        new=rows[t]['hand_l1_bw'];old=evaluated['preserved_baseline'][1][t]['hand_l1_bw'];delta=None if new is None or old is None else new-old
        reason=('Hand-load lower bound unavailable; preserve failure.' if delta is None else
            ('No meaningful hand-load distinction at this frame; selection uses the complete trajectory.' if abs(delta)<.005 else
             f'Under the same allocation rule, hand-load lower bound changes {old:.3f} to {new:.3f} BW.'))
        explanations.append(dict(frame=int(frame),phase=str(guide['phase'][t]),selected_profile=selected,
            reason=reason,hand_load_change_bw=delta,contact_status='pass' if d['max_contact_error_cm'][t]<=.1 else 'FAIL',
            support_status=d['support_audit']['frames'][t]['balance_audit_status'],unload_status=d['support_audit']['frames'][t]['unload_audit_status'],
            interpretation='Intended phase from automatic contact timing; not a verified human technique label.',geometry_uncertain=True))
    pd.DataFrame(explanations).to_csv(out/'phase_explanations.csv',index=False)
    from .technique_render import render,review_page,evidence_sheet
    render(root,inputs,result,d,rows,explanations,out);review_page(out,ranked,explanations);evidence_sheet(root,inputs,out)
    if identity(root,baseline,config)!=ident:raise RuntimeError('Inputs/code changed during candidate experiment; do not publish incomplete results.')
    summary=dict(provenance=ident,baseline_code_differences=inputs['baseline_code_differences'],selected_profile=selected,
        selection_status='passes_current_numerical_model' if rank_key(best)[0]==0 else 'best_effort_diagnostic_only_no_eligible_plan',
        recommendation_eligible_under_model=bool(rank_key(best)[0]==0),frame_count=len(result['frames']),fps=inputs['fps'],body=result['body'],
        selected_metrics=best,baseline_metrics=baseline_m,elapsed_seconds=time.monotonic()-started,
        video='selected_plan.mp4',review='review.html',all_geometry_uncertain=True,physical_validation=False,
        caveats=['Quasi-static 2D model, assumed foot force direction and uniform image scale.',
          'Hand-load minimum is a force-allocation lower bound, not measured effort or endurance.',
          'Many frames permit zero hand load in this model; zero is not a human force estimate.',
          'Frozen approach poses through frame 102 and first/last two poses preserve the preceding result there.',
          'Contact timing stays automatic; the saved H44-H28 human timing is shown separately, never overwritten.',
          'No new hold order, hip-depth rotation, hooks, smearing, muscle fatigue or dynamics.',
          'No learned technique classifier was trained; source technique labels remain unreviewed.'])
    summary['output_sha256']={str(p.relative_to(out)):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'summary.json',summary)
    return {**summary,'output_dir':str(out)}
