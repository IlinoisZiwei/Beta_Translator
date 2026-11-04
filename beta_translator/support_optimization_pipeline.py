"""Preserved baseline, support optimization, independent audits and new artifacts."""
from collections import Counter
from pathlib import Path
import hashlib,json,tempfile,uuid
import numpy as np
import pandas as pd

from . import support_optimization
from .support_optimization import optimize_support_sequence,optimization_settings
from .support_audit import audit_settings,audit_support_sequence,attach_audit,release_events
from .gravity_sequence import diagnose_gravity_sequence
from .gravity_sequence_pipeline import run_trial3_gravity_sequence,gravity_provenance,render_gravity_sequence,_verified_cached
from .full_sequence_pipeline import load_full_sequence_inputs,_tables,_intervals
from .inverse_kinematics import make_body,GRAVITY
from .project_io import file_sha256


def provenance(project):
    return {**gravity_provenance(project),
        'beta_translator/support_optimization.py':file_sha256(support_optimization.__file__),
        'beta_translator/support_optimization_pipeline.py':file_sha256(__file__)}


def load_initial(project,height_cm=180.,ratios=None,mass_kg=70.,foot_friction=.8,max_force_n=None,
                 settings=None,audit_config=None,progress=print):
    baseline=run_trial3_gravity_sequence(project,height_cm,ratios,mass_kg,foot_friction,max_force_n,settings,progress,audit_config)
    inputs=load_full_sequence_inputs(project)
    with np.load(Path(baseline['output_dir'])/'trajectory.npz',allow_pickle=False) as z:
        result=dict(frames=z['frames'].copy(),poses=z['poses'].copy(),joints=z['joints_cm'].copy(),
            contact_forces_n=z['contact_forces_n'].copy(),center_of_mass_cm=z['center_of_mass_cm'].copy())
    result.update(reference=inputs['reference'],visibility=inputs['visibility'],reference_endpoints=inputs['reference_endpoints'],
        schedule=inputs['schedule'],body=baseline['body'],lengths=make_body(**baseline['body']),settings=baseline['settings'],
        mass_kg=baseline['physics']['mass_kg'],foot_friction=baseline['physics']['foot_friction'],
        max_force_component_n=baseline['physics']['max_force_component_n'],optimizer=baseline['optimizer'],method=baseline['method'])
    return inputs,result,baseline


def identity(project,baseline,optimization=None,audit_config=None):
    path=(Path(baseline['output_dir'])/'summary.json').resolve()
    return dict(inputs=provenance(project),baseline_summary=path.relative_to(Path(project).resolve()).as_posix(),
        baseline_summary_sha256=file_sha256(path),body=baseline['body'],physics=baseline['physics'],
        settings=baseline['settings'],optimization_settings=optimization_settings(optimization),
        support_audit_settings=audit_settings(audit_config))


def diagnose(result,reference_outside,audit_config):
    d=diagnose_gravity_sequence(result,reference_outside)
    a=audit_support_sequence(result,d,audit_config);attach_audit(d,a)
    load=np.full(len(result['frames']),np.nan)
    for event in release_events(result['schedule'],result['frames']):
        t=event['release_index']-1
        load[t]=np.linalg.norm(result['contact_forces_n'][t,event['moving_indices']],axis=1).sum()/(result['mass_kg']*GRAVITY)
    d['departing_load_bw']=load
    d['flags']['departing_load_remaining']=np.nan_to_num(load)>result['settings']['balance_tolerance']
    d['numerical_checks_pass'] &= ~d['flags']['departing_load_remaining'] & ~d['flags']['rapid_pose_change']
    d['issues']=[';'.join(k for k,v in d['flags'].items() if v[t]) for t in range(len(load))]
    d['status'][d['flags']['departing_load_remaining'] | d['flags']['rapid_pose_change']]='constraint_warning'
    d['counts']={k:int(v.sum()) for k,v in d['flags'].items()}
    return d


def comparison(initial,result,before,after):
    rows=[]
    for key in ('contact_residual','large_contact_residual','support_balance_failed','support_balance_uncertain',
                'support_margin_low','preunload_failed','preunload_uncertain','preunload_margin_low',
                'departing_load_remaining','rapid_pose_change','force_imbalance','moment_imbalance','optimizer_unconverged'):
        rows.append(dict(metric=key,unit='frames',before=before['counts'][key],after=after['counts'][key]))
    for key,unit,a,b in [
        ('combined_numerical_pass','frames',before['numerical_checks_pass'].sum(),after['numerical_checks_pass'].sum()),
        ('max_contact_gap','cm',before['max_contact_error_cm'].max(),after['max_contact_error_cm'].max()),
        ('max_hip_step','cm/frame',before['hip_step_cm'].max(),after['hip_step_cm'].max()),
        ('max_angle_step','deg/frame',before['angle_step_deg'].max(),after['angle_step_deg'].max())]:
        rows.append(dict(metric=key,unit=unit,before=float(a),after=float(b)))
    return pd.DataFrame(rows)


def publish(project,inputs,initial,result,run_identity,result_root=None,progress=print,
            input_provenance_fn=None,diagnostic_fn=None,artifact_builder=None):
    root=Path(result_root) if result_root else Path(project)/'outputs/trial3/support_optimized'
    key=hashlib.sha256(json.dumps(run_identity,sort_keys=True).encode()).hexdigest()[:12]
    destination=root/f'target_{result["body"]["height_cm"]:g}cm_{key}'
    if destination.exists():destination=destination.with_name(destination.name+'_'+uuid.uuid4().hex[:8])
    check=diagnostic_fn or diagnose
    d=check(result,inputs['reference_outside'],run_identity['support_audit_settings'])
    before=check(initial,inputs['reference_outside'],run_identity['support_audit_settings']);a=d['support_audit']
    table=comparison(initial,result,before,d)
    frames,limbs=_tables(result,d,inputs['fps']);frames=frames.merge(pd.DataFrame(a['frames']),on='frame',validate='one_to_one')
    frames['numerical_checks_pass']=d['numerical_checks_pass'];frames['balanced']=d['balanced']
    frames['net_force_n']=d['net_force_norm_n'];frames['net_moment_nm']=d['net_moment_nm']
    frames['departing_load_bw']=d['departing_load_bw']
    frames['com_x_cm']=result['center_of_mass_cm'][:,0];frames['com_y_cm']=result['center_of_mass_cm'][:,1]
    limbs['contact_validity']=a['contact_state'].ravel()
    for name in ('eligible','conditional'):limbs[name+'_support']=a[name].ravel()
    for name,force in [('candidate',result['contact_forces_n']),*a['forces_n'].items()]:
        for axis,i in (('x',0),('y',1)):limbs[name+'_force_'+axis+'_n']=force[:,:,i].ravel()
    with tempfile.TemporaryDirectory(prefix='beta-support-optimized-') as temp:
        stage=Path(temp)
        if progress:progress(f'Rendering all {len(result["frames"])} optimized frames and unchanged independent checks.')
        files=render_gravity_sequence(inputs,result,d,stage)
        trial_id=inputs.get('trial_id','trial3')
        if trial_id not in ('trial1','trial2','trial3'):raise ValueError('Invalid trial identifier.')
        video=stage/f'{trial_id}_support_optimized_{result["body"]["height_cm"]:g}cm_full_sequence.mp4'
        files[0].rename(video);files[0]=video
        tables={'frame_diagnostics.csv':frames,'limb_diagnostics.csv':limbs,'comparison.csv':table,
            'issue_intervals.csv':pd.DataFrame(_intervals(d,inputs['fps'])),
            'release_events.csv':pd.DataFrame(a['events'],columns=['event_id','release_index','release_frame','moving_indices','moving_limbs','from_labels','to_labels','timing_uncertain','timing_reason']),
            'preunload_checks.csv':pd.DataFrame(a['unloading']) if a['unloading'] else pd.DataFrame(columns=['event_id','frame','release_frame','status']),
            'optimization_constraints.csv':pd.DataFrame(result['optimization_constraints'])}
        for name,df in tables.items():df.to_csv(stage/name,index=False);files.append(stage/name)
        np.savez_compressed(stage/'trajectory.npz',frames=result['frames'],poses=result['poses'],joints_cm=result['joints'],
            initial_poses=initial['poses'],reference_poses=result['reference'],visibility=result['visibility'],
            targets_cm=result['schedule']['targets'],decoded_labels=result['schedule']['labels'],
            contact_forces_n=result['contact_forces_n'],center_of_mass_cm=result['center_of_mass_cm'],
            eligible=a['eligible'],conditional=a['conditional'],eligible_forces_n=a['forces_n']['eligible'],conditional_forces_n=a['forces_n']['conditional'],
            contact_errors_cm=d['contact_errors_cm'],numerical_checks_pass=d['numerical_checks_pass'])
        files.append(stage/'trajectory.npz')
        extra=artifact_builder(stage,inputs,initial,result,before,d) if artifact_builder else {}
        files.extend(extra.get('files',[]))
        final_unload={r['event_id']:r for r in a['unloading']}
        summary=dict(schema_version=1,provenance=run_identity,body=result['body'],physics=run_identity['physics'],
            settings=result['settings'],optimization_settings=result['optimization_settings'],optimization_history=result['optimization_history'],
            support_audit_settings=run_identity['support_audit_settings'],method=result['method'],optimizer=result['optimizer'],
            frame_count=len(result['frames']),first_frame=int(result['frames'][0]),last_frame=int(result['frames'][-1]),
            fps=inputs['fps'],duration_seconds=len(result['frames'])/inputs['fps'],video_name=video.name,
            comparison=table.to_dict('records'),flag_counts=d['counts'],status_counts=dict(Counter(map(str,d['status']))),
            balance_audit_counts=dict(Counter(r['balance_audit_status'] for r in a['frames'])),
            preunload_frame_counts=dict(Counter(r['unload_audit_status'] for r in a['frames'])),
            last_pre_release_check_counts=dict(Counter(r['status'] for r in final_unload.values())),
            release_event_count=len(a['events']),release_timing_uncertain_count=sum(e['timing_uncertain'] for e in a['events']),
            numerical_pass_frame_count=int(d['numerical_checks_pass'].sum()),max_segment_length_error_cm=d['max_segment_length_error_cm'],
            source_climber_height_cm=None,
            limitations=[
                'Soft global objectives seek improved poses; convergence is not a global optimum or physical validation.',
                'Prospective COM intervals assume desired contacts can be reached; final audits independently exclude missed holds.',
                'IRLS favors achievable contacts and can increase errors on incompatible contacts; all original targets remain audited.',
                'Support confidence and calibration are unchanged; conditional support never becomes confirmed evidence.',
                'Contact labels/timing stay fixed. Direct hold changes retain timing uncertainty; ground support is not invented.',
                'Unloading encourages a gradual reduction in departing candidate force and fixed-pose reserve without departing limbs.',
                'The final pre-release candidate load is flagged above 1% body weight; counterfactual readiness is a separate check.',
                'Assumed 2D quasi-static force capacities/friction omit depth, inertia, measured normals, grip strength and joint torques.',
                'Temporal penalties and motion limits do not establish dynamic feasibility. Unresolved constraints remain visible.'],
            output_sha256={p.name:file_sha256(p) for p in files},**extra.get('summary',{}))
        (stage/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');files.append(stage/'summary.json')
        if (input_provenance_fn or provenance)(project)!=run_identity['inputs']:raise RuntimeError('Inputs/code changed; no results installed.')
        if 'baseline_summary' in run_identity and file_sha256(Path(project)/run_identity['baseline_summary'])!=run_identity['baseline_summary_sha256']:
            raise RuntimeError('Baseline changed; no results installed.')
        destination.parent.mkdir(parents=True,exist_ok=True);destination.mkdir(exist_ok=False)
        for p in files:(destination/p.name).write_bytes(p.read_bytes())
    return {**summary,'output_dir':str(destination),'video_file':str(destination/video.name)}


def run_trial3_support_optimization(project,height_cm=180.,ratios=None,mass_kg=70.,foot_friction=.8,
        max_force_n=None,settings=None,audit_config=None,optimization=None,progress=print):
    project=Path(project).resolve()
    inputs,initial,baseline=load_initial(project,height_cm,ratios,mass_kg,foot_friction,max_force_n,settings,audit_config,progress)
    ident=identity(project,baseline,optimization,audit_config)
    key=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:12]
    target=project/'outputs/trial3/support_optimized'/f'target_{height_cm:g}cm_{key}'
    cached=_verified_cached(target,ident)
    if cached:
        if progress:progress('Reused verified support-optimized full sequence.')
        return cached
    result=optimize_support_sequence(initial,inputs['fps'],audit_config,optimization,inputs['reference_outside'],progress)
    return publish(project,inputs,initial,result,ident,progress=progress)
