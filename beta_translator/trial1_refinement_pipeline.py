"""Repeatable Trial1 refinement, preserving the preceding ground-support run."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

from . import trial1_ground_pipeline as ground
from .support_refinement import refine_support_sequence,refinement_settings,RefinementObjective,stationarity
from .support_optimization_pipeline import publish
from .gravity_sequence_pipeline import _verified_cached
from .inverse_kinematics import make_body
from .project_io import file_sha256

PREVIOUS='outputs/trial1/ground_support/target_180cm_bd8c3dc75cd8'


def provenance(project):
    p=Path(project);values=ground.provenance(p)
    for name in ('support_refinement.py','trial1_refinement_pipeline.py'):
        values['beta_translator/'+name]=file_sha256(Path(__file__).with_name(name))
    for name in ('summary.json','trajectory.npz'):
        values[PREVIOUS+'/'+name]=file_sha256(p/PREVIOUS/name)
    return values


def prepare(project,height_cm=180.,mass_kg=70.,foot_friction=.8,max_force_n=None,ratios=None,
            settings=None,audit_config=None,optimization=None,refinement=None,progress=print):
    p=Path(project).resolve()
    inputs,identity=ground.prepare(p,height_cm,mass_kg,foot_friction,max_force_n,ratios,settings,audit_config,optimization)
    old=json.loads((p/PREVIOUS/'summary.json').read_text())
    path=p/PREVIOUS/'trajectory.npz'
    if file_sha256(path)!=old['output_sha256']['trajectory.npz']:raise ValueError('Previous ground trajectory integrity check failed.')
    same_annotation=old['provenance']['inputs']['data/trial1_ground_support.json']==identity['inputs']['data/trial1_ground_support.json']
    with np.load(path,allow_pickle=False) as z:
        if not np.array_equal(z['frames'],inputs['schedule']['frames']):raise ValueError('Previous native frame IDs changed.')
        same_schedule=(np.array_equal(z['decoded_labels'],inputs['schedule']['labels'])
            and np.array_equal(z['known'],inputs['schedule']['known'])
            and np.array_equal(z['targets_cm'],inputs['schedule']['targets'])
            and np.array_equal(z['contact_offsets_cm'],inputs['schedule']['contact_offsets_cm']))
        if (same_annotation and same_schedule and identity['body']==old['body'] and identity['physics']==old['physics']
                and identity['settings']==old['settings']):
            initial=dict(frames=z['frames'].copy(),poses=z['poses'].copy(),joints=z['joints_cm'].copy(),
                contact_forces_n=z['contact_forces_n'].copy(),center_of_mass_cm=z['center_of_mass_cm'].copy(),
                reference=inputs['reference'],visibility=inputs['visibility'],reference_endpoints=inputs['reference_endpoints'],
                schedule=inputs['schedule'],body=old['body'],lengths=make_body(**old['body']),settings=old['settings'],
                mass_kg=old['physics']['mass_kg'],foot_friction=old['physics']['foot_friction'],
                max_force_component_n=old['physics']['max_force_component_n'],optimizer=old['optimizer'],
                trial1_context=inputs['trial1_context'],method=old['method'])
            mode='previous_ground_trajectory'
        else:
            initial=None;mode='fresh_gravity_for_changed_configuration'
    identity.update(inputs=provenance(p),trial='trial1_refinement',refinement_settings=refinement_settings(refinement),
        initialization_mode=mode,baseline_summary=PREVIOUS+'/summary.json',baseline_summary_sha256=file_sha256(p/PREVIOUS/'summary.json'))
    inputs['previous_ground_summary']=old;inputs['initialization_mode']=mode
    return inputs,initial,identity


def diagnose(result,reference_outside,audit_config):
    d=ground.ground_diagnose(result,reference_outside,audit_config);n=len(result['frames'])
    point=np.zeros(n,bool);unload=np.zeros(n,bool)
    for row in result.get('singleton_com_constraints',[]):
        if row['status']=='exact_nominal_singleton':
            point[row['index']]=True
            if row['kind']=='unloading':unload[row['index']]=True
    d['flags'].update(singleton_support_requires_exact_com=point,singleton_unload_not_robust=unload)
    d['issues']=[';'.join(k for k,v in d['flags'].items() if v[t]) for t in range(n)]
    d['counts']={k:int(v.sum()) for k,v in d['flags'].items()}
    return d


def artifacts(stage,inputs,initial,result,before,after):
    extra=ground.artifacts(stage,inputs,initial,result,before,after)
    summary=extra['summary'];project=Path(inputs['project']);old=inputs['previous_ground_summary']
    video=project/PREVIOUS/old['video_name']
    if file_sha256(video)!=old['output_sha256'][old['video_name']]:raise ValueError('Previous ground video failed integrity check.')
    previous_ref='../../ground_support/'+Path(PREVIOUS).name+'/'+old['video_name']
    page=stage/'review.html';html=page.read_text().replace(summary['baseline_video_name'],previous_ref)
    html=html.replace('ground experiment','refined ground support').replace('Previous 180 cm no-ground video','Previous 180 cm ground video')
    html=html.replace('The previous video is the preserved 180 cm run with no ground support; its contact model differs.',
        'The previous video is the preserved 180 cm ground-support run. The new run retains its contact evidence and audit thresholds when the configuration is unchanged.')
    html=html.replace('comparison.csv compares the fresh ground gravity initialization with the refined poses using identical ground evidence; published_comparison.csv compares the two published models.',
        'comparison.csv compares the initialization with the refined poses using identical ground evidence. If body or evidence changes, initialization is regenerated with gravity IK; the previous video remains the labeled 180 cm reference.')
    html=html.replace('Ground gravity baseline vs refined poses','Initialization vs refined poses')
    html=html.replace('<a href="published_comparison.csv">Previous published vs new model</a> · ','')
    html=html.replace('</main>', '<p class="note">Exact singleton COM alignment can make nominal balance possible while remaining non-robust to the unchanged ±2 cm COM perturbation. Low reserve and uncertain timing remain warnings. Numerical convergence is not physical validation.</p><p><a href="convergence.json">Convergence checks</a> · <a href="singleton_com_constraints.csv">Exact nominal support constraints</a></p></main>')
    page.write_text(html)
    redundant=stage/'published_comparison.csv';extra['files'].remove(redundant);redundant.unlink()
    summary.pop('published_model_comparison',None)
    objective=RefinementObjective(result,inputs['fps'],result['optimization_settings'],after['support_audit']['settings'],result['refinement_settings'])
    values=objective.initial_coordinates(result);residual,jac=objective.evaluate(values)
    raw,scale,active,gradient=stationarity(values,residual,jac,objective.lower.ravel(),objective.upper.ravel())
    convergence=dict(optimizer=result['optimizer'],settings=result['refinement_settings'],
        recomputed_refreshed_projected_gradient=gradient,
        recomputed_raw_projected_gradient=float(np.max(abs(np.where(active,0.,raw)))),
        residual_norm=float(np.linalg.norm(residual)),global_frames=len(result['frames']),
        enforced_singleton_frames=sorted(objective.fixed),
        interpretation='First-order numerical convergence of the constrained reweighted objective; not a global optimum or physical validation.',
        rounds=result['optimization_history'])
    path=stage/'convergence.json';path.write_text(json.dumps(convergence,indent=2,allow_nan=False)+'\n');extra['files'].append(path)
    path=stage/'singleton_com_constraints.csv';pd.DataFrame(result['singleton_com_constraints']).to_csv(path,index=False);extra['files'].append(path)
    rows=[]
    for round_ in result['optimization_history']:
        for step in round_['inner']['history']:rows.append({'round':round_['round'],**step})
    path=stage/'solver_history.csv';pd.DataFrame(rows).to_csv(path,index=False);extra['files'].append(path)
    summary.update(trial='Trial1 refined ground support',refinement_settings=result['refinement_settings'],
        previous_run=PREVIOUS,baseline_video_name=previous_ref,initialization=inputs['initialization_mode'],
        comparison_definition='Same evidence and audit thresholds: initialization vs refined trajectory. The saved default 180 cm run starts from the preceding ground-support result.',
        recomputed_projected_gradient=gradient,singleton_com_frame_count=len(objective.fixed),
        refinement_limitations=['Exact singleton nominal balance has no robustness to a nonzero COM-x perturbation.',
            'Contact restoration remains a soft objective; unresolved contacts must still be reported.',
            'Numerical first-order convergence does not establish a global optimum or physical climbing feasibility.'])
    return extra


def run_trial1_refinement(project,height_cm=180.,mass_kg=70.,foot_friction=.8,max_force_n=None,ratios=None,
        settings=None,audit_config=None,optimization=None,refinement=None,progress=print):
    p=Path(project).resolve()
    inputs,initial,ident=prepare(p,height_cm,mass_kg,foot_friction,max_force_n,ratios,settings,audit_config,optimization,refinement,progress)
    root=p/'outputs/trial1/refined_support';key=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:12]
    cached=_verified_cached(root/f'target_{height_cm:g}cm_{key}',ident)
    if cached:
        if progress:progress('Reused verified Trial1 converged-refinement artifacts.')
        return cached
    if initial is None:
        if progress:progress('Body, physics or ground schedule changed: creating a fresh gravity initialization.')
        initial=ground.initialize(inputs,ident,progress)
    result=refine_support_sequence(initial,inputs['fps'],audit_config,optimization,refinement,progress)
    return publish(p,inputs,initial,result,ident,result_root=root,progress=progress,
        input_provenance_fn=provenance,diagnostic_fn=diagnose,artifact_builder=artifacts)
