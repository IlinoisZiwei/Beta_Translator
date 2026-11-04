"""Trial1 ground-evidence overlay and one coupled whole-video optimization."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
import pandas as pd

from .trial1_sequence import load_inputs, input_provenance
from .trial1_support_pipeline import trial1_diagnose, review_html
from .ground_support import load_ground_inputs
from .gravity_sequence import GravityProblem, solve_gravity_sequence
from .gravity_sequence_pipeline import _verified_cached
from .support_optimization import optimize_support_sequence, optimization_settings
from .support_optimization_pipeline import publish
from .support_audit import audit_settings
from .contact_geometry import contact_points
from .inverse_kinematics import RATIOS
from .project_io import file_sha256

BASELINE = 'outputs/trial1/support_optimized/target_180cm_77aeef37c158'


def provenance(project, annotation_path=None):
    p=Path(project);a=Path(annotation_path) if annotation_path else p/'data/trial1_ground_support.json'
    hashes=input_provenance(p)
    for name in ('ground_support.py','trial1_ground_pipeline.py','trial1_support_pipeline.py'):
        hashes['beta_translator/'+name]=file_sha256(Path(__file__).with_name(name))
    hashes['data/trial1_ground_support.json']=file_sha256(a)
    for name in ('summary.json',):
        hashes[BASELINE+'/'+name]=file_sha256(p/BASELINE/name)
    return hashes


def prepare(project, height_cm=180., mass_kg=70., foot_friction=.8, max_force_n=None,
            ratios=None, settings=None, audit_config=None, optimization=None, annotation_path=None):
    p=Path(project);inputs=load_ground_inputs(p,load_inputs(p),height_cm,annotation_path)
    body=dict(height_cm=float(height_cm),ratios=dict(RATIOS if ratios is None else ratios))
    problem=GravityProblem(inputs['reference'],inputs['visibility'],inputs['reference_endpoints'],inputs['schedule'],
        body,inputs['fps'],mass_kg,foot_friction,max_force_n,settings)
    old=json.loads((p/BASELINE/'summary.json').read_text())
    current=input_provenance(p)
    for name,digest in old['provenance']['inputs'].items():
        if not name.startswith('beta_translator/') and current.get(name) != digest:
            raise ValueError('Saved warm start uses different source evidence: '+name)
    ident=dict(inputs=provenance(p,annotation_path),trial='trial1_ground',body=body,
        physics=dict(mass_kg=float(mass_kg),foot_friction=float(foot_friction),max_force_component_n=problem.max_force_n),
        settings=problem.config,support_audit_settings=audit_settings(audit_config),
        optimization_settings=optimization_settings(optimization),ground_model=inputs['ground_annotation'],
        baseline_summary=BASELINE+'/summary.json',baseline_summary_sha256=file_sha256(p/BASELINE/'summary.json'))
    inputs['old_summary']=old;inputs['project']=str(p)
    return inputs,ident


def initialize(inputs, ident, progress=print):
    initial=solve_gravity_sequence(inputs['reference'],inputs['visibility'],inputs['reference_endpoints'],
        inputs['schedule'],ident['body'],inputs['fps'],ident['physics']['mass_kg'],
        ident['physics']['foot_friction'],ident['physics']['max_force_component_n'],ident['settings'],progress)
    initial['trial1_context']=inputs['trial1_context']
    return initial


def ground_diagnose(result, reference_outside, audit_config):
    d=trial1_diagnose(result,reference_outside,audit_config);s=result['schedule'];n=len(result['frames'])
    ground=s['contact_type']=='ground';source=s['ground_evidence']
    d['flags'].update(ground_geometry_uncertain=ground.any(axis=1),
        ground_candidate_unconfirmed=(source=='ground_candidate').any(axis=1),
        ground_contact_conflict=s['ground_conflict'].any(axis=1),
        ground_observation_gap=(result['frames']<=74)&np.any(np.isin(source[:,2:],['unknown','ground_candidate']),axis=1),
        approach_quasistatic_assumption=result['frames']<=74)
    d['issues']=[';'.join(k for k,v in d['flags'].items() if v[t]) for t in range(n)]
    d['counts']={k:int(v.sum()) for k,v in d['flags'].items()}
    return d


def artifacts(stage,inputs,initial,result,before,after):
    project=Path(inputs['project']);s=result['schedule'];a=inputs['ground_annotation'];files=[]
    # Preserve the prior video unchanged, with a portable reference in the review.
    old=inputs['old_summary'];oldvideo=project/BASELINE/old['video_name']
    if file_sha256(oldvideo)!=old['output_sha256'][old['video_name']]:raise ValueError('Old video integrity check failed.')
    baseline_ref='../../support_optimized/'+Path(BASELINE).name+'/'+oldvideo.name
    optimized=f'trial1_support_optimized_{result["body"]["height_cm"]:g}cm_full_sequence.mp4'
    display_result={**result,'schedule':{**s,'labels':s['labels'].astype('<U100').copy()}}
    for t,i in zip(*np.where(s['contact_type']=='ground')):
        display_result['schedule']['labels'][t,i]=f'{s["labels"][t,i]} → {s["support_ids"][t,i]} (conditional)'
    page=review_html(inputs,display_result,after,baseline_ref,optimized)
    page=page.replace('Trial1 · 180 cm',f'Trial1 · {result["body"]["height_cm"]:g} cm · ground experiment')
    page=page.replace('Gravity baseline','Previous 180 cm no-ground video').replace('Gravity baseline','Previous no-ground video')
    page=page.replace('Contacts are unreviewed predictions except the separately shown saved H44 → H28 reference.',
        'Ground intervals were conservatively inspected by the assistant, not user-confirmed. Sole points, normal and friction are assumptions; ground remains conditional. Candidates and ambiguous intervals have no ground force. Other contacts retain their raw predictions.')
    page=page.replace('Before/after metrics','Ground gravity baseline vs refined poses')
    page=page.replace('<p><a href="comparison.csv">','<p><a href="ground_evidence.csv">Ground evidence</a> · <a href="published_comparison.csv">Previous published vs new model</a> · <a href="comparison.csv">')
    page=page.replace('Select baseline to compare the same source frame visually.',
        'The previous video is the preserved 180 cm run with no ground support; its contact model differs. The table always uses the NEW result. comparison.csv compares the fresh ground gravity initialization with the refined poses using identical ground evidence; published_comparison.csv compares the two published models.')
    (stage/'review.html').write_text(page);files.append(stage/'review.html')
    (stage/'ground_annotation.json').write_text(json.dumps(a,indent=2)+'\n');files.append(stage/'ground_annotation.json')
    rows=[]
    for r in inputs['ground_rows']:
        t,i=r['frame'],r['limb_index'];rows.append({**r,
            'effective_contact_type':str(s['contact_type'][t,i]),'effective_support_id':str(s['support_ids'][t,i]),
            'support_active':bool(s['known'][t,i] and s['contact_type'][t,i]=='ground'),
            'candidate_ground_force_x_n':float(result['contact_forces_n'][t,i,0]) if s['contact_type'][t,i]=='ground' else 0.,
            'candidate_ground_force_y_n':float(result['contact_forces_n'][t,i,1]) if s['contact_type'][t,i]=='ground' else 0.})
    pd.DataFrame(rows).to_csv(stage/'ground_evidence.csv',index=False);files.append(stage/'ground_evidence.csv')
    limbfile=stage/'limb_diagnostics.csv';limbs=pd.read_csv(limbfile)
    for name in ('contact_type','support_ids','ground_evidence','ground_conflict','fit_weight','source_confidence'):
        limbs[name]=s[name].ravel()
    cp=contact_points(result['joints'][:,:,2],s)
    for axis,j in (('x',0),('y',1)):
        limbs['contact_point_'+axis+'_cm']=cp[:,:,j].ravel()
        limbs['contact_offset_'+axis+'_cm']=s['contact_offsets_cm'][:,:,j].ravel()
    limbs.to_csv(limbfile,index=False)
    # Include effective support identities/geometry beside unchanged raw labels.
    path=stage/'trajectory.npz'
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    data.update(contact_points_cm=cp,contact_offsets_cm=s['contact_offsets_cm'],support_ids=s['support_ids'],
        contact_type=s['contact_type'],ground_evidence=s['ground_evidence'],known=s['known'],supported=s['supported'])
    np.savez_compressed(path,**data)
    baseline=stage/'ground_baseline_trajectory.npz'
    np.savez_compressed(baseline,poses=initial['poses'],joints_cm=initial['joints'],contact_forces_n=initial['contact_forces_n'],center_of_mass_cm=initial['center_of_mass_cm'])
    files.append(baseline)
    review=stage/'reviewed_reference_comparison.csv';inputs['reference_comparison'].to_csv(review,index=False);files.append(review)
    records=[]
    for key in ('contact_residual','support_balance_failed','support_balance_uncertain','preunload_failed',
                'preunload_uncertain','departing_load_remaining','rapid_pose_change','optimizer_unconverged'):
        records.append(dict(metric=key,previous_published_no_ground=old['flag_counts'][key],
            ground_optimized=after['counts'][key],unit='frames',
            interpretation='Contact model changed; not an isolated optimizer comparison.'))
    pd.DataFrame(records).to_csv(stage/'published_comparison.csv',index=False);files.append(stage/'published_comparison.csv')
    # A native-image evidence sheet for auditing shoe points and floor regions.
    cap=cv2.VideoCapture(str(project/'videos/trial1.mp4'));cells=[]
    evidence={(r['frame'],r['limb_index']):r for r in inputs['ground_rows']}
    for t in (0,6,18,29,40,62,63,67):
        cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read()
        if not ok:raise RuntimeError('Cannot read ground review source frame.')
        keys=a['floor_region_keyframes'];times=[k['frame'] for k in keys];poly=np.array([k['polygon_xy'] for k in keys])
        polygon=np.array([[np.interp(t,times,poly[:,k,j]) for j in range(2)] for k in range(4)],np.int32)
        cv2.polylines(im,[polygon],True,(220,120,20),2)
        for i in (2,3):
            r=evidence[t,i];xy=(round(r['sole_x_px']),round(r['sole_y_px']))
            color=(30,170,255) if r['state']!='off_ground' else (255,150,60)
            cv2.circle(im,xy,6,color,2)
        im=im[650:960,200:490];im=cv2.resize(im,(348,372))
        cv2.rectangle(im,(0,0),(348,64),(245,245,245),-1)
        cv2.putText(im,f'Frame {t}',(6,18),0,.5,(20,20,20),1)
        for j,i in enumerate((2,3)):
            cv2.putText(im,('L: ' if i==2 else 'R: ')+evidence[t,i]['state'],(6,38+18*j),0,.44,(20,20,20),1)
        cells.append(im)
    cap.release()
    sheet=stage/'ground_evidence.jpg';cv2.imwrite(str(sheet),np.vstack([np.hstack(cells[:4]),np.hstack(cells[4:])]))
    files.append(sheet)
    return dict(files=files,summary=dict(trial='Trial1 ground experiment',ground_model=a,
        ground_contact_limb_frames=int((s['contact_type']=='ground').sum()),
        ground_contact_frames=int((s['contact_type']=='ground').any(axis=1).sum()),
        ground_eligible_limb_frames=int((after['support_audit']['eligible']&(s['contact_type']=='ground')).sum()),
        published_model_comparison=records,review_page='review.html',
        comparison_definition='comparison.csv: fresh gravity baseline with ground evidence vs support-refined poses using identical evidence. published_comparison.csv: old and new published models (different support evidence).',
        initialization='Fresh gravity IK with the same ground evidence, then full-sequence IRLS refinement.',
        baseline_optimizer=initial['optimizer'],previous_run=BASELINE,baseline_video_name=baseline_ref,
        geometry_limitations=a['assumptions'],
        early_balance_failed_frames=int(after['flags']['support_balance_failed'][:49].sum())))


def run_trial1_ground_optimization(project,height_cm=180.,mass_kg=70.,foot_friction=.8,max_force_n=None,
        ratios=None,settings=None,audit_config=None,optimization=None,progress=print):
    p=Path(project).resolve()
    inputs,ident=prepare(p,height_cm,mass_kg,foot_friction,max_force_n,ratios,settings,audit_config,optimization)
    root=p/'outputs/trial1/ground_support';key=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:12]
    cached=_verified_cached(root/f'target_{height_cm:g}cm_{key}',ident)
    if cached:
        if progress:progress('Reused verified Trial1 ground-support full sequence.')
        return cached
    if progress:progress('Optimizing all 417 source frames together; ground observations remain conditional.')
    initial=initialize(inputs,ident,progress)
    result=optimize_support_sequence(initial,inputs['fps'],audit_config,optimization,inputs['reference_outside'],progress)
    return publish(p,inputs,initial,result,ident,result_root=root,progress=progress,
        input_provenance_fn=provenance,diagnostic_fn=ground_diagnose,artifact_builder=artifacts)
