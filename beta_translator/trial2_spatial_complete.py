"""Refine the best complete Trial2 candidate and explain residual contradictions."""
from pathlib import Path
import hashlib,json,shutil,time,textwrap
import numpy as np
import pandas as pd
import cv2
from .project_io import file_sha256,convert_video
from .inverse_kinematics import make_body,LIMBS
from .scene3d import contact_span_certificate
from .trial2_scene3d import Trial2Scene
from .trial2_spatial_data import load_trial2,episode_regions,runs
from .trial2_spatial_pipeline import run_trial2_spatial,load_fit,run_case,write_json,common_load_comparison
from .trial2_spatial_optimizer import rank_plans
from .trial2_spatial_render import render_outputs


def burn_check_flags(out,selected):
    """Make every independent failure type visible in the final main video."""
    metrics=pd.read_csv(out/selected/'metrics.csv').fillna('')
    names={'contact_gap':'CONTACT GAP','bone_length':'BONE LENGTH','joint_bend':'BEND','wall_penetration':'WALL','below_floor':'FLOOR',
        'motion_warning':'MOTION','support_failed':'SUPPORT','preunload_failed':'UNLOAD',
        'not_converged':'SOLVER','source_pose_missing':'MISSING SOURCE'}
    path=out/'selected_plan_3d.mp4';cap=cv2.VideoCapture(str(path));raw=out/'flags_encoding.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),30,(1600,1000))
    if not writer.isOpened():raise RuntimeError('Cannot open flag-overlay encoder')
    for f,row in metrics.iterrows():
        ok,im=cap.read()
        if not ok:raise RuntimeError('Selected video is incomplete before flag overlay')
        tags=[names.get(x,x) for x in str(row.issues).split(';') if x]
        cv2.rectangle(im,(1160,878),(1585,987),(248,249,251),-1)
        cv2.putText(im,'Independent check flags:',(1172,900),cv2.FONT_HERSHEY_SIMPLEX,.48,(60,80,100),1,cv2.LINE_AA)
        for i,line in enumerate(textwrap.wrap(' / '.join(tags) if tags else 'No numerical failures on this frame',width=40)):
            cv2.putText(im,line,(1172,924+i*20),cv2.FONT_HERSHEY_SIMPLEX,.43,(45,65,185) if tags else (50,135,50),1,cv2.LINE_AA)
        writer.write(im)
        poster=out/f'frame_{f:04d}.jpg'
        if poster.exists():cv2.imwrite(str(poster),im)
    assert not cap.read()[0],'Selected video has unexpected extra frames'
    cap.release();writer.release();encoded=out/'flags_h264.mp4';convert_video(raw,encoded);encoded.replace(path);raw.unlink()


def review_failures(project,data,out,selected,height_cm,ratios):
    cfg=json.loads((out/selected/'summary.json').read_text())['config'];fit=load_fit(out/selected)
    scene=Trial2Scene(project,cfg['middle_deg'],cfg['upper_deg'],cfg['spacing_cm'])
    regions=episode_regions(data,scene,height_cm);body=make_body(height_cm,ratios)
    centers=[];radii=[]
    for e in regions['episodes']:
        x0,y0,x1,y1=e['rectangle_px']
        xy=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        offset=np.asarray(e['offset_cm']) if cfg['profile']!='center' else np.zeros(3)
        if cfg['profile']=='center':xy=np.tile(e['center_px'],(4,1))
        points=scene.intersect(xy,e['panel'])+offset;center=points.mean(0)
        centers.append(center);radii.append(float(np.linalg.norm(points-center,axis=1).max()))
    centers=np.asarray(centers);radii=np.asarray(radii);pair_rows=[]
    for t,f in enumerate(data['frames']):
        for row in contact_span_certificate(fit['targets'][t],data['known'][t],body):
            i,j=row['limb_i'],row['limb_j'];a,b=regions['mapping'][t,[i,j]]
            minimum=max(0.,float(np.linalg.norm(centers[a]-centers[b])-radii[a]-radii[b]))
            whole_patch_bound=max(0.,minimum-row['maximum_span_cm'])/2
            pair_rows.append(dict(frame=int(f),limb_a=LIMBS[i],limb_b=LIMBS[j],
                selected_point_lower_bound_cm=row['min_possible_max_endpoint_error_cm'],
                whole_patch_lower_bound_cm=whole_patch_bound,
                scope='Conditional on this panel/camera/scale, patch, offset and body; not human impossibility.'))
    pd.DataFrame(pair_rows).to_csv(out/'contact_span_review.csv',index=False)
    timing=[]
    for l in range(4):
        changed=np.flatnonzero(data['known'][1:,l]&data['known'][:-1,l]&(data['labels'][1:,l]!=data['labels'][:-1,l]))+1
        for f in changed:
            step=float(np.linalg.norm(fit['targets'][f,l]-fit['targets'][f-1,l]))
            timing.append(dict(frame=int(f),limb=LIMBS[l],from_label=str(data['labels'][f-1,l]),to_label=str(data['labels'][f,l]),
                target_step_cm=step,required_endpoint_gap_if_step_at_most_8cm=max(0.,step-8)/2,
                raw_labels_preserved=True,scope='Numerical continuity bound for selected fixed episode targets, not a speed limit for humans.'))
    pd.DataFrame(timing).to_csv(out/'timing_conflicts.csv',index=False)
    m=pd.read_csv(out/selected/'metrics.csv');intervals=[]
    for column in ['contact_failed','geometric_failed','motion_warning']:
        for a,b in runs(m[column].to_numpy(bool)):
            intervals.append(dict(issue=column,start_frame=int(a),end_frame=int(b-1),frames=int(b-a)))
    pd.DataFrame(intervals).to_csv(out/'failure_intervals.csv',index=False)
    bounds=pd.DataFrame(pair_rows).groupby('frame').whole_patch_lower_bound_cm.max()
    report=dict(whole_patch_span_conflict_frames=int((bounds>.1).sum()),max_whole_patch_endpoint_bound_cm=float(bounds.max()),
        instantaneous_known_contact_switches=len(timing),floor_edge_height_disagreement_cm=scene.floor_edge_disagreement,
        missing_source_pose_frames=int((~data['identity_observed']|~np.any(data['visibility']>0,axis=(1,2))).sum()),
        identity_rejected_frames=int((~data['identity_observed']).sum()),ground_proposal_limb_frames=int(data['ground'].sum()),
        physical_validation=False,interpretation='Geometry and timing contradictions are conditional diagnoses; solver failure is reported independently.')
    write_json(out/'failure_review.json',report)
    return report


def run_complete_trial2(project,native_iterations=600,progress=print,**kwargs):
    p=Path(project).resolve();parent_summary=run_trial2_spatial(p,progress=progress,**kwargs)
    parent=Path(parent_summary['output_dir']);config=dict(parent_summary['provenance']['config'])
    hashes=dict(parent_summary['provenance']['input_sha256'])
    hashes[str((parent/'summary.json').relative_to(p))]=file_sha256(parent/'summary.json')
    hashes['beta_translator/trial2_spatial_complete.py']=file_sha256(Path(__file__))
    identity=dict(config={**config,'native_iterations':int(native_iterations)},input_sha256=hashes)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    out=parent.parent/f'target_{config["height_cm"]:g}cm_{key}_native'
    if (out/'summary.json').exists():
        saved=json.loads((out/'summary.json').read_text())
        if saved['provenance']!=identity:raise ValueError('Native cache provenance mismatch')
        for name,digest in saved['output_sha256'].items():
            if not (out/name).is_file() or file_sha256(out/name)!=digest:raise ValueError('Changed native artifact: '+name)
        return {**saved,'output_dir':str(out)}
    if out.exists():raise FileExistsError('Incomplete native run preserved: '+str(out))
    out.mkdir();started=time.monotonic();rows=[]
    for r in parent_summary['comparison']:
        shutil.copytree(parent/r['candidate'],out/r['candidate']);rows.append(dict(r))
    for name in ['ground_evidence.csv','angle_comparison.csv']:shutil.copy2(parent/name,out/name)
    data=load_trial2(p);parent_name=parent_summary['selected_candidate'];initial=load_fit(parent/parent_name)
    cfg=json.loads((parent/parent_name/'summary.json').read_text())['config']
    cfg={**cfg,'max_iterations':int(native_iterations),'stride':1,'analytic_contacts':False}
    row,_=run_case(p,data,out,'native_refinement',cfg,initial=initial,progress=progress)
    row['comparison_caveat']='Extra native-frame pose DOFs and compute; refinement, not equal-budget technique ablation.'
    rows.append(row);ranked,fields=rank_plans(common_load_comparison(out,rows));selected=ranked[0]['candidate']
    pd.DataFrame(ranked).to_csv(out/'candidate_comparison.csv',index=False)
    mt=pd.read_csv(out/selected/'metrics.csv');phases=[]
    for t,f in enumerate(data['frames']):
        upper=mt.iloc[t].upper_contacts;middle=mt.iloc[t].middle_contacts
        phase='Ground departure (proposed)' if data['ground'][t].any() else ('Mixed support across panel hinge' if upper and middle else ('Upper-panel movement' if upper else 'Middle-panel movement / unresolved support'))
        if (data['unload'][t]>0).any():phase+='; pre-release check'
        phases.append(dict(frame=int(f),phase=phase,selected_candidate=selected,selection_reason=ranked[0]['selection_reason'],
            issues=str(mt.iloc[t].issues),geometry_uncertain=True))
    pd.DataFrame(phases).to_csv(out/'phase_explanations.csv',index=False)
    review=review_failures(p,data,out,selected,config['height_cm'],config['ratios'])
    render_outputs(p,data,out,ranked,selected)
    burn_check_flags(out,selected)
    for name,digest in hashes.items():
        if file_sha256(p/name)!=digest:raise RuntimeError('Input changed during native refinement: '+name)
    result=dict(provenance=identity,parent_experiment=str(parent.relative_to(p)),parent_selected_candidate=parent_name,
        selected_candidate=selected,ranking_fields=fields,comparison=ranked,failure_review=review,
        frame_count=len(data['frames']),first_frame=0,last_frame=int(data['frames'][-1]),fps=30.,
        seconds=time.monotonic()-started,parent_seconds=parent_summary['seconds'],physical_validation=False,
        interpretation='Nine complete candidates plus native-frame refinement with additional compute. Geometry, contact and force assumptions remain conditional.',
        output_sha256={str(f.relative_to(out)):file_sha256(f) for f in sorted(out.rglob('*')) if f.is_file()})
    write_json(out/'summary.json',result)
    return {**result,'output_dir':str(out)}
