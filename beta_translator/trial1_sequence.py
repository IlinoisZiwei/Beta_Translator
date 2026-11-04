"""Verified Trial1 native frames under its existing uniform image-scale model."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd

from .contact_review import load_review_bundle
from .full_sequence import build_schedule
from .pose_adapter import LIMB_LANDMARKS,to_frame0,pose_row_to_vector,angle_visibility_weights,project_plane
from .project_io import file_sha256,load_alignment,load_reviewed_holds,load_annotations
from .inverse_kinematics import LIMBS

TRIAL1_VIDEO_SHA256='db656ed38a93a619f6df7dbd65804718860aa9191e650a1a92952ab6063a007a'
TRIAL1_SCALE=float(np.hypot(80,-21)/46)


def input_provenance(project):
    p=Path(project)
    names=['videos/trial1.mp4','outputs/state/reviewed_hold_map.npz','outputs/state/alignment_cache.npz',
        'outputs/state/pose_manifest.json','outputs/tables/pose_coordinates.csv','outputs/automatic/wholebody.npz',
        'outputs/automatic/predictions/automatic_contacts.json','outputs/automatic/predictions/limb_contacts.csv',
        'outputs/automatic/predictions/contact_events.csv','data/H44_H28_confirmed_reference.json',
        'models/pose_landmarker.task','models/contact_v1/deployment.json']
    config=json.loads((p/'models/contact_v1/deployment.json').read_text())
    for record in config['checkpoints']:
        if Path(record['file']).name!=record['file']:raise ValueError('Invalid model filename.')
        names.append('models/contact_v1/'+record['file'])
    if (p/'outputs/state/identity_guard.npz').exists():names.append('outputs/state/identity_guard.npz')
    hashes={name:file_sha256(p/name) for name in names}
    for name in ('trial1_sequence.py','gravity_sequence.py','full_sequence.py','full_sequence_pipeline.py',
                 'inverse_kinematics.py','motion_ik.py','pose_adapter.py','project_io.py','contact_review.py',
                 'identity_tracking.py','support_audit.py','support_optimization.py','support_optimization_pipeline.py',
                 'gravity_sequence_pipeline.py','contact_geometry.py'):
        hashes['beta_translator/'+name]=file_sha256(Path(__file__).with_name(name))
    return hashes


def compare_confirmed_reference(schedule,reference):
    """Audit saved human evidence; never overwrite model labels with that evidence."""
    start,end=reference['frames']['before'],reference['frames']['after']
    release,arrival=reference['motion']['release_frame'],reference['motion']['arrival_frame']
    moving=reference['moving_limb'];rows=[]
    for t in range(start,end+1):
        for i,limb in enumerate(LIMBS):
            if limb==moving:
                expected=reference['contacts']['before'][limb] if t<release else ('free' if t<arrival else reference['contacts']['after'][limb])
            elif reference['motion']['support_contacts_maintained']:
                expected=reference['contacts']['before'][limb]
            else:continue
            predicted=str(schedule['labels'][t,i])
            rows.append(dict(frame=t,limb=limb,reviewed_contact=expected,predicted_contact=predicted,
                confidence=float(schedule['confidence'][t,i]),agrees=predicted==expected,
                evidence='Saved user-confirmed H44-H28 reference; no labels changed.'))
    return pd.DataFrame(rows)


def load_inputs(project):
    project=Path(project).resolve();video=project/'videos/trial1.mp4';root=project/'outputs'
    if file_sha256(video)!=TRIAL1_VIDEO_SHA256:
        raise ValueError('Trial1 uniform scale belongs to the supplied video/reference frame 0 only.')
    prediction,contacts=load_review_bundle(video,root,root/'automatic/wholebody.npz',project/'models/contact_v1',root/'automatic/predictions')
    records,masks,run_id,selected=load_reviewed_holds(root/'state/reviewed_hold_map.npz',video,reference_frame_number=0)
    with np.load(root/'state/reviewed_hold_map.npz',allow_pickle=False) as z:background=z['reference_frame'].copy()
    h,w=background.shape[:2]
    cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if count!=417 or not np.isclose(fps,30):raise ValueError('Expected all 417 native Trial1 frames at 30 FPS.')
    reference=load_annotations(project/'data/H44_H28_confirmed_reference.json',video,run_id,selected,count)
    if not np.isclose(reference['body']['px_per_cm'],TRIAL1_SCALE,atol=1e-10,rtol=0):raise ValueError('Saved Trial1 scale changed.')
    with np.load(root/'state/alignment_cache.npz',allow_pickle=False) as z:settings=json.loads(str(z['settings'].item()))
    alignment=load_alignment(root/'state/alignment_cache.npz',video,root/'tables/pose_coordinates.csv',settings)
    if alignment is None:raise ValueError('Trial1 alignment does not match the pose/video inputs.')
    manifest=json.loads((root/'state/pose_manifest.json').read_text())
    if (manifest['inputs']['video_sha256']!=file_sha256(video)
            or manifest['inputs']['model_sha256']!=file_sha256(project/'models/pose_landmarker.task')
            or manifest['pose_sha256']!=file_sha256(root/'tables/pose_coordinates.csv')):
        raise ValueError('Trial1 observed pose manifest is stale.')
    poses=pd.read_csv(root/'tables/pose_coordinates.csv').set_index('frame')
    if not poses.index.is_unique or set(poses.index)!=set(range(count)):
        raise ValueError('Every native Trial1 frame requires an observed pose; no hidden interpolation.')
    matrix=np.array([[1/TRIAL1_SCALE,0,0],[0,-1/TRIAL1_SCALE,h/TRIAL1_SCALE],[0,0,1.]])
    hold_positions={r['hold_id']:project_plane([[r['cx_px'],r['cy_px']]],matrix)[0].tolist() for r in records}
    covered=[r['hold_id'] for r in records if 0<=r['cx_px']<w and 0<=r['cy_px']<h]
    geometry=dict(image_to_wall_cm=matrix.tolist(),hold_positions_cm=hold_positions,supported_hold_ids=covered,
        reference_frame_number=0,candidate_run_id=run_id,coverage_kind='mapped reference image; no plane-calibration hull',
        px_per_cm=TRIAL1_SCALE,scale_source='46 cm measured image gap; uniform scale, image-up gravity assumed')
    schedule=build_schedule(contacts,np.arange(count),geometry,prediction['accept_threshold'])
    names=list(dict.fromkeys(name for chain in LIMB_LANDMARKS.values() for name in chain))
    references=[];weights=[];endpoints=[];outside=[];aligned=[]
    for t in range(count):
        if t not in alignment or alignment[t][2]!='ok':raise ValueError(f'Frame {t}: unreliable Trial1 wall alignment.')
        row=poses.loc[t].copy()
        if 'pose_source' in row and not str(row.pose_source).startswith('observed'):raise ValueError('Interpolated pose is not observed evidence.')
        points=to_frame0([[row[n+'_x_px'],row[n+'_y_px']] for n in names],alignment,t)
        if not np.isfinite(points).all():raise ValueError(f'Frame {t}: invalid pose coordinates.')
        for name,xy in zip(names,points):row[name+'_x_px'],row[name+'_y_px']=xy
        references.append(pose_row_to_vector(row,None,None,matrix));weights.append(angle_visibility_weights(row))
        ends=[[row[LIMB_LANDMARKS[l][-1]+'_x_px'],row[LIMB_LANDMARKS[l][-1]+'_y_px']] for l in LIMBS]
        endpoints.append(project_plane(ends,matrix));aligned.append(points)
        outside.append(bool(np.any((points[:,0]<0)|(points[:,0]>=w)|(points[:,1]<0)|(points[:,1]>=h))))
    review=compare_confirmed_reference(schedule,reference)
    disagreement=np.zeros(count,bool);disagreement[review.loc[~review.agrees,'frame'].unique()]=True
    return dict(reference=np.asarray(references),visibility=np.asarray(weights),reference_endpoints=np.asarray(endpoints),
        reference_outside=np.asarray(outside),aligned_reference=np.asarray(aligned),schedule=schedule,fps=float(fps),
        geometry=geometry,background=background,records=records,masks=masks,selected=selected,prediction=prediction,
        trial_id='trial1',trial_label='Trial1',source_video=str(video),review_reference=reference,reference_comparison=review,
        trial1_context=dict(reference_disagreement=disagreement),
        render_assumptions='UNIFORM SCALE + IMAGE-UP GRAVITY | assumed mass {mass:g} kg | perspective/depth not recovered',
        render_contact_legend='Eligible = reached, confident, mapped contact UNDER UNIFORM SCALE. Conditional adds uncertain labels.')
