"""Convert each trial's evidence to one shared schema; no body constraints here."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from .contact_regions3d import homography
from .body_regions import episode_regions
from .body_scenes import ProjectedScene, PlanarScene
from .scene3d import tilt_rotation
from .ik3d import rotation
from .inverse_kinematics import LIMBS
from .pose_adapter import LIMB_LANDMARKS


def load_config(project, trial):
    path=Path(trial)
    if path.suffix!='.json':path=Path(project)/'configs/trials'/f'{trial}.json'
    cfg=json.loads(path.read_text())
    if cfg['schema_version']!=1:raise ValueError('Unsupported trial config')
    return cfg


def unload_weights(known, labels):
    out=np.zeros(known.shape)
    for limb in range(4):
        for f in np.flatnonzero(known[:-1,limb]&(labels[1:,limb]!=labels[:-1,limb]))+1:
            for j in range(max(0,f-6),f):
                if labels[j,limb]==labels[f-1,limb]:out[j,limb]=max(out[j,limb],(j-f+7)/6)
    return out


def normalized_observations(project, root, data):
    """Original observations in the reviewed reference-image coordinate system."""
    r=Path(project)/root
    with np.load(r/'state/alignment_cache.npz',allow_pickle=False) as z:inverse=np.linalg.inv(z['matrices'])
    table=pd.read_csv(r/'tables/pose_coordinates.csv').set_index('frame').reindex(data['frames'])
    px=np.array([[[table[f'{name}_{axis}_px'].to_numpy() for axis in ('x','y')]
                 for name in LIMB_LANDMARKS[l]] for l in LIMBS]).transpose(3,0,1,2)
    vis=np.array([[table[f'{name}_visibility'].to_numpy() for name in LIMB_LANDMARKS[l]] for l in LIMBS]).transpose(2,0,1)
    px=np.stack([homography(v,H) for v,H in zip(px,inverse)])
    with np.load(r/'automatic/wholebody.npz',allow_pickle=False) as z:
        points=np.stack([homography(v,H) for v,H in zip(z['keypoints'],inverse)]);scores=z['scores'].copy()
    valid=np.isfinite(px).all(-1);vis=np.where(valid,np.nan_to_num(vis),0.)
    observed=np.any(vis>0,axis=(1,2))
    guard=r/'state/identity_guard.npz'
    if guard.exists():
        with np.load(guard,allow_pickle=False) as z:observed &= z['observed'].astype(bool)
        vis *= observed[:,None,None];scores *= observed[:,None]
    # Finite seed placeholders carry zero evidence weight, never new observations.
    px=np.nan_to_num(px)
    return dict(observed_px=px,visibility=vis,points=points,scores=scores,identity_observed=observed)


def load_trial(project, config, height_cm=180.):
    p=Path(project);cfg=load_config(p,config) if isinstance(config,(str,Path)) else config
    kind=cfg['adapter'];root=cfg['input_root'];baseline=p/cfg['baseline']
    with np.load(baseline,allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    if kind=='joined_panels':
        from .trial2_spatial_data import load_trial2
        from .trial2_scene3d import Trial2Scene
        data=load_trial2(p);scene=Trial2Scene(p,**cfg['scene'])
        legacy=old
    elif kind=='projected_ground':
        from .trial1_sequence import load_inputs
        from .ground_support import load_ground_inputs
        source=load_ground_inputs(p,load_inputs(p),height_cm)
        s=source['schedule'];n=len(s['frames']);ground=s['contact_type']=='ground'
        labels=np.where(ground,s['support_ids'],s['labels'])
        data=dict(frames=s['frames'],fps=source['fps'],known=s['known'],hold_known=s['known']&~ground,
            ground=ground,labels=labels,raw_labels=s['labels'],confidence=s['confidence'],
            records={r['hold_id']:r for r in source['records']},masks=source['masks'])
        scene=ProjectedScene(source['geometry']['image_to_wall_cm'],source['background'].shape[1::-1],cfg['scene']['depth_cm'])
        data.update(normalized_observations(p,root,data))
        joints=np.concatenate([old['joints_cm'],np.full((*old['joints_cm'].shape[:-1],1),cfg['scene']['depth_cm'])],axis=-1)
        values=np.zeros((n,22));values[:,:2]=old['poses'][:,:2];values[:,2]=cfg['scene']['depth_cm'];values[:,3]=old['poses'][:,2]
        legacy=dict(joints=joints,values=values)
    elif kind=='calibrated_plane':
        from .trial3_3d_pipeline import load_inputs
        source=load_inputs(p);s=source['old'];n=len(source['frames'])
        data=dict(frames=source['frames'],fps=source['fps'],known=source['known'],hold_known=source['known'],
            ground=np.zeros((n,4),bool),labels=source['labels'],raw_labels=source['labels'],confidence=source['confidence'],
            records={r['hold_id']:r for r in s['records']},masks=s['masks'])
        data.update(normalized_observations(p,root,data))
        scene=PlanarScene(source['camera'],s['background'].shape[1::-1],**cfg['scene'])
        Q=tilt_rotation(cfg['scene']['tilt_deg']);shift=np.array([0.,cfg['scene']['origin_y_cm'],0.])
        joints=old['joints_cm']@Q.T+shift;values=old['values'].copy()
        values[:,:3]=values[:,:3]@Q.T+shift
        values[:,3:6]=Rotation.from_matrix(Q@rotation(values[:,3:6])).as_euler('ZYX')
        legacy=dict(joints=joints,values=values)
    else:raise ValueError('Unknown scene/evidence adapter: '+kind)
    n=len(data['frames'])
    if n!=cfg['frame_count'] or data['fps']!=cfg['fps']:raise ValueError('Config/video native time axis mismatch')
    if not np.array_equal(data['frames'],np.arange(n)):raise ValueError('Native frames must remain complete and ordered')
    if len(legacy['joints'])!=n:raise ValueError('Baseline frame count mismatch')
    data['unload']=unload_weights(data['known'],data['labels'])
    regions=episode_regions(data,scene,height_cm)
    # Every adapter produces the same persistent episode coordinates.
    boxes=np.array([e['rectangle_px'] for e in regions['episodes']]);anchors=np.array([e['anchor_px'] for e in regions['episodes']])
    legacy['episode_uv']=np.divide(anchors-boxes[:,:2],boxes[:,2:]-boxes[:,:2],out=np.full_like(anchors,.5),where=boxes[:,2:]!=boxes[:,:2])
    return cfg,data,scene,regions,legacy
