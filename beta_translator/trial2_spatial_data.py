"""Frozen Trial2 observations, explicit ground proposals and episode contact patches."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
from .contact_features import LANDMARKS
from .inverse_kinematics import LIMBS,make_body
from .pose_adapter import LIMB_LANDMARKS
from .contact_regions3d import homography,interior_rectangle
from .trial2_scene3d import line_height
from .ik3d import kinematics
from .project_io import validate_reference_alignment,file_sha256


def runs(mask):
    a=np.asarray(mask,bool);edges=np.flatnonzero(np.diff(np.r_[False,a,False]))
    return list(zip(edges[::2],edges[1::2]))


def load_trial2(project):
    p=Path(project);root=p/'outputs/trial2'
    reference=validate_reference_alignment(root/'state/reviewed_hold_map.npz',root/'state/alignment_cache.npz')
    if reference!=450:raise ValueError('This Trial2 panel hypothesis is annotated on reference frame 450.')
    align=np.load(root/'state/alignment_cache.npz',allow_pickle=False);n=len(align['frames'])
    pose=pd.read_csv(root/'tables/pose_coordinates.csv')
    if pose.frame.duplicated().any():raise ValueError('Duplicate pose frame.')
    # Target-identity rejection removes rows. Restore the native time axis with
    # missing observations, never compress time or infer observations from display poses.
    pose=pose.set_index('frame').reindex(np.arange(n))
    table=pd.read_csv(root/'automatic/predictions/limb_contacts.csv').fillna('unknown')
    if table.duplicated(['frame','limb']).any():raise ValueError('Duplicate native-frame contacts.')
    labels=np.stack([table[table.limb==l].set_index('frame').loc[np.arange(n),'candidate_hold'].to_numpy(str) for l in LIMBS],axis=1)
    confidence=np.stack([table[table.limb==l].set_index('frame').loc[np.arange(n),'confidence'].to_numpy(float) for l in LIMBS],axis=1)
    if not np.array_equal(align['frames'],np.arange(n)):raise ValueError('Alignment native-frame mismatch.')
    matrices=align['matrices'];inverse=np.linalg.inv(matrices)
    observed=np.array([[[pose[f'{name}_{axis}_px'].to_numpy() for axis in ('x','y')] for name in LIMB_LANDMARKS[l]] for l in LIMBS]).transpose(3,0,1,2)
    visibility=np.array([[pose[f'{name}_visibility'].to_numpy() for name in LIMB_LANDMARKS[l]] for l in LIMBS]).transpose(2,0,1)
    observed=np.stack([homography(v,H) for v,H in zip(observed,inverse)])
    guard=np.load(root/'state/identity_guard.npz',allow_pickle=False)['observed']
    visibility=np.nan_to_num(visibility)*guard[:,None,None]
    visibility[~np.isfinite(observed).all(-1)]=0
    # Interpolation seeds the optimizer only. Missing evidence stays zero-weight.
    filled=observed.copy().reshape(n,-1)
    for i in range(filled.shape[1]):
        valid=np.isfinite(filled[:,i]);filled[:,i]=np.interp(np.arange(n),np.flatnonzero(valid),filled[valid,i])
    observed=filled.reshape(n,4,3,2)
    wb=np.load(root/'automatic/wholebody.npz',allow_pickle=False)
    points=np.stack([homography(v,H) for v,H in zip(wb['keypoints'],inverse)])
    scores=wb['scores']*guard[:,None]
    holds=np.load(root/'state/reviewed_hold_map.npz',allow_pickle=False);meta=json.loads(str(holds['metadata']))
    if meta['video_sha256']!=file_sha256(p/'videos/trial2.mp4'):raise ValueError('Hold map belongs to a different video.')
    records={x['hold_id']:x for x in meta['records']}
    masks={l:holds[l] for l in np.unique(labels) if l in records}
    known=np.isin(labels,list(masks));ground=np.zeros((n,4),bool);evidence=[]
    spec=json.loads((p/'data/trial2_spatial_hypotheses.json').read_text())
    for limb in (2,3):
        ids=np.unique(LANDMARKS[limb][1:]);foot=np.median(points[:,ids],axis=1)
        visible=np.sum(scores[:,ids]>.5,axis=1)>=2
        dy=foot[:,1]-line_height(foot,spec['floor_edge_px'])
        step=np.linalg.norm(np.diff(foot,axis=0,prepend=foot[:1]),axis=1)
        proposal=visible&(abs(dy)<=18)&(step<=4)&~known[:,limb]&(foot[:,0]>350)&(foot[:,0]<1380)
        for a,b in runs(proposal):
            if b-a>=10:ground[a:b,limb]=True
        evidence.extend(dict(frame=int(f),limb=LIMBS[limb],raw_label=str(labels[f,limb]),
            floor_edge_gap_px=float(abs(dy[f])),source_step_px=float(step[f]),visible=bool(visible[f]),
            ground_proposed=bool(ground[f,limb]),manually_confirmed=False) for f in range(n))
    effective=labels.astype('U32');effective[ground]='ground_proposal'
    unload=np.zeros((n,4));active=known|ground
    for i in range(4):
        for f in np.flatnonzero(active[:-1,i]&(effective[1:,i]!=effective[:-1,i]))+1:
            for j in range(max(0,f-6),f):
                if effective[j,i]==effective[f-1,i]:unload[j,i]=max(unload[j,i],(j-f+7)/6)
    return dict(frames=np.arange(n),fps=30.,raw_labels=labels,labels=effective,confidence=confidence,
        known=active,hold_known=known,ground=ground,unload=unload,observed_px=observed,
        visibility=visibility,points=points,scores=scores,records=records,masks=masks,
        identity_observed=guard,alignment_status=align['statuses'],ground_evidence=evidence,
        reference_image=holds['reference_frame'],source_height_cm=173.)


from .body_regions import episode_regions


def seed_pose(data,scene,height_cm,ratios=None,depth_cm=25.):
    px=data['observed_px'];hip_px=px[:,2:,0].mean(1);panel=scene.panel(hip_px)
    # Same panel for each seed body; final optimization uses perspective projection,
    # fixed bones and each contact's own panel instead of this seed unprojection.
    rough=scene.intersect(px,panel[:,None,None],depth_cm)
    rough=np.clip(rough,[-800,-100,-900],[800,1200,500])
    hip=rough[:,2:,0].mean(1);shoulder=rough[:,:2,0].mean(1);torso=shoulder-hip
    q=np.zeros((len(px),22));q[:,:3]=hip
    q[:,3]=np.arctan2(-torso[:,0],torso[:,1]);q[:,5]=np.arctan2(torso[:,2],np.linalg.norm(torso[:,:2],axis=1))
    vectors=np.diff(rough,axis=2)
    for l in range(4):
        for bone in range(2):
            v=vectors[:,l,bone];q[:,6+l*4+bone*2]=np.unwrap(np.arctan2(v[:,0],-v[:,1]))
            q[:,7+l*4+bone*2]=np.arctan2(v[:,2],np.linalg.norm(v[:,:2],axis=1))
    q[:,3]=np.unwrap(q[:,3])
    return q
