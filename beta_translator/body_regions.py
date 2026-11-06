"""Shared image-evidence contact regions; no trial-specific body rules."""
import numpy as np
from scipy.spatial import cKDTree
from .contact_features import LANDMARKS
from .inverse_kinematics import LIMBS
from .contact_regions3d import interior_rectangle

def episode_regions(data,scene,height_cm):
    episodes=[];mapping=np.full(data['known'].shape,-1,int);evidence=[]
    for limb in range(4):
        labels=data['labels'][:,limb];starts=np.flatnonzero(np.r_[True,labels[1:]!=labels[:-1]])
        for a,b in zip(starts,np.r_[starts[1:],len(labels)]):
            if not data['known'][a,limb]:continue
            hold=str(labels[a]);ids=np.unique(LANDMARKS[limb][1:]);samples=[];directions=[]
            ground=bool(data['ground'][a,limb])
            if ground:
                anchor=np.median(data['points'][a:b][:,ids].reshape(-1,2),axis=0)
                box=np.r_[anchor,anchor];center=anchor;panel=3;evidence_count=b-a
            else:
                rec=data['records'][hold];center=np.array([rec['cx_px'],rec['cy_px']]);panel=int(scene.panel(center))
                mask=data['masks'][hold].copy();yy,xx=np.where(mask)
                on_panel=scene.panel(np.c_[xx,yy])==panel;mask[yy[~on_panel],xx[~on_panel]]=False
                yy,xx=np.where(mask);pixels=np.c_[xx,yy];tree=cKDTree(pixels)
                for f in range(a,b):
                    valid=data['scores'][f,ids]>.5
                    if not valid.any():continue
                    pts=data['points'][f,ids[valid]];dist,k=tree.query(pts);best=np.argmin(dist)
                    accepted=bool(dist[best]<=12)
                    evidence.append(dict(frame=int(f),limb=LIMBS[limb],hold=hold,episode=len(episodes),
                        gap_px=float(dist[best]),accepted=accepted,expert_confirmed=False))
                    if not accepted:continue
                    anchor=pixels[k[best]];samples.append(anchor)
                    jid=LANDMARKS[limb][0]
                    if data['scores'][f,jid]>.5:
                        delta=scene.intersect(data['points'][f,jid],panel)-scene.intersect(anchor,panel)
                        directions.append(delta)
                anchor=pixels[tree.query(np.median(samples,axis=0) if len(samples)>=3 else center)[1]]
                box=interior_rectangle(mask,anchor)
                if len(samples)<3:box=np.r_[anchor,anchor]
                anchor=np.clip(anchor,box[:2],box[2:]);evidence_count=len(samples)
            normal=scene.normals[panel]
            direction=np.median(directions,axis=0) if len(directions)>=3 else np.array([0.,-1. if limb<2 else 1.,0.])
            direction=direction-(direction@normal)*normal;direction/=max(np.linalg.norm(direction),1e-12)
            offset=(.035 if limb<2 else .045)*height_cm*direction+(.012 if limb<2 else .025)*height_cm*normal
            if ground:offset=np.array([0.,.035*height_cm,0.])
            if not ground and evidence_count<3:offset=np.zeros(3)
            episodes.append(dict(episode=len(episodes),limb=LIMBS[limb],limb_index=limb,hold=hold,
                start_frame=int(a),end_frame=int(b-1),panel=panel,rectangle_px=box.tolist(),anchor_px=anchor.tolist(),
                center_px=center.tolist(),offset_cm=offset.tolist(),evidence_frames=evidence_count,
                ground_proposal=ground,expert_confirmed=False,offset_measured=False))
            mapping[a:b,limb]=len(episodes)-1
    return dict(episodes=episodes,mapping=mapping,evidence=evidence)


