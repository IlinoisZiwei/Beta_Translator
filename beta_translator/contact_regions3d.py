"""Evidence-bounded episode contacts and explicit wrist/ankle contact connectors.

Masks are projected regions, not measured hold surfaces. Offsets are unmeasured
rigid connector hypotheses. Neither classifier confidence nor contact IDs change.
"""
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .contact_features import LANDMARKS
from .inverse_kinematics import LIMBS
from .pose_adapter import project_plane


def homography(points, matrix):
    p=np.asarray(points)
    q=np.concatenate([p,np.ones((*p.shape[:-1],1),dtype=p.dtype)],axis=-1)@np.asarray(matrix).T
    return q[...,:2]/q[...,2:3]


def interior_rectangle(mask, anchor, radius_px=24):
    """Closed pixel rectangle entirely inside a one-pixel-eroded mask.

All four adjacent pixels at subpixel positions are inside the original mask;
concave gaps and holes are never filled by a convex hull. A small hold can
degenerate to a point, which remains explicit in the episode report.
"""
    safe=cv2.erode(np.asarray(mask,np.uint8),np.ones((3,3),np.uint8))>0
    if not safe.any():safe=np.asarray(mask,bool)
    yy,xx=np.where(safe)
    if not len(xx):raise ValueError('Empty contact mask.')
    d=(xx-anchor[0])**2+(yy-anchor[1])**2
    k=np.argmin(d);x=int(xx[k]);y=int(yy[k]);box=[x,y,x,y]
    for _ in range(4*int(radius_px)+4):
        choices=[]
        for edge,delta in [(0,-1),(1,-1),(2,1),(3,1)]:
            b=box.copy();b[edge]+=delta;x0,y0,x1,y1=b
            if x0<0 or y0<0 or x1>=safe.shape[1] or y1>=safe.shape[0]:continue
            if max(abs(x0-x),abs(x1-x),abs(y0-y),abs(y1-y))>radius_px:continue
            if safe[y0:y1+1,x0:x1+1].all():
                # Prefer balanced expansion; deterministic ties are reproducible.
                choices.append(((x1-x0+1)*(y1-y0+1),-abs((x1-x0)-(y1-y0)),b))
        if not choices:break
        box=max(choices,key=lambda z:z[:2])[2]
    return np.array(box,float)


def build_regions(project,data,height_cm=180.):
    root=Path(project);H=np.asarray(data['old']['geometry']['image_to_wall_cm'])
    z=np.load(root/'outputs/trial3/automatic/wholebody.npz',allow_pickle=False)
    points,scores=z['keypoints'],z['scores']
    align=np.load(root/'outputs/trial3/state/alignment_cache.npz',allow_pickle=False)['matrices']
    episodes=[];mapping=np.full(data['known'].shape,-1,int);evidence=[]
    for limb in range(4):
        labels=data['labels'][:,limb];starts=np.flatnonzero(np.r_[True,labels[1:]!=labels[:-1]])
        for start,stop in zip(starts,np.r_[starts[1:],len(labels)]):
            if not data['known'][start,limb]:continue
            hold=str(labels[start]);mask=np.asarray(data['old']['masks'][hold],bool)
            yy,xx=np.where(mask);pixels=np.c_[xx,yy];tree=cKDTree(pixels)
            samples=[];joint_directions=[];distances=[]
            for f in range(start,stop):
                ids=np.unique(LANDMARKS[limb][1:]);valid=(scores[f,ids]>.5)&np.isfinite(points[f,ids]).all(1)
                if not valid.any():continue
                p=project_plane(points[f,ids[valid]],np.linalg.inv(align[f]));dist,near=tree.query(p)
                k=np.argmin(dist);accepted=bool(dist[k]<=12.)
                evidence.append(dict(frame=int(f),limb=LIMBS[limb],hold=hold,episode=len(episodes),
                    landmark_gap_px=float(dist[k]),accepted=accepted,expert_confirmed=False))
                if not accepted:continue
                close=p[dist<=1.5];anchor=np.median(close,axis=0) if len(close) else p[k]
                anchor=pixels[tree.query(anchor)[1]];samples.append(anchor);distances.append(dist[k])
                jid=LANDMARKS[limb][0]
                if scores[f,jid]>.5 and np.isfinite(points[f,jid]).all():
                    wrist=project_plane(points[f,jid:jid+1],np.linalg.inv(align[f]))[0]
                    delta=project_plane(np.array([wrist,anchor]),H)
                    joint_directions.append(delta[0]-delta[1])
            has_evidence=len(samples)>=3
            if has_evidence:anchor=pixels[tree.query(np.median(samples,axis=0))[1]]
            else:
                center=homography(data['targets'][start,limb,:2],np.linalg.inv(H))
                anchor=pixels[tree.query(center)[1]]
            rect=interior_rectangle(mask,anchor)
            # Failed evidence permits only a fixed conservative point, no search.
            if not has_evidence:rect=np.r_[anchor,anchor]
            prior=np.clip(anchor,rect[:2],rect[2:])
            direction=np.median(joint_directions,axis=0) if len(joint_directions)>=3 else np.array([0.,-1. if limb<2 else 1.])
            direction/=max(float(np.linalg.norm(direction)),1e-12)
            # Body-scaled illustrative offsets; source magnitude is deliberately
            # not transferred because source height and surface depth are unknown.
            planar=height_cm*(.035 if limb<2 else .045)
            outward=height_cm*(.012 if limb<2 else .025)
            offset=np.r_[planar*direction,outward] if has_evidence else np.zeros(3)
            e=dict(episode=len(episodes),limb=LIMBS[limb],limb_index=limb,hold=hold,
                start_frame=int(start),end_frame=int(stop-1),evidence_frames=len(samples),
                evidence_available=has_evidence,expert_confirmed=False,
                center_cm=data['targets'][start,limb].tolist(),anchor_px=anchor.tolist(),
                prior_px=prior.tolist(),rectangle_px=rect.tolist(),offset_cm=offset.tolist(),
                direction_observations=len(joint_directions),offset_measured=False,
                region_is_surface=False,region_rule='Closed rectangle inside observed mask, one-pixel erosion when available; max 24 px per edge.',
                offset_rule='Source-derived planar direction only; hand 0.035h + 0.012h outward; foot 0.045h + 0.025h outward.')
            mapping[start:stop,limb]=len(episodes);episodes.append(e)
    return dict(episodes=episodes,mapping=mapping,H=H,evidence=evidence)


def contacts_from_parameters(parameters,regions,profile):
    """One contact position per episode, shared by every supporting native frame."""
    e=regions['episodes'];uv=np.asarray(parameters).reshape(-1,2)
    box=np.array([x['rectangle_px'] for x in e]);prior=np.array([x['prior_px'] for x in e])
    if profile=='center':p=np.array([x['center_cm'] for x in e],dtype=uv.dtype)
    else:
        xy=prior if profile=='evidence_point' else box[:,:2]+uv*(box[:,2:]-box[:,:2])
        p=np.c_[homography(xy,regions['H']),np.zeros(len(e))]
    offset=np.array([x['offset_cm'] for x in e]) if profile.startswith('region_offset') else np.zeros((len(e),3))
    if profile=='region_offset_half':offset*=.5
    index=np.maximum(regions['mapping'],0);known=regions['mapping']>=0
    contact=p[index]*known[...,None];delta=offset[index]*known[...,None]
    return contact,delta,p


def knot_basis(data,stride=4):
    """One global piecewise-linear pose basis; every native frame has residuals.

    Contact transition neighbors are included so the basis need not interpolate
    swivel/free direction conventions across a long interval. This is not a
    concatenation of locally selected solutions.
    """
    from scipy.sparse import csr_matrix
    n=len(data['frames'])
    if stride<1 or int(stride)!=stride:raise ValueError('Positive integer knot stride required.')
    change=np.flatnonzero(np.any(data['labels'][1:]!=data['labels'][:-1],axis=1))+1
    knots=np.unique(np.clip(np.r_[np.arange(0,n,stride),n-1,change-1,change,change+1],0,n-1)).astype(int)
    right=np.clip(np.searchsorted(knots,np.arange(n),side='right'),1,len(knots)-1);left=right-1
    if len(knots)==1:return knots,csr_matrix(np.ones((n,1)))
    fraction=(np.arange(n)-knots[left])/(knots[right]-knots[left])
    B=csr_matrix((np.r_[1-fraction,fraction],(np.r_[np.arange(n),np.arange(n)],np.r_[left,right])),shape=(n,len(knots)))
    return knots,B
