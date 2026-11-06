"""Shared fixed-length body and reduced anatomical arm kinematics.

Coordinates/ranges: pinned O2MConverter MoBL .osim, data/trial2_arm_model_sources.json.
MoBL local axes are used for shoulder, elbow, forearm and wrist. Shoulder-girdle
translations, scapular rhythm, non-collinear bone offsets and ligament forces are
omitted to retain the existing fixed-length Beta Translator skeleton. Left arm is
a sagittal reflection of the right chain. Hand pose is inferred, not observed.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .ik3d import rotation, direction
from .inverse_kinematics import MASS_FRACTIONS, CENTER_OF_MASS_FRACTIONS

from .body_constraints import ARM_NAMES, ARM_LIMITS_DEG, ARM_SPEED_DEG_S, ARM_ACCEL_DEG_S2, DEFAULT_LIMITS
QDIM = 28
# Source axes, in OpenSim's forward/up/right coordinates.
AXES = np.array([[.0048,.99908918,.04240001],[-.99826136,.0023,.05889802],
                 [.04940001,.03660001,.99810825],[-.01716099,.99266564,-.11966796],
                 [-.819064,-.135611,-.557444],[.95642673,-.25220693,.14710396]])
AXES /= np.linalg.norm(AXES,axis=1,keepdims=True)
# Forward -> toward wall (-Z), up -> +Y, right -> +X. Proper rotation.
OSIM_TO_BODY = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
MIRROR = np.diag([-1.,1.,1.])


def axis_rotation(angle, axis):
    x,y,z=axis
    K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return (np.eye(3)+np.sin(angle)[...,None,None]*K+
            (1-np.cos(angle))[...,None,None]*(K@K))


def arm_frames(q):
    """Right-side local segment frames; signed elbow hinge, no swivel circle."""
    p,e,r,b,s,d,f=np.moveaxis(q,-1,0)
    A=axis_rotation(p,AXES[0])@axis_rotation(e,AXES[1])@axis_rotation(r-p,AXES[0])
    F=A@axis_rotation(b,AXES[2])
    H=F@axis_rotation(s,AXES[3])@axis_rotation(d,AXES[4])@axis_rotation(f,AXES[5])
    return tuple(OSIM_TO_BODY@x@OSIM_TO_BODY.T for x in (A,F,H))


def kinematics(q, lengths, height_cm):
    q=np.atleast_2d(q);R=rotation(q[:,3:6]);hip=q[:,:3]
    up,right=R[:,:,1],R[:,:,0];shoulder=hip+lengths['torso']*up
    roots=np.stack([shoulder-lengths['shoulder_w']/2*right,shoulder+lengths['shoulder_w']/2*right,
                    hip-lengths['hip_w']/2*right,hip+lengths['hip_w']/2*right],axis=1)
    limbs=[];orientations=[];palms=[];hand_tips=[]
    for i in range(2):
        local=arm_frames(q[:,6+7*i:13+7*i])
        if i==0:local=tuple(MIRROR@x@MIRROR for x in local)
        A,F,H=(R@x for x in local)
        E=roots[:,i]-lengths['upper_arm']*A[:,:,1]
        W=E-lengths['forearm']*F[:,:,1]
        # Same 0.035h / 0.012h connector lengths as the legacy hand offset;
        # orientation now follows the wrist chain, rather than a free connector.
        tip=W-.035*height_cm*H[:,:,1]
        contact=tip-.012*height_cm*H[:,:,2]
        limbs.append(np.stack([roots[:,i],E,W],axis=1))
        orientations.extend([A,F,H]);palms.append(contact);hand_tips.append(tip)
    legs=q[:,20:28].reshape(-1,2,4)
    for i in range(2):
        a=legs[:,i];K=roots[:,i+2]+lengths['thigh']*direction(a[:,0],a[:,1])
        E=K+lengths['shank']*direction(a[:,2],a[:,3])
        limbs.append(np.stack([roots[:,i+2],K,E],axis=1))
    j=np.stack(limbs,axis=1);f,c=MASS_FRACTIONS,CENTER_OF_MASS_FRACTIONS
    head=shoulder+c['head']*lengths['torso']*up
    com=f['head']*head+f['torso']*(hip+c['torso']*lengths['torso']*up)
    for i in range(4):
        u,l=('upper_arm','forearm') if i<2 else ('thigh','shank')
        com+=f[u]*(j[:,i,0]+c[u]*(j[:,i,1]-j[:,i,0]))+f[l]*(j[:,i,1]+c[l]*(j[:,i,2]-j[:,i,1]))
    return dict(joints=j,com=com,head=head,palm_contacts=np.stack(palms,axis=1),
                hand_tips=np.stack(hand_tips,axis=1),orientations=np.stack([R,*orientations],axis=1))


def bounds_and_rates():
    lo=np.r_[[-1000,-30,-1500],np.deg2rad([-110,-120,-90]),np.tile(np.deg2rad(ARM_LIMITS_DEG[:,0]),2),np.tile([-30,-np.pi/2],4)]
    hi=np.r_[[1000,1600,1000],np.deg2rad([110,120,90]),np.tile(np.deg2rad(ARM_LIMITS_DEG[:,1]),2),np.tile([30,np.pi/2],4)]
    L=DEFAULT_LIMITS
    speed=np.r_[[L.hip_speed_cm_s]*3,np.deg2rad([L.torso_speed_deg_s]*3),np.tile(np.deg2rad(ARM_SPEED_DEG_S),2),np.deg2rad([L.leg_speed_deg_s]*8)]
    accel=np.r_[[L.hip_acceleration_cm_s2]*3,np.deg2rad([L.torso_acceleration_deg_s2]*3),np.tile(np.deg2rad(ARM_ACCEL_DEG_S2),2),np.deg2rad([L.leg_acceleration_deg_s2]*8)]
    return lo,hi,speed,accel


def initialize(legacy, lengths, height_cm, progress=None):
    """Prioritize wrist endpoints; an abnormal legacy elbow is only a weak hint.

    Fit against the actual bounded torso roots. Retry plane/elevation seeds if
    the wrist remains >0.5 cm away, rather than inheriting a bad elbow branch.
    This initialization is always followed by the constrained sequence solve.
    """
    j=legacy['joints'];n=len(j);q=np.zeros((n,QDIM));q[:,:3]=legacy['values'][:,:3]
    q[:,3:6]=Rotation.from_matrix(rotation(legacy['values'][:,3:6])).as_euler('ZYX')
    lo,hi,_,_=bounds_and_rates();q[:,3:6]=np.clip(q[:,3:6],lo[3:6],hi[3:6])
    for i in range(2):
        prev=np.deg2rad([30,90,-30,70,0,0,0.]);R=rotation(q[:,3:6])
        root=q[:,:3]+lengths['torso']*R[:,:,1]+(-1 if i==0 else 1)*lengths['shoulder_w']/2*R[:,:,0]
        local=np.einsum('tji,tkj->tki',R,j[:,i,1:]-root[:,None])
        if i==0:local=local@MIRROR
        for t in range(n):
            def fun(a):
                A,F,_=arm_frames(a);E=-lengths['upper_arm']*A[:,1];W=E-lengths['forearm']*F[:,1]
                return np.r_[(E-local[t,0])/50,(W-local[t,1])/2,.03*(a-prev)]
            seeds=[prev]
            if t==0:seeds.extend(np.deg2rad([p,e,-30,70,0,0,0.]) for p in [-60,60,120] for e in [45,120])
            fits=[least_squares(fun,np.clip(a,lo[6:13],hi[6:13]),bounds=(lo[6:13],hi[6:13]),max_nfev=45) for a in seeds]
            chosen=min(fits,key=lambda r:r.cost)
            A,F,_=arm_frames(chosen.x)
            wrist=-lengths['upper_arm']*A[:,1]-lengths['forearm']*F[:,1]
            if t>0 and np.linalg.norm(wrist-local[t,1])>.5:
                alternatives=[np.deg2rad([p,e,-30,70,0,0,0.]) for p in [-60,60,120] for e in [45,120]]
                fits.extend(least_squares(fun,a,bounds=(lo[6:13],hi[6:13]),max_nfev=60) for a in alternatives)
                chosen=min(fits,key=lambda r:r.cost)
            prev=chosen.x;q[t,6+7*i:13+7*i]=prev
            if progress and t%200==0:progress(f'arm initialization {i+1}/2 frame {t}/{n}',flush=True)
    for i in range(2):
        vec=np.diff(j[:,i+2],axis=1)
        for b in range(2):
            v=vec[:,b];q[:,20+4*i+2*b]=np.unwrap(np.arctan2(v[:,0],-v[:,1]))
            q[:,21+4*i+2*b]=np.arctan2(v[:,2],np.linalg.norm(v[:,:2],axis=1))
    return np.clip(q,lo,hi)


def rotation_steps_deg(matrices, fps):
    relative=np.swapaxes(matrices[:-1],-1,-2)@matrices[1:]
    cos=(np.trace(relative,axis1=-2,axis2=-1)-1)/2
    return np.rad2deg(np.arccos(np.clip(cos,-1,1)))*fps


def elbow_swivel_rates(j, fps):
    """Parallel-transport diagnostic. Suppress ill-defined near-straight circles."""
    S,E,W=j[:,:2,0],j[:,:2,1],j[:,:2,2]
    a=W-S;a/=np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
    u=E-S-np.sum((E-S)*a,axis=-1,keepdims=True)*a
    radius=np.linalg.norm(u,axis=-1);u/=np.maximum(radius[...,None],1e-12)
    v=np.cross(a[:-1],a[1:]);c=np.sum(a[:-1]*a[1:],axis=-1)
    transported=u[:-1]+np.cross(v,u[:-1])+np.cross(v,np.cross(v,u[:-1]))/np.maximum((1+c)[...,None],1e-10)
    degrees=np.rad2deg(np.arctan2(np.sum(a[1:]*np.cross(transported,u[1:]),axis=-1),np.sum(transported*u[1:],axis=-1)))
    valid=(radius[:-1]>5)&(radius[1:]>5)&(c>0)
    return np.where(valid,abs(degrees)*fps,np.nan)
