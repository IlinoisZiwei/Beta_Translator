"""Anatomical hip–knee–ankle/subtalar chain in the shared body.

Source rotations AND knee translations: pinned OpenSim Gait2392 parameters.
Long-bone geometry is rigid; the tibial origin follows the coupled knee curve.
Hip spacing/segment scale and shoe geometry remain declared project assumptions.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from . import body_kinematics as arms
from .body_reference import stage1_kinematics
from .body_constraints import LEG_LIMITS_DEG,LEG_SPEED_DEG_S,LEG_ACCEL_DEG_S2
from .inverse_kinematics import MASS_FRACTIONS,CENTER_OF_MASS_FRACTIONS

QDIM = 32
# Shoe frame columns: toe, transverse, upward sole normal, in body coordinates.
SHOE_BASIS = np.array([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])


@lru_cache(maxsize=1)
def source():
    return json.loads((Path(__file__).resolve().parents[1]/'data/lower_limb_source.json').read_text())


@lru_cache(maxsize=8)
def spline_coefficients(x,y):
    """Cubic interpolation with SIMM divided-difference end conditions.

    Equivalent linear system to OpenSim SimmSpline (Apache-2.0, Stanford/Authors),
    solved densely here; evaluation retains complex perturbations for derivatives.
    """
    x,y=np.asarray(x),np.asarray(y);h=np.diff(x);slopes=np.diff(y)/h;n=len(x)
    A=np.zeros((n,n));rhs=np.zeros(n)
    for i in range(1,n-1):
        A[i,i-1:i+2]=[h[i-1],2*(h[i-1]+h[i]),h[i]]
        rhs[i]=slopes[i]-slopes[i-1]
    A[0,:2]=[-h[0],h[0]];A[-1,-2:]=[h[-1],-h[-1]]
    rhs[0]=((slopes[2]-slopes[1])/(x[3]-x[1])-(slopes[1]-slopes[0])/(x[2]-x[0]))*h[0]**2/(x[3]-x[0])
    rhs[-1]=-((slopes[-1]-slopes[-2])/(x[-1]-x[-3])-(slopes[-2]-slopes[-3])/(x[-2]-x[-4]))*h[-1]**2/(x[-1]-x[-4])
    c=np.linalg.solve(A,rhs)
    b=slopes-h*(2*c[:-1]+c[1:]);d=np.diff(c)/h
    end_slope=slopes[-1]+h[-1]*(c[-2]+2*c[-1])
    return b,3*c[:-1],d,end_slope


def spline(value,spec):
    x,y=np.asarray(spec['x']),np.asarray(spec['y'])
    b,c,d,end=spline_coefficients(tuple(x),tuple(y))
    k=np.clip(np.searchsorted(x,np.real(value),side='right')-1,0,len(x)-2)
    dx=value-x[k];out=y[k]+dx*(b[k]+dx*(c[k]+dx*d[k]))
    out=np.where(np.real(value)<x[0],y[0]+(value-x[0])*b[0],out)
    return np.where(np.real(value)>x[-1],y[-1]+(value-x[-1])*end,out)


def joint_rotation(q,name):
    """Parent-relative source rotation, converted to project XYZ coordinates."""
    R=np.broadcast_to(np.eye(3),(*np.shape(q)[:-1],3,3)).copy()
    axes=source()['joints'][name]['axes'][:3];k=0
    for item in axes:
        if item['coordinate'] is None:continue
        axis=np.asarray(item['axis']);axis/=np.linalg.norm(axis)
        R=R@arms.axis_rotation(q[...,k],axis);k+=1
    return arms.OSIM_TO_BODY@R@arms.OSIM_TO_BODY.T


def knee_translation(flexion,side):
    spec=source()['joints']['knee_'+side]['axes'][3:]
    values=[spline(-flexion,a['function']) if a['coordinate'] else np.zeros_like(flexion) for a in spec]
    return np.stack(values,axis=-1)@arms.OSIM_TO_BODY.T


def local_chain(q,lengths,height,side):
    """Hip-relative joint centers and frames; positive knee coordinate is flexion."""
    q=np.asarray(q);H=joint_rotation(q[...,:3],'hip_'+side)
    K=H@joint_rotation(-q[...,3:4],'knee_'+side)
    A=K@joint_rotation(q[...,4:5],'ankle_'+side)
    F=A@joint_rotation(q[...,5:6],'subtalar_'+side)
    neutral=knee_translation(np.asarray(0.),side)
    femur_scale=lengths['thigh']/np.linalg.norm(neutral)
    distal=(H@(neutral*femur_scale)[...,None])[...,0]
    knee=(H@(knee_translation(q[...,3],side)*femur_scale)[...,None])[...,0]
    ankle=knee-lengths['shank']*K[..., :,1]
    offset=np.asarray(source()['joints']['subtalar_'+side]['parent']['translation'])@arms.OSIM_TO_BODY.T*(height/180*100)
    heel_origin=ankle+(A@offset[...,None])[...,0]
    return dict(femur_distal=distal,knee=knee,ankle=ankle,heel_origin=heel_origin,
        thigh_frame=H,shank_frame=K,talus_frame=A,calcaneus_frame=F,foot_frame=F@SHOE_BASIS)


def bounds_and_rates():
    lo,hi,speed,accel=arms.bounds_and_rates()
    return tuple(np.r_[v[:20],np.tile(np.deg2rad(extra),2)] for v,extra in zip(
        (lo,hi,speed,accel),(LEG_LIMITS_DEG[:,0],LEG_LIMITS_DEG[:,1],LEG_SPEED_DEG_S,LEG_ACCEL_DEG_S2)))


def kinematics(q,lengths,height):
    q=np.atleast_2d(q)
    # Upper body is exactly the source-checked shared arm chain.
    base=np.c_[q[:,:20],np.zeros((len(q),8))]
    g=stage1_kinematics(base,lengths,height);R=g['orientations'][:,0]
    j=g['joints'].copy();rigid=np.stack([j[:,:,:2],j[:,:,1:]],axis=2)
    leg_frames=[];feet=[];origins=[];distals=[]
    for i,side in enumerate(('l','r')):
        a=local_chain(q[:,20+6*i:26+6*i],lengths,height,side);root=j[:,i+2,0].copy()
        def world(v):return root+np.einsum('tij,tj->ti',R,v)
        j[:,i+2,1]=world(a['knee']);j[:,i+2,2]=world(a['ankle'])
        distal=world(a['femur_distal']);distals.append(distal)
        rigid[:,i+2,0]=np.stack([root,distal],axis=1)
        rigid[:,i+2,1]=j[:,i+2,1:]
        leg_frames.extend(R@a[key] for key in ('thigh_frame','shank_frame','talus_frame','calcaneus_frame'))
        feet.append(R@a['foot_frame']);origins.append(world(a['heel_origin']))
    # Keep the existing aggregate COM fractions; no extra foot mass is invented.
    m,c=MASS_FRACTIONS,CENTER_OF_MASS_FRACTIONS
    com=m['head']*g['head']+m['torso']*(q[:,:3]+c['torso']*lengths['torso']*R[:,:,1])
    for i in range(4):
        for b,name in enumerate(('upper_arm','forearm') if i<2 else ('thigh','shank')):
            seg=rigid[:,i,b];com+=m[name]*(seg[:,0]+c[name]*(seg[:,1]-seg[:,0]))
    g.update(joints=j,com=com,rigid_segments=rigid,femur_distal=np.stack(distals,axis=1),
        foot_frames=np.stack(feet,axis=1),foot_origins=np.stack(origins,axis=1),
        orientations=np.concatenate([g['orientations'],np.stack(leg_frames,axis=1)],axis=1))
    return g


def initialize(legacy,lengths,height,foot_reference=None,progress=None):
    upper=arms.initialize(legacy,lengths,height,progress);q=np.c_[upper[:,:20],np.zeros((len(upper),12))]
    R=arms.rotation(q[:,3:6]);lo,hi,_,_=bounds_and_rates()
    for i,side in enumerate(('l','r')):
        root=q[:,:3]+(-1 if i==0 else 1)*lengths['hip_w']/2*R[:,:,0]
        obs=np.einsum('tji,tkj->tki',R,legacy['joints'][:,i+2,1:]-root[:,None])
        previous=np.deg2rad([45,-20,0,60,0,0.]);lower,upper=lo[20+6*i:26+6*i],hi[20+6*i:26+6*i]
        for t in range(len(q)):
            def residual(a):
                g=local_chain(a,lengths,height,side)
                pieces=[(g['knee']-obs[t,0])/8,(g['ankle']-obs[t,1])/2,.04*(a-previous)]
                if foot_reference is not None:
                    target=R[t].T@foot_reference[t,i]
                    pieces.append(.5*(g['foot_frame']-target).ravel())
                return np.concatenate(pieces)
            seeds=[previous]
            if t==0:seeds.extend(np.deg2rad([f,-30,r,80,0,0]) for f in [20,80] for r in [-30,30])
            fits=[least_squares(residual,np.clip(a,lower,upper),bounds=(lower,upper),max_nfev=60) for a in seeds]
            best=min(fits,key=lambda x:x.cost);previous=best.x;q[t,20+6*i:26+6*i]=previous
    return q
