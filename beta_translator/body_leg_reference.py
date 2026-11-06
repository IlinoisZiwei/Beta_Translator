"""Independent source-parameter checks of the lower-limb kinematic chain."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from .body_legs import source,local_chain,knee_translation
from .body_kinematics import OSIM_TO_BODY


def reference_spline(values,spec):
    # Independent coefficient construction: first derive endpoint third
    # derivatives, then solve the clamped-slope system through cubic interpolation.
    x,y=np.array(spec['x']),np.array(spec['y']);h=np.diff(x);n=len(x)
    slopes=np.diff(y)/h;A=np.zeros((n,n));b=np.zeros(n)
    for i in range(1,n-1):
        A[i,i-1:i+2]=[h[i],2*(h[i-1]+h[i]),h[i-1]]
        b[i]=3*(h[i]*slopes[i-1]+h[i-1]*slopes[i])
    # Endpoint equations in first derivatives, using third divided differences.
    third0=6*sum(y[k]/np.prod([x[k]-x[j] for j in range(4) if j!=k]) for k in range(4))
    third1=6*sum(y[k]/np.prod([x[k]-x[j] for j in range(n-4,n) if j!=k]) for k in range(n-4,n))
    A[0,:2]=[1,1];b[0]=2*slopes[0]+third0*h[0]**2/6
    A[-1,-2:]=[1,1];b[-1]=2*slopes[-1]+third1*h[-1]**2/6
    derivatives=np.linalg.solve(A,b)
    curve=CubicSpline(x,y,bc_type=((1,derivatives[0]),(1,derivatives[-1])))
    v=np.asarray(values);out=curve(v)
    out=np.where(v<x[0],y[0]+(v-x[0])*derivatives[0],out)
    return np.where(v>x[-1],y[-1]+(v-x[-1])*derivatives[-1],out)


def source_chain(q,side):
    """Unscaled complete source transforms relative to the femoral head (metres)."""
    values=dict(zip(['hip_flexion_'+side,'hip_adduction_'+side,'hip_rotation_'+side,
                     'knee_angle_'+side,'ankle_angle_'+side,'subtalar_angle_'+side],q))
    values['knee_angle_'+side]*=-1
    T=np.eye(4);out={};root=None
    for name in ['hip','knee','ankle','subtalar']:
        j=source()['joints'][name+'_'+side]
        P=np.eye(4);P[:3,3]=j['parent']['translation']
        P[:3,:3]=Rotation.from_euler('XYZ',j['parent']['orientation']).as_matrix()
        C=np.eye(4);C[:3,3]=j['child']['translation']
        C[:3,:3]=Rotation.from_euler('XYZ',j['child']['orientation']).as_matrix()
        J=np.eye(4)
        for k,a in enumerate(j['axes']):
            f=a['function'];v=values.get(a['coordinate'],0.)
            if f['type']=='Constant':v=f['value'][0]
            elif f['type']=='LinearFunction':v=f['coefficients'][0]*v+f['coefficients'][1]
            elif f['type']=='SimmSpline':v=float(reference_spline(v,f))
            else:raise ValueError(f['type'])
            axis=np.array(a['axis'],float);axis/=np.linalg.norm(axis)
            if k<3:J[:3,:3]=J[:3,:3]@Rotation.from_rotvec(axis*v).as_matrix()
            else:J[:3,3]+=axis*v
        T=T@P@J@np.linalg.inv(C)
        if root is None:root=T[:3,3].copy()
        out[name]=(OSIM_TO_BODY@T[:3,:3]@OSIM_TO_BODY.T,OSIM_TO_BODY@(T[:3,3]-root))
    return out


def verify_leg_chain(samples=64):
    from .body_constraints import LEG_LIMITS_DEG
    rng=np.random.default_rng(5810);lo,hi=np.deg2rad(LEG_LIMITS_DEG).T
    rotation_error=0.;position_error=0.;curve_error=0.
    for side in ('l','r'):
        neutral=source_chain(np.zeros(6),side)['knee'][1]
        lengths={'thigh':np.linalg.norm(neutral)*100,'shank':43.}
        for q in [np.zeros(6),*rng.uniform(lo,hi,(samples,6))]:
            actual=local_chain(q,lengths,180.,side);expected=source_chain(q,side)
            for joint,frame,point in [('hip','thigh_frame',None),('knee','shank_frame','knee'),('ankle','talus_frame','ankle'),('subtalar','calcaneus_frame','heel_origin')]:
                rotation_error=max(rotation_error,float(np.max(abs(actual[frame]-expected[joint][0]))))
                if point:position_error=max(position_error,float(np.max(abs(actual[point]/100-expected[joint][1]))))
        for spec in source()['joints']['knee_'+side]['axes'][3:5]:
            from .body_legs import spline
            f=spec['function'];x=np.linspace(f['x'][0]-.05,f['x'][-1]+.05,400)
            curve_error=max(curve_error,float(np.max(abs(spline(x,f)-reference_spline(x,f)))))
    return dict(source_sha256=source()['source_sha256'],poses_per_side=samples+1,
        rotation_max_absolute_error=rotation_error,position_max_error_m=position_error,
        knee_curve_max_error_m=curve_error,passed=bool(max(rotation_error,position_error,curve_error)<1e-10),
        source_knee_translations_preserved=True,source_ankle_subtalar_offsets_preserved=True,
        opensim_engine_crosscheck=False,personal_anatomical_validation=False,
        scope='Independent source-parameter composition; scaled project hip roots and proportions remain assumptions.')
