"""Parameter-dependent specification checks, independent of a trial or optimizer.

SI units unless a function name says otherwise. These are audit functions:
supplying matrices or a capacity table does not create a calibrated human model.
Missing inputs return unknown. There are no guessed strength/fatigue defaults.
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


def result(name, status, reason='', **values):
    if status not in ('pass','fail','unknown','not_applicable'):raise ValueError('Invalid status')
    return dict(id=name,status=status,reason=reason,**values)


def unknown(name, reason):return result(name,'unknown',reason)


def bounded(name, values=None, lower=None, upper=None, tolerance=None):
    if any(x is None for x in [values,lower,upper,tolerance]):
        return unknown(name,'Values, directional bounds and a declared tolerance are required')
    x,lo,hi=np.broadcast_arrays(np.asarray(values,float),lower,upper)
    if not np.isfinite(tolerance) or tolerance<0:raise ValueError('Invalid tolerance')
    if np.any(lo>hi):raise ValueError('Lower bound exceeds upper bound')
    finite=np.isfinite(x)&np.isfinite(lo)&np.isfinite(hi)
    excess=np.maximum(lo-x,x-hi);failed=finite&(excess>tolerance)
    return result(name,'fail' if failed.any() else ('unknown' if not finite.all() else 'pass'),
        'Sampled numeric bounds only',failed_samples=int(failed.sum()),unknown_samples=int((~finite).sum()),
        max_excess=float(max(0.,np.max(excess[finite]))) if finite.any() else None)


def shoulder_envelope(coordinates=None, posture_samples=None, lower=None, upper=None, tolerance=None):
    """Calibrated (plane,elevation) domain; no extrapolation into unseen poses."""
    if any(x is None for x in [coordinates,posture_samples,lower,upper,tolerance]):
        return unknown('shoulder_envelope','Calibrated posture-dependent axial envelope is missing')
    q=np.asarray(coordinates,float);samples=np.asarray(posture_samples,float)
    if q.shape[-1]!=3 or samples.ndim!=2 or samples.shape[1]!=2:raise ValueError('Expected p,e,r coordinates and p,e samples')
    if not np.isfinite(samples).all():raise ValueError('Non-finite calibration domain')
    lo=LinearNDInterpolator(samples,np.asarray(lower,float))(q[...,:2])
    hi=LinearNDInterpolator(samples,np.asarray(upper,float))(q[...,:2])
    return bounded('shoulder_envelope',q[...,2],lo,hi,tolerance)


def rotation_rates(matrices=None, times=None, speed_rad_s=None, acceleration_rad_s2=None):
    if any(x is None for x in [matrices,times,speed_rad_s,acceleration_rad_s2]):
        return unknown('rotation_rates','Rotations, timestamps and justified caps are required')
    R=np.asarray(matrices,float);t=np.asarray(times,float)
    if R.ndim==3:R=R[:,None]
    if not np.isfinite(t).all() or len(t)<3 or len(R)!=len(t) or np.any(np.diff(t)<=0):raise ValueError('At least three increasing timestamps required')
    if not np.allclose(R.swapaxes(-1,-2)@R,np.eye(3),atol=1e-8) or not np.allclose(np.linalg.det(R),1,atol=1e-8):
        raise ValueError('Proper rotations required')
    if not np.isfinite([speed_rad_s,acceleration_rad_s2]).all() or min(speed_rad_s,acceleration_rad_s2)<0:raise ValueError('Finite nonnegative rate caps required')
    relative=R[:-1].swapaxes(-1,-2)@R[1:]
    local=Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(len(t)-1,-1,3)/np.diff(t)[:,None,None]
    world=np.einsum('tsij,tsj->tsi',R[:-1],local)
    acc=np.diff(world,axis=0)/np.diff((t[1:]+t[:-1])/2)[:,None,None]
    speed=np.linalg.norm(world,axis=-1);accel=np.linalg.norm(acc,axis=-1)
    passed=bool(np.all(speed<=speed_rad_s) and np.all(accel<=acceleration_rad_s2))
    return result('rotation_rates','pass' if passed else 'fail','Observable intervals only; no pre-video derivative or subframe certificate',
        max_speed_rad_s=float(speed.max()),max_acceleration_rad_s2=float(accel.max()),
        speed_intervals=len(speed),acceleration_intervals=len(accel))


def segment_distance(a,b,c,d):
    """Closest distance between finite 3D segments, including degenerate points."""
    a,b,c,d=map(lambda x:np.asarray(x,float),(a,b,c,d))
    u=b-a;v=d-c;w=a-c;aa=u@u;bb=u@v;cc=v@v;dd=u@w;ee=v@w
    if aa<1e-15 and cc<1e-15:return float(np.linalg.norm(w))
    if aa<1e-15:s=0.;t=np.clip(ee/cc,0,1)
    elif cc<1e-15:t=0.;s=np.clip(-dd/aa,0,1)
    else:
        denom=aa*cc-bb*bb
        s=np.clip((bb*ee-cc*dd)/denom,0,1) if denom>1e-15 else 0.
        t=(bb*s+ee)/cc
        if t<0:t=0.;s=np.clip(-dd/aa,0,1)
        elif t>1:t=1.;s=np.clip((bb-dd)/aa,0,1)
    return float(np.linalg.norm(w+s*u-t*v))


def self_collision(segments_m=None, radii_m=None, pairs=None, clearance_m=None):
    if any(x is None for x in [segments_m,radii_m,pairs,clearance_m]):
        return unknown('self_collision','Body dimensions, explicit non-exempt pairs and clearance are missing')
    segments=np.asarray(segments_m,float);r=np.asarray(radii_m,float)
    if segments.ndim!=4 or segments.shape[-2:]!=(2,3) or len(r)!=segments.shape[1]:raise ValueError('Expected [time,segment,endpoint,xyz]')
    if not np.isfinite(segments).all() or not np.isfinite(r).all() or np.any(r<=0) or not np.isfinite(clearance_m) or clearance_m<0:raise ValueError('Invalid capsule profile')
    if not pairs:return unknown('self_collision','No collision pairs checked')
    margins=[]
    for i,j in pairs:
        if i==j or min(i,j)<0 or max(i,j)>=len(r):raise ValueError('Invalid collision pair')
        margins.extend(segment_distance(*s[i],*s[j])-r[i]-r[j]-clearance_m for s in segments)
    return result('self_collision','fail' if min(margins)<0 else 'pass','Sampled capsules only; swept collision remains unknown',
        minimum_margin_m=float(min(margins)),pair_count=len(pairs))


def torque_capacity(required_nm=None, lower_nm=None, upper_nm=None, tolerance_nm=None):
    return bounded('torque_capacity',required_nm,lower_nm,upper_nm,tolerance_nm)


def fatigue(times=None, recruitment_per_s=None, fatigue_rate=None, recovery_rate=None, initial=None):
    """3CC state equations with supplied piecewise-constant recruitment.

    Exact linear interval integration preserves the state-sum equation; invalid
    recruitment may leave the nonnegative simplex and is reported as failure,
    never repaired by clipping. A capacity/recruitment controller is still needed
    before this becomes a coupled muscle-strength model.
    """
    if any(x is None for x in [times,recruitment_per_s,fatigue_rate,recovery_rate,initial]):
        return unknown('fatigue','Initial state, recruitment and calibrated fatigue/recovery rates are missing')
    t=np.asarray(times,float);C=np.asarray(recruitment_per_s,float);state=np.asarray(initial,float)
    if not np.isfinite(t).all() or t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0) or C.shape!=(len(t)-1,):raise ValueError('One recruitment rate per time interval required')
    if state.shape!=(3,) or not np.isfinite(state).all() or np.any(state<0) or not np.isclose(state.sum(),1):raise ValueError('Initial active/resting/fatigued fractions must sum to one')
    if fatigue_rate<0 or recovery_rate<0 or not np.isfinite([fatigue_rate,recovery_rate,*C]).all():raise ValueError('Invalid rates')
    states=[state.copy()]
    for dt,c in zip(np.diff(t),C):
        A=np.array([[-fatigue_rate,0,0,c],[0,0,recovery_rate,-c],
                    [fatigue_rate,0,-recovery_rate,0],[0,0,0,0]],float)
        state=(expm(A*dt)@np.r_[state,1.])[:3];states.append(state.copy())
    states=np.asarray(states);valid=bool(np.all(states>=-1e-10) and np.allclose(states.sum(1),1,atol=1e-10))
    return result('fatigue','pass' if valid else 'fail','State-equation check only; not validated endurance or capacity coupling',
        states=states.tolist(),capacity_coupling_validated=False)


def centroidal_dynamics(mass_kg=None, com_m=None, com_acceleration_m_s2=None,
                        angular_momentum_rate_nm=None, points_m=None, forces_n=None,
                        moments_nm=None, force_tolerance_n=None, moment_tolerance_nm=None):
    args=[mass_kg,com_m,com_acceleration_m_s2,angular_momentum_rate_nm,points_m,forces_n,
          moments_nm,force_tolerance_n,moment_tolerance_nm]
    if any(x is None for x in args):return unknown('centroidal_dynamics','Mass, momentum, external wrenches and declared tolerances are missing')
    if not np.isfinite(mass_kg) or mass_kg<=0:raise ValueError('Positive mass required')
    if not np.isfinite([force_tolerance_n,moment_tolerance_nm]).all() or min(force_tolerance_n,moment_tolerance_nm)<0:raise ValueError('Finite nonnegative tolerances required')
    com=np.asarray(com_m);force=np.asarray(forces_n);points=np.asarray(points_m)
    linear=mass_kg*np.asarray(com_acceleration_m_s2)-mass_kg*np.array([0.,-9.81,0.])-force.sum(-2)
    angular=np.asarray(angular_momentum_rate_nm)-(np.cross(points-com[...,None,:],force)+moments_nm).sum(-2)
    lr=np.linalg.norm(linear,axis=-1);ar=np.linalg.norm(angular,axis=-1)
    if not np.isfinite(lr).all() or not np.isfinite(ar).all():return unknown('centroidal_dynamics','Non-finite dynamics input')
    return result('centroidal_dynamics','pass' if np.all(lr<=force_tolerance_n) and np.all(ar<=moment_tolerance_nm) else 'fail',
        'All external wrenches must be included; joint dynamics and forward replay remain separate',
        max_force_residual_n=float(np.max(lr)),max_moment_residual_nm=float(np.max(ar)))


def joint_dynamics(mass_matrix=None, bias=None, acceleration=None, actuation_map=None,
                   torque=None, contact_jacobian=None, contact_wrench=None, passive=None, tolerance=None):
    args=[mass_matrix,bias,acceleration,actuation_map,torque,contact_jacobian,contact_wrench,passive,tolerance]
    if any(x is None for x in args):return unknown('joint_dynamics','Consistent floating-base dynamics operators and parameters are missing')
    M,h,vdot,S,tau,J,f,p=map(np.asarray,args[:-1])
    if not np.allclose(M,M.swapaxes(-1,-2),atol=1e-9) or np.any(np.linalg.eigvalsh(M)<=0):raise ValueError('Positive definite symmetric inertia required')
    if M.shape[-1]<6 or np.any(abs(S[...,:6])>1e-12):raise ValueError('Floating root must be unactuated')
    residual=(M@vdot[...,None])[...,0]+h-(S.swapaxes(-1,-2)@tau[...,None])[...,0]-(J.swapaxes(-1,-2)@f[...,None])[...,0]-p
    tol=np.asarray(tolerance)
    if np.any(tol<0) or not np.isfinite(tol).all():raise ValueError('Declared generalized-force tolerances required')
    if not np.isfinite(residual).all():return unknown('joint_dynamics','Non-finite dynamics input')
    return result('joint_dynamics','pass' if np.all(abs(residual)<=tol) else 'fail',
        'Operator residual only; not a generated or calibrated musculoskeletal model',
        max_absolute_residual=float(np.max(abs(residual))))
