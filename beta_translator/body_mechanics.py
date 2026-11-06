"""Reduced rigid-segment dynamics, collision and actuator screening (SI units).

These are active numerical models under a user profile, not calibrated muscles.
Net joint moments include passive-axis reactions; they are not generalized
MoBL/OpenSim actuation torques. Floating-body balance is checked separately.
"""
import numpy as np

SEGMENTS=('torso','head','left_upper_arm','left_forearm','left_hand',
          'right_upper_arm','right_forearm','right_hand','left_thigh','left_shank',
          'left_foot','right_thigh','right_shank','right_foot')
JOINTS=tuple(f'{side}_{j}' for side in ('left','right') for j in ('shoulder','elbow','wrist'))+tuple(
    f'{side}_{j}' for side in ('left','right') for j in ('hip','knee','ankle'))
# Adjacent capsules overlap by construction; all other pairs are tested.
EXEMPT={(0,1),(0,2),(0,5),(0,8),(0,11),(2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(11,12),(12,13)}
PAIRS=tuple((i,j) for i in range(14) for j in range(i+1,14) if (i,j) not in EXEMPT)


def positive(x): return np.where(np.real(x)>0,x,0)
def clip(x,lo,hi): return np.where(np.real(x)<lo,lo,np.where(np.real(x)>hi,hi,x))
def norm(x,axis=-1): return np.sqrt(np.sum(x*x,axis=axis)+1e-24)


def segments(g,profile):
    """Same geometry feeds COM, inertia, collisions, contact and rendering."""
    j=g['joints'];hip=j[:,2:,0].mean(1);shoulder=j[:,:2,0].mean(1)
    R=g['orientations'];d=profile['dimensions_cm']
    lines=[np.stack([hip,shoulder],1),np.stack([g['head'],g['head']],1)]
    frames=[R[:,0],R[:,0]]
    for side in range(2):
        lines.extend([g['rigid_segments'][:,side,0],g['rigid_segments'][:,side,1],
                      np.stack([j[:,side,2],g['hand_tips'][:,side]],1)])
        frames.extend([R[:,1+side*3],R[:,2+side*3],R[:,3+side*3]])
    for side in range(2):
        foot=g['foot_origins'][:,side];F=g['foot_frames'][:,side]
        lines.extend([g['rigid_segments'][:,side+2,0],g['rigid_segments'][:,side+2,1],
                      np.stack([foot,foot+d['foot_length']*F[:,:,0]],1)])
        frames.extend([R[:,7+side*4],R[:,8+side*4],F])
    return np.stack(lines,1)/100,np.stack(frames,1)


def segment_distance(s,t):
    """Vectorized closest capsule-axis distance; real branches retain CS derivatives."""
    a,b=s[:,0],s[:,1];c,d=t[:,0],t[:,1];u=b-a;v=d-c;w=a-c
    dot=lambda x,y:np.sum(x*y,axis=-1)
    aa,bb,cc,dd,ee=dot(u,u),dot(u,v),dot(v,v),dot(u,w),dot(v,w)
    den=aa*cc-bb*bb
    safe=lambda x:np.where(abs(np.real(x))<1e-16,1.,x)
    x=clip((bb*ee-cc*dd)/safe(den),0,1);x=np.where(abs(np.real(den))<1e-16,0,x)
    y=(bb*x+ee)/safe(cc)
    x=np.where(np.real(y)<0,clip(-dd/safe(aa),0,1),x)
    x=np.where(np.real(y)>1,clip((bb-dd)/safe(aa),0,1),x)
    y=clip(y,0,1)
    x=np.where(np.real(cc)<1e-16,clip(-dd/safe(aa),0,1),x)
    y=np.where(np.real(aa)<1e-16,clip(ee/safe(cc),0,1),y)
    return norm(w+x[:,None]*u-y[:,None]*v)


def shoulder_limits(q,config):
    p,e=q[...,0]*180/np.pi,q[...,1]*180/np.pi
    x=np.array(config['plane_deg']);y=np.array(config['elevation_deg'])
    i=np.clip(np.searchsorted(x,np.real(p))-1,0,len(x)-2)
    k=np.clip(np.searchsorted(y,np.real(e))-1,0,len(y)-2)
    u=(p-x[i])/(x[i+1]-x[i]);v=(e-y[k])/(y[k+1]-y[k])
    def interp(name):
        z=np.array(config[name]);return ((1-u)*(1-v)*z[i,k]+u*(1-v)*z[i+1,k]+(1-u)*v*z[i,k+1]+u*v*z[i+1,k+1])*np.pi/180
    return interp('lower_deg'),interp('upper_deg')


def mechanics(g,q,forces_bw,contacts_cm,profile,fps):
    b=profile['biomechanics'];s,R=segments(g,profile);n=len(s)
    # Apply the same native-frame finite differences directly. Building and
    # squaring an n-by-n matrix inside every Jacobian probe scales cubically.
    deriv=lambda x:np.gradient(x,1/fps,axis=0,edge_order=2)
    second=lambda x:deriv(deriv(x))
    mass=np.array(b['mass_fractions'])*profile['mass_kg']
    centers=s[:,:,0]+np.array(b['com_fractions'])[None,:,None]*(s[:,:,1]-s[:,:,0])
    radii=np.array(b['capsule_radii_cm'])/100
    length2=np.sum((s[:,:,1]-s[:,:,0])**2,axis=-1)
    # Cylinder long axis follows frame Y except feet (frame X); head is a sphere.
    axial=mass[None]*radii[None]**2/2
    transverse=mass[None]*(3*radii[None]**2+length2)/12
    diag=np.stack([transverse,np.broadcast_to(axial,transverse.shape),transverse],-1)
    diag[:,[10,13]]=diag[:,[10,13]][:,:,[1,0,2]]
    diag[:,1]=.4*mass[1]*radii[1]**2
    inertia=(R*diag[:,:,None,:])@R.swapaxes(-1,-2)
    spin=deriv(R)@R.swapaxes(-1,-2)
    omega=np.stack([spin[:,:,2,1]-spin[:,:,1,2],spin[:,:,0,2]-spin[:,:,2,0],spin[:,:,1,0]-spin[:,:,0,1]],-1)/2
    alpha=deriv(omega);angular_momentum=(inertia@omega[...,None])[...,0]
    inertial_moment=(inertia@alpha[...,None])[...,0]+np.cross(omega,angular_momentum)
    demand=mass[None,:,None]*(second(centers)-[0,-9.81,0])
    bw=profile['mass_kg']*9.81;forces=forces_bw*bw;contacts=contacts_cm/100
    com=np.sum(mass[None,:,None]*centers,axis=1)/profile['mass_kg']
    linear=(demand.sum(1)-forces.sum(1))/bw
    angular=(np.sum(inertial_moment+np.cross(centers-com[:,None],demand),axis=1)-np.cross(contacts-com[:,None],forces).sum(1))/bw
    moments=[];speeds=[]
    j=g['joints']/100
    for limb,start in enumerate((2,5,8,11)):
        for joint in range(3):
            origin=j[:,limb,joint]
            moment=(inertial_moment[:,start+joint:start+3]+np.cross(centers[:,start+joint:start+3]-origin[:,None],demand[:,start+joint:start+3])).sum(1)
            moment-=np.cross(contacts[:,limb]-origin,forces[:,limb])
            parent=R[:,0] if joint==0 else R[:,start+joint-1]
            moments.append(np.einsum('tji,tj->ti',parent,moment))
            parent_w=omega[:,0] if joint==0 else omega[:,start+joint-1]
            speeds.append(norm(omega[:,start+joint]-parent_w))
    moments=np.stack(moments,1);speeds=np.stack(speeds,1)
    caps=np.where(np.real(moments)>=0,np.array(b['torque_positive_nm']),np.array(b['torque_negative_nm']))
    caps=caps/(1+speeds[...,None]/b['velocity_scale_rad_s'])
    utilization=norm(moments/caps)
    collision=np.stack([segment_distance(s[:,i],s[:,k])-radii[i]-radii[k]-b['collision_clearance_cm']/100 for i,k in PAIRS],1)
    shoulder=np.stack([q[:,6:9],q[:,13:16]],1);lo,hi=shoulder_limits(shoulder,b['shoulder_envelope'])
    excess=positive(lo-shoulder[:,:,2])+positive(shoulder[:,:,2]-hi)
    residual=np.c_[40*excess,200*positive(-collision),
                   30*linear/b['force_tolerance_bw'],30*angular/b['moment_tolerance_bw_m']]
    return dict(residual=residual,utilization=utilization,shoulder_excess_rad=excess,
                collision_margin_m=collision,force_residual_bw=linear,moment_residual_bw_m=angular,
                joint_moments_nm=moments,capacity_nm=caps,segment_centers_m=centers,
                com_m=com,com_acceleration_m_s2=second(com),segment_inertia_kg_m2=inertia)


def fatigue_coupling(demand,config,fps,with_jacobian=False):
    """Joint-actuator 3CC proxy: positive implicit transitions and causal derivatives.

    Recruitment targets the normalized demanded moment. Fatigued fraction reduces
    capacity; implicit Euler preserves the simplex without clipping states.
    A short clip starts at the explicitly supplied state, never inferred history.
    """
    n,k=demand.shape;state=np.broadcast_to(config['initial_fatigue_state'],(k,3)).copy()
    states=[state.copy()];sens=np.zeros((k,3,n));derivatives=[sens.copy()]
    F=config['fatigue_rate_s'];R=config['recovery_rate_s'];ka=config['activation_rate_s'];kd=config['deactivation_rate_s'];dt=1/fps
    Gprime=np.array([[kd,ka,0],[-kd,-ka,0],[0,0,0.]])
    for t in range(n-1):
        target=np.clip(demand[t],0,1);u=ka*target;v=kd*(1-target)
        G=np.zeros((k,3,3));G[:,0,0]=-v-F;G[:,0,1]=u;G[:,1,0]=v;G[:,1,1]=-u;G[:,1,2]=R;G[:,2,0]=F;G[:,2,2]=-R
        B=np.eye(3)-dt*G;state=np.linalg.solve(B,state[...,None])[...,0]
        if with_jacobian:
            sens=np.linalg.solve(B,sens)
            sens[:,:,t]+=np.linalg.solve(B,dt*np.einsum('ij,kj->ki',Gprime,state)[...,None])[...,0]*((demand[t]>0)&(demand[t]<1))[:,None]
            derivatives.append(sens.copy())
        states.append(state.copy())
    states=np.stack(states);available=1-states[:,:,2]
    effective=demand/available
    over=np.maximum(effective-1,0);fatigue_excess=np.maximum(states[:,:,2]-config['max_fatigued_fraction'],0)
    residual=np.stack([40*over,40*fatigue_excess,.1*states[:,:,2]],-1)
    jac=None
    if with_jacobian:
        from scipy.sparse import coo_matrix
        df=np.stack(derivatives)[:,:,2,:]
        de=df*demand[:,:,None]/available[:,:,None]**2
        de+=np.eye(n)[:,None,:]/available[:,:,None]
        value=np.stack([40*de*(effective>1)[...,None],40*df*(states[:,:,2]>config['max_fatigued_fraction'])[...,None],.1*df],2)
        # Preserve every nonzero causal derivative without materializing COO
        # indices for the many exact zeros introduced by recruitment clipping.
        nz=np.flatnonzero(value)
        rows=nz//n;cols=(nz%n)*k+(rows//3)%k
        jac=coo_matrix((value.ravel()[nz],(rows,cols)),shape=(n*k*3,n*k)).tocsr()
    return dict(states=states,effective_utilization=effective,residual=residual,jacobian=jac)
