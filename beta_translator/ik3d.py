"""Fixed-length spatial IK and conditional 3D quasi-static support experiments.

The wall, camera, body proportions, contact normals and forces are hypotheses.
No learned depth or source keypoint is promoted to a measured 3D position.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix, diags, kron, vstack
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, QhullError
from .inverse_kinematics import make_body, MASS_FRACTIONS, CENTER_OF_MASS_FRACTIONS
from .scene3d import project, upward_force
from .support_refinement import sparse_direct_least_squares, stationarity

POSE_DIM = 22
DIM = 34


def rotation(angles):
    """Batched Rz(roll) Ry(yaw) Rx(pitch), also valid for complex-step derivatives."""
    a,b,c = np.moveaxis(angles, -1, 0)
    ca,sa,cb,sb,cc,sc = np.cos(a),np.sin(a),np.cos(b),np.sin(b),np.cos(c),np.sin(c)
    return np.stack([ca*cb, ca*sb*sc-sa*cc, ca*sb*cc+sa*sc,
                     sa*cb, sa*sb*sc+ca*cc, sa*sb*cc-ca*sc,
                     -sb, cb*sc, cb*cc], axis=-1).reshape(-1,3,3)


def direction(a,b):
    return np.stack([np.sin(a)*np.cos(b), -np.cos(a)*np.cos(b), np.sin(b)], axis=-1)


def limb_basis(delta,helper=None):
    distance=np.sqrt((delta*delta).sum(axis=-1)+1e-20)
    axis=delta/distance[...,None]
    if helper is None:
        helper=np.broadcast_to(np.array([0.,0.,1.]),axis.shape).copy()
        helper[np.abs(axis[...,2].real)>.95]=[1.,0.,0.]
    u=np.cross(axis,helper);u=u/np.sqrt((u*u).sum(axis=-1)+1e-20)[...,None]
    return distance,axis,u,np.cross(axis,u)


def kinematics(poses, lengths, targets=None, known=None,basis_helper=None):
    q = np.atleast_2d(poses)
    rot = rotation(q[:,3:6]);hip=q[:,:3]
    up,right = rot[:,:,1],rot[:,:,0]
    shoulder=hip+lengths['torso']*up
    roots=np.stack([shoulder-lengths['shoulder_w']/2*right, shoulder+lengths['shoulder_w']/2*right,
                    hip-lengths['hip_w']/2*right,hip+lengths['hip_w']/2*right],axis=1)
    angles=q[:,6:22].reshape(-1,4,4)
    l1=np.array([lengths['upper_arm']]*2+[lengths['thigh']]*2)
    l2=np.array([lengths['forearm']]*2+[lengths['shank']]*2)
    upper=direction(angles[:,:,0],angles[:,:,1]);lower=direction(angles[:,:,2],angles[:,:,3])
    middle=roots+l1[None,:,None]*upper;end=middle+l2[None,:,None]*lower
    if targets is not None:
        d,axis,u,v=limb_basis(np.asarray(targets)-roots,basis_helper)
        # A 0.02 cm reach reserve avoids the infinite elbow-circle derivative at
        # perfect extension. Bone lengths remain exact; the 0.1 cm audit is unchanged.
        lo=np.abs(l1-l2)+.001;hi=l1+l2-.02
        reach=np.where(d.real<lo,lo,np.where(d.real>hi,hi,d))
        a=(l1*l1-l2*l2+reach*reach)/(2*reach)
        radius=np.sqrt(positive(l1*l1-a*a)+1e-18)
        swivel=angles[:,:,0]
        bend=np.cos(swivel)[...,None]*u+np.sin(swivel)[...,None]*v
        contact_middle=roots+a[...,None]*axis+radius[...,None]*bend
        contact_end=roots+reach[...,None]*axis
        middle=np.where(np.asarray(known)[...,None],contact_middle,middle)
        end=np.where(np.asarray(known)[...,None],contact_end,end)
    joints=np.stack([roots,middle,end],axis=2)
    f,c=MASS_FRACTIONS,CENTER_OF_MASS_FRACTIONS
    head=shoulder+c['head']*lengths['torso']*up
    com=f['head']*head+f['torso']*(hip+c['torso']*lengths['torso']*up)
    for i in range(4):
        u,l=('upper_arm','forearm') if i<2 else ('thigh','shank')
        com+=f[u]*(roots[:,i]+c[u]*(middle[:,i]-roots[:,i]))+f[l]*(middle[:,i]+c[l]*(end[:,i]-middle[:,i]))
    return joints,com,head


def initial_pose(reference, depth_cm, lengths):
    q=np.zeros((len(reference),POSE_DIM));q[:,:2]=reference[:,:2];q[:,2]=depth_cm;q[:,3]=reference[:,2]
    a=reference[:,3:].reshape(-1,4,2)
    angles=q[:,6:].reshape(-1,4,4);angles[:,:,0]=a[:,:,0];angles[:,:,2]=a.sum(axis=-1)
    l1=np.array([lengths['upper_arm']]*2+[lengths['thigh']]*2)
    l2=np.array([lengths['forearm']]*2+[lengths['shank']]*2)
    angles[:,:,1]=np.arcsin(np.clip(-depth_cm*.35/l1,-.8,.8))
    angles[:,:,3]=np.arcsin(np.clip(-depth_cm*.65/l2,-.8,.8))
    return q


def positive(a):
    # A fixed real-valued active set retains the correct derivative off the kink.
    return np.where(np.real(a)>0,a,0.)


def smooth_positive(a,width):
    return .5*(a+np.sqrt(a*a+width*width))


class SpatialProblem:
    def __init__(self, data, tilt_deg, depth_cm, height_cm=180, contact_weight=20., depth_weight=1., force_weight=25., wall_weight=100., ratios=None,support_weight=300.):
        if not np.isfinite([tilt_deg,depth_cm,height_cm,contact_weight,depth_weight,force_weight,wall_weight,support_weight]).all() or not (-30<=tilt_deg<=60 and 3<=depth_cm<=80):
            raise ValueError('Finite tilt [-30, 60] degrees and hip depth prior [3, 80] cm required.')
        if min(height_cm,contact_weight,depth_weight,wall_weight)<=0 or min(force_weight,support_weight)<0:raise ValueError('Invalid weights or height.')
        if len(data['frames'])<1 or (len(data['frames'])>1 and not np.all(np.diff(data['frames'])==1)):
            raise ValueError('A solve must contain consecutive native frames; isolated keyframes need separate solves.')
        self.data=data;self.n=len(data['frames']);self.lengths=make_body(height_cm,ratios)
        self.height=height_cm;self.depth=depth_cm;self.tilt=tilt_deg
        self.contact_weight=contact_weight;self.depth_weight=depth_weight;self.force_weight=force_weight
        self.wall_weight=wall_weight
        self.support_weight=support_weight
        self.required=upward_force(tilt_deg);self.mu=.8;self.limit=1.
        self.known=np.asarray(data['known'],bool)
        self.targets=np.nan_to_num(data['targets'])
        self.seed=initial_pose(data['reference'],depth_cm,self.lengths)
        seed_joints,_,_=kinematics(self.seed,self.lengths)
        delta=self.targets-seed_joints[:,:,0]
        seed_axis=delta/np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1e-10)
        self.basis_helper=np.broadcast_to(np.array([0.,0.,1.]),delta.shape).copy()
        self.basis_helper[np.abs(seed_axis[...,2])>.95]=[1.,0.,0.]
        _,axis,u,v=limb_basis(delta,self.basis_helper)
        elbow=seed_joints[:,:,1]-seed_joints[:,:,0]
        swivel=np.arctan2((elbow*v).sum(axis=-1),(elbow*u).sum(axis=-1))
        for i in range(4):self.seed[:,6+4*i]=np.where(self.known[:,i],swivel[:,i],self.seed[:,6+4*i])
        self.seed[:,6:]=np.unwrap(self.seed[:,6:],axis=0)
        self.lower=np.full((self.n,DIM),-np.inf);self.upper=np.full((self.n,DIM),np.inf)
        self.lower[:,2]=3.;self.upper[:,2]=90.
        self.lower[:,3:6]=[-1.5,-1.2,-1.2];self.upper[:,3:6]=[1.5,1.2,1.2]
        self.lower[:,7:22:2]=-1.52;self.upper[:,7:22:2]=1.52
        self.lower[:,22:]=-self.limit;self.upper[:,22:]=self.limit
        lf=self.lower[:,22:].reshape(self.n,4,3);uf=self.upper[:,22:].reshape(self.n,4,3)
        lf[:,2:,2]=0.;lf[~self.known]=0.;uf[~self.known]=0.
        if force_weight==0:lf[:]=0.;uf[:]=0.
        self.unload=np.asarray(data.get('unload',np.zeros((self.n,4))))
        self.polygons=[];poly_cache={}
        for t in range(self.n):
            key=(tuple(self.targets[t].ravel()),tuple(self.known[t]))
            if key not in poly_cache:
                poly_cache[key]=(support_polygon(self.targets[t],self.known[t],tilt_deg,height_cm) if support_weight>0
                    else dict(equations=np.empty((0,3)),center=np.zeros(3)))
            self.polygons.append(poly_cache[key])
        self.poly_rows=max([len(p['equations']) for p in self.polygons],default=0)
        self.poly_a=np.zeros((self.n,self.poly_rows,2));self.poly_b=np.zeros((self.n,self.poly_rows));self.poly_center=np.zeros((self.n,3))
        for t,p in enumerate(self.polygons):
            m=len(p['equations']);self.poly_a[t,:m]=p['equations'][:,:2];self.poly_b[t,:m]=p['equations'][:,2]+.01
            self.poly_center[t]=p['center']
        # Scale time derivatives by fps: behavior does not silently change with FPS.
        # Limb angles change meaning when a contact releases (swivel versus free
        # direction). Smooth actual XYZ joints, never those mixed coordinates.
        scales=np.r_[[np.inf]*22,[1.]*12]
        if self.n>1:
            d1=diags([-np.ones(self.n-1),np.ones(self.n-1)],[0,1],shape=(self.n-1,self.n))
            parts=[kron(d1,diags(.035*data['fps']/scales))]
            if self.n>2:
                d2=diags([np.ones(self.n-2),-2*np.ones(self.n-2),np.ones(self.n-2)],[0,1,2],shape=(self.n-2,self.n))
                parts.append(kron(d2,diags(.004*data['fps']**2/scales)))
            self.smooth=vstack(parts,format='csr')
            joint_parts=[kron(d1,diags(np.full(36,.025*data['fps']/40.)))]
            if self.n>2:joint_parts.append(kron(d2,diags(np.full(36,.003*data['fps']**2/40.))))
            self.joint_smooth=vstack(joint_parts,format='csr')
        else:self.smooth=coo_matrix((0,DIM)).tocsr()
        if self.n==1:self.joint_smooth=coo_matrix((0,36)).tocsr()

    def local(self,x):
        j,com,head=kinematics(x[:,:22],self.lengths,self.targets,self.known,self.basis_helper)
        force=x[:,22:].reshape(-1,4,3)*self.known[:,:,None]
        pixels,z=project(j,self.data['K'],self.data['R'],self.data['t'])
        guide=((pixels-self.data['observed_px'])/40.*np.sqrt(self.data['visibility'][...,None])).reshape(self.n,-1)
        contact=(self.contact_weight*(j[:,:,2]-self.targets)*self.known[:,:,None]).reshape(self.n,-1)
        # Friction pyramid INSIDE the circular Coulomb cone for assumed wall-normal feet.
        ft=force[:,2:];normal=ft[:,:,2];cap=self.mu*normal/np.sqrt(2.)
        friction=np.stack([smooth_positive(ft[:,:,0]-cap,1e-4),smooth_positive(-ft[:,:,0]-cap,1e-4),
                           smooth_positive(ft[:,:,1]-cap,1e-4),smooth_positive(-ft[:,:,1]-cap,1e-4)],axis=-1)
        net=force.sum(axis=1)-self.required
        moment=np.cross((self.targets-com[:,None])/self.height,force).sum(axis=1)
        upper=j[:,:,1]-j[:,:,0];lower=j[:,:,2]-j[:,:,1]
        l1=np.array([self.lengths['upper_arm']]*2+[self.lengths['thigh']]*2)
        l2=np.array([self.lengths['forearm']]*2+[self.lengths['shank']]*2)
        cosine=(upper*lower).sum(axis=-1)/(l1*l2)
        joint=positive(np.cos(np.deg2rad(155.))-cosine)*10.
        wall=smooth_positive(-j[:,:,:,2],.01)*self.wall_weight
        behind=positive(10.-z)*.5
        depth=self.depth_weight*(x[:,2]-self.depth)/10.
        # Small pose prior prevents unconstrained branch flips; not measured anatomy.
        pose=.08*(x[:,3:22]-self.seed[:,3:22])
        relative=(com-self.poly_center)/self.height
        support_xy=np.stack([relative[:,0],-self.required[2]*relative[:,1]+self.required[1]*relative[:,2]],axis=1)
        reserve=smooth_positive(np.einsum('tki,ti->tk',self.poly_a,support_xy)+self.poly_b,1e-4)*self.support_weight
        return np.concatenate([guide,contact,wall.reshape(self.n,-1),behind.reshape(self.n,-1),
            joint,depth[:,None],pose,reserve,self.force_weight*net,self.force_weight*moment,
            (self.force_weight*friction).reshape(self.n,-1),.07*force.reshape(self.n,-1),
            (2.*force*self.unload[:,:,None]).reshape(self.n,-1)],axis=1)

    def evaluate(self,flat):
        x=flat.reshape(self.n,DIM);local=self.local(x);nr=local.shape[1]
        deriv=np.empty((self.n,nr,DIM));complex_x=x.astype(complex)
        joints=kinematics(x[:,:22],self.lengths,self.targets,self.known,self.basis_helper)[0].reshape(self.n,36)
        jd=np.zeros((self.n,36,DIM))
        for k in range(DIM):
            complex_x[:,k]+=1e-25j;deriv[:,:,k]=self.local(complex_x).imag/1e-25
            if k<22:jd[:,:,k]=kinematics(complex_x[:,:22],self.lengths,self.targets,self.known,self.basis_helper)[0].imag.reshape(self.n,36)/1e-25
            complex_x[:,k]-=1e-25j
        rows=np.broadcast_to(np.arange(self.n*nr).reshape(self.n,nr,1),deriv.shape).ravel()
        cols=np.broadcast_to(np.arange(self.n)[:,None,None]*DIM+np.arange(DIM),deriv.shape).ravel()
        jac=coo_matrix((deriv.ravel(),(rows,cols)),shape=(self.n*nr,self.n*DIM)).tocsr();jac.eliminate_zeros()
        jr=np.broadcast_to(np.arange(self.n*36).reshape(self.n,36,1),jd.shape).ravel()
        jc=np.broadcast_to(np.arange(self.n)[:,None,None]*DIM+np.arange(DIM),jd.shape).ravel()
        jj=coo_matrix((jd.ravel(),(jr,jc)),shape=(self.n*36,self.n*DIM)).tocsr();jj.eliminate_zeros()
        return np.r_[local.ravel(),self.smooth@flat,self.joint_smooth@joints.ravel()],vstack([jac,self.smooth,self.joint_smooth@jj],format='csr')

    def solve(self, initial=None, max_iterations=600, progress=None):
        start=time.monotonic()
        if initial is None:
            x=np.zeros((self.n,DIM));x[:,:22]=self.seed
            x[:,22:].reshape(self.n,4,3)[:,:,:]=self.required[None,None,:]/np.maximum(1,self.known.sum(axis=1))[:,None,None]
        else:x=np.array(initial,copy=True)
        fit=sparse_direct_least_squares(self.evaluate,x.ravel(),self.lower.ravel(),self.upper.ravel(),
            settings={'max_inner_iterations':int(max_iterations),'projected_gradient_tolerance':1e-5},progress=progress)
        if not fit['success']:
            previous={k:v for k,v in fit.items() if k!='x'}
            x0=fit['x'];r,J=self.evaluate(x0)
            scale=np.tile(np.r_[[40.]*3,[1.]*19,[1.]*12],self.n)
            def cost_gradient(y):
                rr,jj=self.evaluate(y*scale)
                return float(rr@rr/2),np.asarray(jj.T@rr).ravel()*scale
            backup=minimize(cost_gradient,x0/scale,jac=True,method='L-BFGS-B',
                bounds=list(zip(self.lower.ravel()/scale,self.upper.ravel()/scale)),
                options={'maxiter':int(max_iterations),'maxls':40,'maxcor':30,'ftol':1e-14,'gtol':1e-7})
            rr,jj=self.evaluate(backup.x*scale)
            if float(rr@rr/2)<fit['cost']:
                fit['x']=backup.x*scale;fit['cost']=float(rr@rr/2)
                fit['optimality']=stationarity(fit['x'],rr,jj,self.lower.ravel(),self.upper.ravel())[-1]
                fit['success']=bool(fit['optimality']<=1e-5 and backup.success)
                fit['message']='L-BFGS-B fallback; fresh projected gradient audited: '+str(backup.message)
                fit['nfev']+=int(backup.nfev);fit['nit']+=int(backup.nit)
            fit['initial_direct_solver']=previous
            fit['fallback_attempted']=True
        values=fit.pop('x').reshape(self.n,DIM);j,com,head=kinematics(values[:,:22],self.lengths,self.targets,self.known,self.basis_helper)
        fit['seconds']=time.monotonic()-start
        return dict(values=values,joints=j,com=com,head=head,solver=fit,basis_helper=self.basis_helper,candidate_forces_fitted=self.force_weight>0)


def support_polygon(points,active,tilt_deg,height=180.):
    """Conservative sampled inner hull of feasible projected COM, with force reserve.

    LP vertices use 0.65 foot friction and 0.9 BW component limits; independent
    audits use the declared 0.8 / 1.0 assumptions. Missing/degenerate polygons
    are explicitly left unregularized and still audited, never called passes.
    """
    points=np.asarray(points);active=np.asarray(active,bool)
    center=points[active].mean(0) if active.any() else np.zeros(3)
    req=upward_force(tilt_deg);A=np.zeros((6,14));A[:3,:12]=np.tile(np.eye(3),(1,4))
    for i,p in enumerate((points-center)/height):
        a,b,c=p;A[3:,3*i:3*i+3]=[[0,-c,b],[c,0,-a],[-b,a,0]]
    A[4,12]=req[2];A[5,12]=-req[1];A[3,13]=1.
    ub=[]
    for i in (2,3):
        for axis in (0,1):
            for sign in (-1.,1.):
                row=np.zeros(14);row[3*i+axis]=sign;row[3*i+2]=-.65/np.sqrt(2);ub.append(row)
    bounds=[]
    for i in range(4):bounds.extend([(-.9,.9),(-.9,.9),(0 if i>=2 else -.9,.9)] if active[i] else [(0,0)]*3)
    bounds.extend([(None,None)]*2);vertices=[]
    for theta in np.linspace(0,2*np.pi,32,endpoint=False):
        c=np.r_[np.zeros(12),np.cos(theta),np.sin(theta)]
        fit=linprog(c,A_ub=np.asarray(ub),b_ub=np.zeros(len(ub)),A_eq=A,b_eq=np.r_[req,np.zeros(3)],bounds=bounds,method='highs')
        if fit.success:vertices.append(fit.x[-2:])
    vertices=np.unique(np.round(vertices,12),axis=0) if vertices else np.empty((0,2))
    equations=np.empty((0,3))
    if len(vertices)>=3:
        try:equations=ConvexHull(vertices).equations
        except QhullError:pass
    return dict(equations=equations,vertices=vertices,center=center)


def force_feasible(points, com, active, tilt_deg, mu=.8, limit=1.,foot_normal='wall'):
    """Independent 6D wrench LP. Never use forces assigned to a missed contact."""
    active=np.asarray(active,bool)
    if not active.any():return False
    A=np.zeros((6,12));A[:3]=np.tile(np.eye(3),(1,4))
    for i,p in enumerate(np.asarray(points)-com):
        a,b,c=p/180.
        A[3:,3*i:3*i+3]=[[0,-c,b],[c,0,-a],[-b,a,0]]
    if foot_normal not in ('wall','world_up'):raise ValueError('Unknown hypothesized foot normal.')
    normal=np.array([0.,0.,1.]) if foot_normal=='wall' else upward_force(tilt_deg)
    tangents=[np.array([1.,0.,0.]),np.cross(normal,[1.,0.,0.])]
    ub=[]
    for i in (2,3):
        row=np.zeros(12);row[3*i:3*i+3]=-normal;ub.append(row)
        for tangent in tangents:
            for sign in (-1.,1.):
                row=np.zeros(12);row[3*i:3*i+3]=sign*tangent-mu*normal/np.sqrt(2);ub.append(row)
    bounds=[]
    for i in range(4):
        bounds.extend([(-limit,limit)]*3 if active[i] else [(0,0)]*3)
    result=linprog(np.zeros(12),A_ub=np.asarray(ub),b_ub=np.zeros(len(ub)),A_eq=A,
                   b_eq=np.r_[upward_force(tilt_deg),np.zeros(3)],bounds=bounds,method='highs')
    return bool(result.success)


def audit(result, data, tilt_deg, height_cm=180,ratios=None):
    j,com=result['joints'],result['com'];rows=[];lengths=make_body(height_cm,ratios)
    pixels,cam_z=project(j,data['K'],data['R'],data['t'])
    targets=np.nan_to_num(data['targets']);known=np.asarray(data['known'])
    gap=np.linalg.norm(j[:,:,2]-targets,axis=-1)
    segment=np.linalg.norm(np.diff(j,axis=2),axis=-1)
    expected=np.array([[lengths['upper_arm'],lengths['forearm']]]*2+[[lengths['thigh'],lengths['shank']]]*2)
    for t,f in enumerate(data['frames']):
        matched=known[t]&(gap[t]<=.1)
        eligible=matched&data['supported'][t]&~data['low_confidence'][t]
        conditional=force_feasible(targets[t],com[t],matched,tilt_deg)
        alternative=force_feasible(targets[t],com[t],matched,tilt_deg,foot_normal='world_up')
        confirmed=force_feasible(targets[t],com[t],eligible,tilt_deg)
        robust=conditional and all(force_feasible(targets[t],com[t]+offset,matched,tilt_deg)
            for offset in ([2,0,0],[-2,0,0],[0,0,2],[0,0,-2]))
        departing=np.asarray(data['unload'][t])>0
        unloaded=force_feasible(targets[t],com[t],matched&~departing,tilt_deg) if departing.any() else None
        upper=j[t,:,1]-j[t,:,0];lower=j[t,:,2]-j[t,:,1]
        cos=(upper*lower).sum(axis=1)/np.prod(expected,axis=1)
        max_flex=float(np.rad2deg(np.arccos(np.clip(cos,-1,1))).max())
        force=result['values'][t,22:].reshape(4,3)*known[t,:,None]
        net=force.sum(axis=0)-upward_force(tilt_deg)
        moment=np.cross((targets[t]-com[t])/height_cm,force).sum(axis=0)
        unresolved=bool((known[t]&~matched).any() or j[t,:,:,2].min()<-.1 or not result['solver']['success'] or not conditional)
        rows.append(dict(frame=int(f),max_contact_gap_cm=float(gap[t,known[t]].max()) if known[t].any() else 0.,
            missed_contacts=int((known[t]&~matched).sum()),max_segment_error_cm=float(np.abs(segment[t]-expected).max()),
            source_guide_rmse_px=float(np.sqrt(np.mean((pixels[t]-data['observed_px'][t])**2))),
            min_wall_clearance_cm=float(j[t,:,:,2].min()),hip_depth_cm=float(result['values'][t,2]),
            max_joint_flex_deg=max_flex,conditional_balance=conditional,eligible_balance=confirmed,
            conditional_balance_upward_footholds=alternative,contact_normals_measured=False,
            conditional_robust_2cm=robust,preunload_without_departing=unloaded,
            all_geometry_uncertain=True,tilt_measured=False,depth_measured=False,
            extrapolated_contacts=int((known[t]&~data['supported'][t]).sum()),
            low_confidence_contacts=int((known[t]&data['low_confidence'][t]).sum()),
            optimizer_converged=result['solver']['success'],minimum_camera_depth_cm=float(cam_z[t].min()),
            status='constraint_warning' if unresolved else 'assumption_uncertain',
            candidate_forces_fitted=result.get('candidate_forces_fitted',True),
            candidate_net_force_error_bw=float(np.linalg.norm(net)) if result.get('candidate_forces_fitted',True) else None,
            candidate_moment_error_bw_height=float(np.linalg.norm(moment)) if result.get('candidate_forces_fitted',True) else None,
            force_on_missed_contact_bw=float(np.linalg.norm(force[known[t]&~matched],axis=1).max()) if (known[t]&~matched).any() else 0.,
            preunload_timing_uncertain=bool(departing.any()),
            camera_tracking_fit_rmse_px=float(data.get('camera_residual_px',np.zeros(len(j)))[t])))
    return rows,gap
