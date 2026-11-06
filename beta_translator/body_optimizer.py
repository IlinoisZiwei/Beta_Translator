"""Jointly fit native poses from any trial with bounded articulated arms and rate penalties.

ROM, force-component, coordinate-rate and acceleration bounds are hard. Support
terms are soft during fitting and independently gated before declaring feasibility.
No Cartesian post-filtering, endpoint snapping, or per-clip winner stitching.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,diags,eye,kron,vstack
from . import body_kinematics as arms
from .inverse_kinematics import make_body
from .ik3d import positive
from .body_support import audit_plan
from .body_contacts import ContactMetadata
from .body_constraints import DEFAULT_LIMITS
from .restored_sequence_solver import solve as constrained_solve


class AnatomicalProblem:
    def __init__(self,data,scene,regions,legacy,initial_q,height_cm=180.,ratios=None,smoothness=1.):
        self.data=data;self.scene=scene;self.regions=regions;self.n=len(data['frames'])
        self.height=height_cm;self.lengths=make_body(height_cm,ratios);self.smoothness=float(smoothness)
        self.limits=DEFAULT_LIMITS
        self.qdim=initial_q.shape[1];self.statedim=self.qdim+12;self.localdim=self.statedim+8
        if self.n<3 or not np.all(np.diff(data['frames'])==1):raise ValueError('Consecutive native frames required.')
        self.fps=float(data['fps']);self.known=np.asarray(data['known'],bool)
        # Reuse the exact contact rectangles, shared episode mapping and offsets.
        self.base=ContactMetadata(data,scene,regions)
        self.e=len(regions['episodes']);mapping=regions['mapping'];ix=np.maximum(mapping,0)
        valid=np.repeat((mapping>=0)[...,None],2,axis=2)
        rr=np.broadcast_to(np.arange(self.n)[:,None,None]*self.localdim+self.statedim+np.arange(8).reshape(1,4,2),valid.shape)
        cc=ix[...,None]*2+np.arange(2)
        rows=np.arange(self.n*self.statedim)//self.statedim*self.localdim+np.arange(self.n*self.statedim)%self.statedim
        self.expand=coo_matrix((np.ones(len(rows)+valid.sum()),
            (np.r_[rows,rr[valid]],np.r_[np.arange(self.n*self.statedim),self.n*self.statedim+cc[valid]])),
            shape=(self.n*self.localdim,self.n*self.statedim+self.e*2)).tocsr()
        lo,hi,self.speed,self.accel=self.pose_bounds()
        blo=np.tile(np.r_[lo,[-self.limits.force_component_bw]*12],(self.n,1));bhi=np.tile(np.r_[hi,[self.limits.force_component_bw]*12],(self.n,1))
        # Unknown/free force variables are exactly zero, not merely masked.
        blo[:,self.qdim:].reshape(self.n,4,3)[~self.known]=0
        bhi[:,self.qdim:].reshape(self.n,4,3)[~self.known]=0
        self.lower=np.r_[blo.ravel(),self.base.lower[-2*self.e:]]
        self.upper=np.r_[bhi.ravel(),self.base.upper[-2*self.e:]]
        q=np.c_[initial_q,np.zeros((self.n,12))]
        if 'audit_forces_bw' in legacy:q[:,self.qdim:]=legacy['audit_forces_bw'].reshape(self.n,12)
        self.initial=np.clip(np.r_[q.ravel(),legacy['episode_uv'].ravel()],self.lower,self.upper)
        self.reference=self.body_kinematics(initial_q)
        self.qref=initial_q.copy()
        self.temporal=[]
        for order,coef in [(1,[-1,1]),(2,[1,-2,1])]:
            D=diags([np.full(self.n-order,x,dtype=float) for x in coef],np.arange(order+1),shape=(self.n-order,self.n)).tocsr()
            self.temporal.append((order,D))
        qs=coo_matrix((np.ones(self.n*self.qdim),(np.arange(self.n*self.qdim),
            (np.arange(self.n)[:,None]*self.localdim+np.arange(self.qdim)).ravel())),shape=(self.n*self.qdim,self.n*self.localdim)).tocsr()@self.expand
        self.hard_rates=vstack([kron(D,diags(self.fps**order/(self.speed if order==1 else self.accel)),format='csr')@qs for order,D in self.temporal],format='csr')

    def pose_bounds(self):
        return arms.bounds_and_rates()

    def body_kinematics(self,q):
        return arms.kinematics(q,self.lengths,self.height)

    def geometry(self,y):
        g=self.body_kinematics(y[:,:self.qdim])
        uv=y[:,self.statedim:].reshape(self.n,4,2)
        xy=self.base.rect[...,:2]+uv*(self.base.rect[...,2:]-self.base.rect[...,:2])
        c=self.scene.intersect(xy,self.base.panels)*self.known[...,None]
        g.update(contacts=c,targets=c+self.base.offset)
        return g

    def local(self,y,g=None):
        g=self.geometry(y) if g is None else g;j,c,com=g['joints'],g['contacts'],g['com']
        guide=((self.scene.pixels(j)-self.data['observed_px'])/35*np.sqrt(self.data['visibility'][...,None])).reshape(self.n,-1)
        delta=(g.get('contact_points',j[:,:,2])-g.get('contact_targets',g['targets']))*self.known[...,None]
        gap=45*delta.reshape(self.n,-1)
        wall=(80*positive(-self.scene.wall_clearance(j))).reshape(self.n,-1)
        floor=(80*positive(-j[...,1])).reshape(self.n,-1)
        if getattr(self.scene,'floor_y_cm',0.) is None:floor=np.zeros_like(floor)
        behind=(2*positive(10-(j@self.scene.R.T+self.scene.t)[...,2])).reshape(self.n,-1)
        v=np.diff(j[:,2:],axis=2);cos=(v[:,:,0]*v[:,:,1]).sum(-1)/(self.lengths['thigh']*self.lengths['shank'])
        bend=positive(np.cos(np.deg2rad(self.limits.leg_bend_deg))-cos)*1000
        pose=.04*(y[:,3:self.qdim]-self.qref[:,3:self.qdim])
        normal=self.scene.normals[self.base.seed_panel];origin=self.scene.origins[self.base.seed_panel]
        depth=(((y[:,:3]-origin)*normal).sum(-1)-25)/20
        prior=((c-self.base.prior)/6*self.base.evidence_scale[...,None]*self.known[...,None]).reshape(self.n,-1)
        grip=(3*(g['palm_contacts']-c[:,:2])*self.known[:,:2,None]).reshape(self.n,-1)
        if 'contact_points' in g:grip=np.zeros_like(grip)  # Already constrained by the same contact residual.
        force=y[:,self.qdim:self.statedim].reshape(self.n,4,3)
        resultant=force.sum(1)-[0,1,0]
        moment=np.cross(c-com[:,None],force).sum(1)/self.height
        nf=np.sum(force*self.base.normals,axis=-1);friction=[positive(-nf[:,2:])]
        for tangent in self.base.tangents:
            ft=np.sum(force*tangent,axis=-1)
            for sign in [-1,1]:friction.append(positive(sign*ft[:,2:]-self.limits.foot_friction/np.sqrt(2)*nf[:,2:]))
        if getattr(self,'dynamic_balance',False):
            resultant=np.zeros_like(resultant);moment=np.zeros_like(moment)
        physics=100*np.c_[resultant,moment,*friction]
        unload=(200*force*self.data['unload'][...,None]).reshape(self.n,-1)
        coupling=(10*delta[...,None]*force[:,:,None,:]).reshape(self.n,-1)
        return np.c_[guide,gap,wall,floor,behind,bend,pose,depth,prior,grip,physics,unload,coupling,.1*y[:,self.qdim:self.statedim]]

    def evaluate(self,flat):
        y=(self.expand@flat).reshape(self.n,self.localdim);g=self.geometry(y);r=self.local(y,g)
        # Smooth physical rotation matrices as well as coordinates: Euler pole
        # ambiguity must not masquerade as segment motion. Frobenius residuals
        # avoid acos singularities and wrap discontinuities in the optimizer.
        features=np.c_[g['joints'].reshape(self.n,36),g['orientations'].reshape(self.n,-1)]
        deriv=np.empty((*r.shape,self.localdim));fd=np.empty((*features.shape,self.localdim));cy=y.astype(complex)
        for k in range(self.localdim):
            cy[:,k]+=1e-25j;cg=self.geometry(cy)
            deriv[:,:,k]=self.local(cy,cg).imag/1e-25
            fd[:,:,k]=np.c_[cg['joints'].reshape(self.n,36),cg['orientations'].reshape(self.n,-1)].imag/1e-25
            cy[:,k]-=1e-25j
        def sparse(d):
            n,m,k=d.shape;row=np.broadcast_to(np.arange(n*m).reshape(n,m,1),d.shape).ravel()
            col=np.broadcast_to(np.arange(n)[:,None,None]*self.localdim+np.arange(self.localdim),d.shape).ravel()
            M=coo_matrix((d.ravel(),(row,col)),shape=(n*m,n*self.localdim)).tocsr();M.eliminate_zeros()
            return M@self.expand
        jac=sparse(deriv);fj=sparse(fd);res=[r.ravel()];jacs=[jac]
        qselect=coo_matrix((np.ones(self.n*self.qdim),(np.arange(self.n*self.qdim),
            (np.arange(self.n)[:,None]*self.localdim+np.arange(self.qdim)).ravel())),shape=(self.n*self.qdim,self.n*self.localdim)).tocsr()@self.expand
        for order,D in self.temporal:
            rate=self.speed if order==1 else self.accel
            scale=self.fps**order/rate
            T=kron(D,diags(scale),format='csr')@qselect
            a=np.asarray(T@flat);gain=self.smoothness*(1.5 if order==1 else 1.)
            res.extend([gain*a,150*positive(a-1),150*positive(-a-1)])
            jacs.extend([gain*T,diags(150.*(a>1))@T,diags(-150.*(a< -1))@T])
            # Same XYZ buffers/audits as the prior experiment; rotations use
            # seconds and have additional smoothness without Cartesian filtering.
            weights=np.r_[np.full(36,.25 if order==1 else 2.),
                          np.full(features.shape[1]-36,self.smoothness*(.35 if order==1 else .025)*self.fps**order)]
            T=kron(D,diags(weights),format='csr')
            res.append(T@features.ravel());jacs.append(T@fj)
            dt=(D@features[:,:36]).reshape(-1,3);norm=np.linalg.norm(dt,axis=1)
            limit,weight=(6,35) if order==1 else (3,25)
            res.append(weight*np.maximum(norm-limit,0))
            G=coo_matrix(((weight*dt/np.maximum(norm[:,None],1e-12)*(norm>limit)[:,None]).ravel(),
                (np.repeat(np.arange(len(norm)),3),np.arange(len(norm)*3))),shape=(len(norm),len(norm)*3)).tocsr()
            select=coo_matrix((np.ones(self.n*36),(np.arange(self.n*36),(np.arange(self.n)[:,None]*features.shape[1]+np.arange(36)).ravel())),shape=(self.n*36,self.n*features.shape[1])).tocsr()
            jacs.append(G@kron(D,eye(36),format='csr')@select@fj)
        return np.concatenate(res),vstack(jacs,format='csr')

    def solve(self,iterations=300,progress=None):
        if self.n>120:
            # Reuse the existing joint-curve representation for long sequences.
            # Forces remain independent at every supported native frame. All
            # residuals, ROM/rate bounds and final audits still use native poses.
            from .temporal_control_basis import control_map
            T,initial,lower,upper,info=control_map(self,8)
            def evaluate(z):
                residual,J=self.evaluate(T@z)
                return residual,J@T
            if progress:progress(f"Joint controls: {len(self.initial)} -> {len(initial)} variables; all {self.n} frames checked")
            fit=constrained_solve(evaluate,initial,lower,upper,self.hard_rates@T,iterations,progress)
            x=T@fit.pop('x');fit['parameterization']=info
        else:
            fit=constrained_solve(self.evaluate,self.initial,self.lower,self.upper,self.hard_rates,iterations,progress)
            x=fit.pop('x')
        y=(self.expand@x).reshape(self.n,self.localdim);g=self.geometry(y)
        return dict(**g,parameters=x,values=y[:,:self.statedim],anatomical_q=y[:,:self.qdim],
                    episode_uv=x[-2*self.e:].reshape(self.e,2),offsets=self.base.offset,solver=fit)


def audit(fit,data,scene,regions,height_cm=180.,ratios=None):
    m,gaps,forces=audit_plan(fit,data,scene,regions,height_cm,ratios)
    q=fit['anatomical_q'];fps=float(data['fps'])
    if q.shape[1]==32:
        from .body_legs import bounds_and_rates
    else:bounds_and_rates=arms.bounds_and_rates
    lo,hi,speed,accel=bounds_and_rates()
    outside=np.maximum(lo-q,0)+np.maximum(q-hi,0)
    # Small comparison epsilon is numerical, not a relaxation of declared limits.
    vel=np.vstack([np.zeros((1,q.shape[1])),abs(np.diff(q,axis=0))*fps])
    acc=np.vstack([np.zeros((2,q.shape[1])),abs(np.diff(q,n=2,axis=0))*fps**2])
    rot=np.vstack([np.zeros((1,fit['orientations'].shape[1])),arms.rotation_steps_deg(fit['orientations'],fps)])
    swivel=arms.elbow_swivel_rates(fit['joints'],fps)
    swivelmax=np.max(np.where(np.isfinite(swivel),swivel,0),axis=1)
    grip=np.linalg.norm(fit['palm_contacts']-fit['contacts'][:,:2],axis=-1)
    m['anatomy_bounds_failed']=np.max(outside,axis=1)>1e-7
    m['rate_failed']=np.any(vel>speed[None]*(1+1e-4),axis=1)
    m['angular_acceleration_failed']=np.any(acc>accel[None]*(1+1e-4),axis=1)
    m['max_rate_ratio']=np.max(vel/speed,axis=1);m['max_acceleration_ratio']=np.max(acc/accel,axis=1)
    m['torso_rotation_deg_s']=rot[:,0];m['max_arm_segment_rotation_deg_s']=rot[:,1:7].max(axis=1)
    m['elbow_swivel_deg_s']=np.r_[0,swivelmax]
    m['elbow_swivel_observable']=np.r_[False,np.any(np.isfinite(swivel),axis=1)]
    m['grip_proxy_gap_cm']=np.max(np.where(data['known'][:,:2],grip,0),axis=1)
    m['grip_proxy_failed']=m.grip_proxy_gap_cm>2
    # No hand orientation observations: passing the proxy is still uncertain.
    m['wrist_orientation_uncertain']=True;m['full_self_collision_checked']=False
    for col,flag in [('anatomy_bounds_failed','anatomy_bounds'),('rate_failed','joint_rate'),
                     ('angular_acceleration_failed','joint_acceleration'),('grip_proxy_failed','grip_proxy')]:
        m.loc[m[col],'issues']=m.loc[m[col],'issues'].astype(str)+';'+flag
    m['issues']=m.issues.astype(str)+';wrist_orientation_inferred;anatomy_approximate'
    return m,gaps,forces


def selection_row(name,metrics,solver):
    m=metrics
    anatomy=m.anatomy_bounds_failed|m.rate_failed|m.angular_acceleration_failed
    support=~m.conditional_balance;unload=m.preunload.eq(False)
    failed=anatomy|m.geometric_failed|support|unload|m.grip_proxy_failed|m.motion_warning
    feasible=bool(not failed.any() and solver['success'])
    return dict(candidate=name,native_frames=len(m),numerically_feasible=feasible,
        anatomy_temporal_failure_frames=int(anatomy.sum()),contact_failure_frames=int(m.contact_failed.sum()),
        geometric_failure_frames=int(m.geometric_failed.sum()),support_failure_frames=int(support.sum()),
        preunload_failure_frames=int(unload.sum()),grip_proxy_failure_frames=int(m.grip_proxy_failed.sum()),
        total_failed_frames=int(failed.sum()),worst_failure_fraction=float(max(m.geometric_failed.mean(),support.mean(),unload.mean(),m.grip_proxy_failed.mean())),
        max_contact_gap_cm=float(m.max_contact_gap_cm.max()),max_rate_ratio=float(m.max_rate_ratio.max()),
        max_acceleration_ratio=float(m.max_acceleration_ratio.max()),max_elbow_swivel_deg_s=float(m.elbow_swivel_deg_s.max()),
        max_torso_rotation_deg_s=float(m.torso_rotation_deg_s.max()),converged=bool(solver['success']),
        uncertainty='Unmeasured geometry, approximate anatomy, inferred wrist orientation; no dynamic or full collision validation.')


def rank_candidates(rows):
    # Feasibility gate precedes preferences. Infeasible diagnostic candidates
    # compare anatomy/temporal failures first, then worst physical failure fraction,
    # not geometry-first ordering that can bypass support failures.
    fields=['anatomy_temporal_failure_frames','worst_failure_fraction','total_failed_frames','grip_proxy_failure_frames','max_contact_gap_cm']
    ordered=sorted(rows,key=lambda r:(not r['numerically_feasible'],*[r[k] for k in fields],not r['converged']))
    winner=ordered[0]
    for i,row in enumerate(ordered):
        row['rank']=i+1
        row['selection_status']='numerically_feasible_under_assumptions' if row['numerically_feasible'] else 'diagnostic_only_not_feasible'
        if i==0:row['selection_reason']='All tested plans checked; '+('all numerical gates passed, uncertainty retained.' if row['numerically_feasible'] else 'no feasible plan; least-violating diagnostic under the declared anatomy/physics ordering.')
        else:
            key=next((k for k in ['numerically_feasible',*fields,'converged'] if row[k]!=winner[k]),None)
            row['selection_reason']=f'First differing criterion: {key} ({row.get(key)} vs {winner.get(key)}).' if key else 'Deterministic tie.'
    return ordered
