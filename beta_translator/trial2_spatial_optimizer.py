"""Whole-sequence fixed-bone IK on joined panels, with conditional support forces."""
import time
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,eye,kron,diags,vstack
from scipy.optimize import linprog
from .ik3d import kinematics,positive,limb_basis
from .inverse_kinematics import make_body
from .contact_regions3d import knot_basis
from .support_refinement import sparse_direct_least_squares,stationarity
from .trial2_spatial_data import seed_pose


def transported_helper(axis,reference,helper):
    """Carry the initial swivel frame with the reach axis without a pole at Z.

    The minimal rotation is smooth away from the antipode of the initial axis.
    Only that antipodal cap uses a second chart; all charts stay nondegenerate.
    """
    cross=np.cross(reference,axis);cosine=np.sum(reference*axis,axis=-1)
    safe=np.where(cosine.real>-.99,1+cosine,1.)
    rotated=helper+np.cross(cross,helper)+np.cross(cross,np.cross(cross,helper))/safe[...,None]
    fallback=np.eye(3)[np.argmin(abs(axis.real),axis=-1)]
    return np.where((cosine.real>-.99)[...,None],rotated,fallback)


from .body_contacts import contact_geometry


class Trial2Problem:
    def __init__(self,data,scene,regions,height_cm=180.,ratios=None,profile='region_offset',
                 support=False,technique='none',stride=4,initial=None,analytic_contacts=True):
        if profile not in ('center','region_offset'):raise ValueError('Unknown contact profile')
        if technique not in ('none','leg_drive','flag_assist','arm_economy'):raise ValueError('Unknown technique')
        self.data=data;self.scene=scene;self.regions=regions;self.n=len(data['frames'])
        self.height=height_cm;self.lengths=make_body(height_cm,ratios);self.profile=profile
        self.support=bool(support);self.technique=technique;self.known=data['known']
        self.analytic=self.known.copy() if analytic_contacts else np.zeros_like(self.known)
        self.direct_switch=np.zeros_like(self.known)
        for limb in range(4):
            changes=np.flatnonzero(self.known[1:,limb]&self.known[:-1,limb]&(data['labels'][1:,limb]!=data['labels'][:-1,limb]))+1
            for f in changes:self.direct_switch[max(0,f-6):min(self.n,f+7),limb]=True
        self.analytic[self.direct_switch]=False
        basis_data={**data,'labels':np.char.add(data['labels'].astype(str),self.analytic.astype(str))}
        self.knots,B=knot_basis(basis_data,stride);self.k=len(self.knots);self.e=len(regions['episodes'])
        self.seed=seed_pose(data,scene,height_cm,ratios)
        self.seed_joints,self.seed_com,_=kinematics(self.seed,self.lengths)
        self.seed_panel=scene.panel(data['observed_px'][:,2:,0].mean(1))
        eps=regions['episodes'];box=np.array([e['rectangle_px'] for e in eps]);anchor=np.array([e['anchor_px'] for e in eps])
        self.uv0=np.divide(anchor-box[:,:2],box[:,2:]-box[:,:2],out=np.full((self.e,2),.5),where=box[:,2:]!=box[:,:2])
        mapping=regions['mapping'];ix=np.maximum(mapping,0);self.ix=ix;self.rect=box[ix]
        self.panels=np.array([e['panel'] for e in eps])[ix]
        self.normals=scene.normals[self.panels]
        self.offset=np.array([e['offset_cm'] for e in eps])[ix]*self.known[...,None]
        if profile=='center':self.offset*=0
        self.center=np.array([e['center_px'] for e in eps])[ix]
        self.prior=scene.intersect(anchor,np.array([e['panel'] for e in eps]))[ix]
        contact,offset,_,_=contact_geometry(scene,regions,self.uv0,profile)
        seed_j=self.seed_joints if initial is None else np.asarray(initial['joints'])
        delta=contact+offset-seed_j[:,:,0]
        self.basis_helper=np.broadcast_to([0.,0.,1.],delta.shape).copy()
        unit=delta/np.maximum(np.linalg.norm(delta,axis=-1)[...,None],1e-12)
        self.reach_axis=unit.copy()
        self.basis_helper[abs(unit[...,2])>.95]=[1.,0.,0.]
        _,axis,u,v=limb_basis(delta,self.basis_helper)
        elbow=seed_j[:,:,1]-seed_j[:,:,0]
        swivel=np.arctan2(np.sum(elbow*v,axis=-1),np.sum(elbow*u,axis=-1))
        count=np.bincount(mapping[mapping>=0],minlength=self.e)
        self.evidence_scale=1/np.sqrt(np.maximum(count[ix],1))
        pmap=kron(B,eye(34),format='coo')
        rows=pmap.row//34*42+pmap.row%34
        valid=np.repeat((mapping>=0)[...,None],2,axis=2)
        rr=np.broadcast_to(np.arange(self.n)[:,None,None]*42+34+np.arange(8).reshape(1,4,2),valid.shape)
        cc=ix[...,None]*2+np.arange(2)
        self.expand=coo_matrix((np.r_[pmap.data,np.ones(valid.sum())],
            (np.r_[rows,rr[valid]],np.r_[pmap.col,self.k*34+cc[valid]])),shape=(self.n*42,self.k*34+self.e*2)).tocsr()
        q=np.c_[self.seed,np.zeros((self.n,12))]
        q[:,23::3]=np.divide(1.,self.known.sum(1),out=np.zeros(self.n),where=self.known.sum(1)>0)[:,None]
        if initial is not None:
            q=np.asarray(initial['values']).copy()
            # Recover physical segment directions before selecting analytic limbs;
            # this makes warm starts independent of earlier parameter conventions.
            vectors=np.diff(seed_j,axis=2)
            for limb in range(4):
                for bone in range(2):
                    v0=vectors[:,limb,bone]
                    q[:,6+4*limb+2*bone]=np.unwrap(np.arctan2(v0[:,0],-v0[:,1]))
                    q[:,7+4*limb+2*bone]=np.arctan2(v0[:,2],np.linalg.norm(v0[:,:2],axis=1))
        for limb in range(4):
            q[:,6+4*limb]=np.where(self.analytic[:,limb],swivel[:,limb],q[:,6+4*limb])
        self.seed=q[:,:22].copy()
        uv=self.uv0 if initial is None else np.asarray(initial['episode_uv'])
        self.initial=np.r_[q[self.knots].ravel(),uv.ravel()]
        lo=np.tile(np.r_[[-1000.,-30.,-1500.],[-10.]*3,np.tile([-30.,-np.pi/2],8),[-1.]*12],(self.k,1))
        hi=np.tile(np.r_[[1000.,1600.,1000.],[10.]*3,np.tile([30.,np.pi/2],8),[1.]*12],(self.k,1))
        if not support:lo[:,22:]=hi[:,22:]=0
        self.lower=np.r_[lo.ravel(),np.zeros(self.e*2)];self.upper=np.r_[hi.ravel(),np.ones(self.e*2)]
        frozen=(box[:,2:]==box[:,:2]).ravel()
        if profile=='center':frozen[:]=True
        self.lower[self.k*34:][frozen]=uv.ravel()[frozen];self.upper[self.k*34:][frozen]=uv.ravel()[frozen]
        self.initial=np.clip(self.initial,self.lower,self.upper)
        self.motion=[]
        for order,coef,limit,weight,smooth in [(1,[-1,1],6.,35.,.25),(2,[1,-2,1],3.,25.,2.)]:
            if self.n>order:
                D=diags([np.full(self.n-order,v,dtype=float) for v in coef],np.arange(order+1),shape=(self.n-order,self.n))
                self.motion.append((kron(D,eye(36),format='csr'),limit,weight,smooth))
        tangent=np.broadcast_to([1.,0.,0.],self.normals.shape).copy()
        tangent-=np.sum(tangent*self.normals,axis=-1)[...,None]*self.normals
        tangent/=np.linalg.norm(tangent,axis=-1)[...,None]
        self.tangents=(tangent,np.cross(self.normals,tangent))
        self.leg_phase=np.zeros(self.n)
        releases=np.flatnonzero(np.any(data['unload'][:,:2]>0,axis=1))
        for f in releases:
            a=max(0,f-15);b=min(self.n,f+16);self.leg_phase[a:b]=np.maximum(self.leg_phase[a:b],np.exp(-((np.arange(a,b)-f)/10)**2))

    def geometry(self,y):
        uv=y[:,34:].reshape(self.n,4,2)
        xy=self.center if self.profile=='center' else self.rect[...,:2]+uv*(self.rect[...,2:]-self.rect[...,:2])
        contact=self.scene.intersect(xy,self.panels)*self.known[...,None]
        target=contact+self.offset
        # A contacted limb cannot fold beyond the declared 155-degree bend.
        # When a target is too close, preserve exact bones and report a gap to the
        # original surface target instead of silently allowing a 180-degree fold.
        roots=kinematics(y[:,:22],self.lengths)[0][:,:,0]
        delta=target-roots;distance=np.sqrt(np.sum(delta*delta,axis=-1)+1e-20)
        # A coincident target has no reach direction. Keep the initial direction
        # and retain a visible miss; a zero vector must not collapse fixed bones.
        delta=np.where((distance.real<1e-8)[...,None],self.reach_axis*1e-7,delta)
        distance=np.sqrt(np.sum(delta*delta,axis=-1)+1e-20)
        l1=np.array([self.lengths['upper_arm']]*2+[self.lengths['thigh']]*2)
        l2=np.array([self.lengths['forearm']]*2+[self.lengths['shank']]*2)
        minimum=np.sqrt(l1*l1+l2*l2+2*l1*l2*np.cos(np.deg2rad(155)))
        scale=np.where(distance.real<minimum,minimum/distance,1.)
        constrained_target=roots+delta*scale[...,None]
        # Transport the reference frame smoothly, rather than normalizing a cross
        # product that can vanish when a fixed helper meets the optimized axis.
        unit=delta/distance[...,None]
        helper=transported_helper(unit,self.reach_axis,self.basis_helper)
        joints,com,head=kinematics(y[:,:22],self.lengths,constrained_target,self.analytic,helper)
        return joints,com,head,contact,target

    def local(self,y):
        j,com,head,c,target=self.geometry(y)
        px=self.scene.pixels(j)
        guide=((px-self.data['observed_px'])/35*np.sqrt(self.data['visibility'][...,None])).reshape(self.n,-1)
        gap=(45*(j[:,:,2]-target)*self.known[...,None]).reshape(self.n,-1)
        clearance=self.scene.wall_clearance(j)
        wall=(80*positive(-clearance)).reshape(self.n,-1)
        floor=(80*positive(-j[...,1])).reshape(self.n,-1)
        camera_z=(j@self.scene.R.T+self.scene.t)[...,2]
        behind=(positive(10-camera_z)*2).reshape(self.n,-1)
        lengths=np.array([[self.lengths['upper_arm'],self.lengths['forearm']]]*2+[[self.lengths['thigh'],self.lengths['shank']]]*2)
        vectors=np.diff(j,axis=2);cos=(vectors[:,:,0]*vectors[:,:,1]).sum(-1)/np.prod(lengths,axis=1)
        flex=positive(np.cos(np.deg2rad(155))-cos)*1000
        pose=.04*(y[:,3:22]-self.seed[:,3:22])
        normal=self.scene.normals[self.seed_panel];origin=self.scene.origins[self.seed_panel]
        depth=(((y[:,:3]-origin)*normal).sum(-1)-25)/20
        prior=((c-self.prior)/6*self.evidence_scale[...,None]*self.known[...,None]).reshape(self.n,-1)
        if self.profile=='center':prior*=0
        force=y[:,22:34].reshape(self.n,4,3)*self.known[...,None]
        resultant=force.sum(1)-[0.,1.,0.]
        moment=np.cross(c-com[:,None],force).sum(1)/self.height
        normal_force=(force*self.normals).sum(-1)
        friction=[positive(-normal_force[:,2:])]
        for tangent in self.tangents:
            ft=(force*tangent).sum(-1)
            for sign in [-1,1]:friction.append(positive(sign*ft[:,2:]-.8/np.sqrt(2)*normal_force[:,2:]))
        physics=np.c_[resultant,moment,*friction]*100 if self.support else np.zeros((self.n,16),dtype=y.dtype)
        unload=(200*force*self.data['unload'][...,None]).reshape(self.n,-1) if self.support else force.reshape(self.n,-1)*0
        force_prior=.1*y[:,22:34]
        foot=self.known[:,2:];count=foot.sum(1);usable=count>0
        foot_center=np.sum(c[:,2:]*foot[...,None],axis=1)/np.maximum(count[:,None],1)
        shift=(com[:,0]-foot_center[:,0])/30*usable
        drive=(y[:,1]-self.seed[:,1]-.03*self.height*self.leg_phase)/20*usable
        technique=np.zeros((self.n,4),dtype=y.dtype)
        if self.technique in ('leg_drive','flag_assist','arm_economy'):technique[:,:2]=np.c_[shift,drive]
        if self.technique=='flag_assist':
            free=self.data['raw_labels'][:,2:]=='free'
            desired=self.seed_joints[:,2:,2,0]+np.array([-.06,.06])*self.height
            technique[:,2:]=(j[:,2:,2,0]-desired)/20*free
        economy=force[:,:2].reshape(self.n,-1)*(.8 if self.technique=='arm_economy' else 0)
        return np.c_[guide,gap,wall,floor,behind,flex,pose,depth,prior,physics,unload,force_prior,technique,economy]

    def evaluate(self,flat):
        y=(self.expand@flat).reshape(self.n,42);r=self.local(y);nr=r.shape[1]
        deriv=np.empty((self.n,nr,42));jd=np.empty((self.n,36,42));cy=y.astype(complex)
        joints=self.geometry(y)[0].reshape(self.n,36)
        for i in range(42):
            cy[:,i]+=1e-25j;deriv[:,:,i]=self.local(cy).imag/1e-25
            jd[:,:,i]=self.geometry(cy)[0].imag.reshape(self.n,36)/1e-25;cy[:,i]-=1e-25j
        def sparse(d):
            a,b,c=d.shape;rows=np.broadcast_to(np.arange(a*b).reshape(a,b,1),d.shape).ravel()
            cols=np.broadcast_to(np.arange(a)[:,None,None]*42+np.arange(42),d.shape).ravel()
            m=coo_matrix((d.ravel(),(rows,cols)),shape=(a*b,a*42)).tocsr();m.eliminate_zeros()
            return m@self.expand
        residuals=[r.ravel()];jacobians=[sparse(deriv)];jj=sparse(jd)
        for D,limit,weight,smooth in self.motion:
            delta=(D@joints.ravel()).reshape(-1,3);norm=np.linalg.norm(delta,axis=1)
            residuals.extend([smooth*delta.ravel(),weight*np.maximum(norm-limit,0)])
            grad=weight*delta/np.maximum(norm[:,None],1e-12)*(norm>limit)[:,None]
            G=coo_matrix((grad.ravel(),(np.repeat(np.arange(len(norm)),3),np.arange(len(norm)*3))),shape=(len(norm),len(norm)*3)).tocsr()
            jacobians.extend([smooth*D@jj,G@D@jj])
        return np.concatenate(residuals),vstack(jacobians,format='csr')

    def solve(self,max_iterations=250,progress=None):
        started=time.monotonic()
        fit=sparse_direct_least_squares(self.evaluate,self.initial,self.lower,self.upper,
            settings={'max_inner_iterations':int(max_iterations)},progress=progress)
        x=fit.pop('x');r,J=self.evaluate(x)
        fit['optimality']=float(stationarity(x,r,J,self.lower,self.upper)[-1])
        fit['success']=bool(fit['success'] and fit['optimality']<=1e-5);fit['seconds']=time.monotonic()-started
        y=(self.expand@x).reshape(self.n,42);j,com,head,c,t=self.geometry(y)
        return dict(parameters=x,values=y[:,:34],episode_uv=x[self.k*34:].reshape(self.e,2),joints=j,
            com=com,head=head,contacts=c,targets=t,offsets=self.offset,solver=fit,knots=self.knots,
            analytic_contacts=self.analytic,basis_helper=self.basis_helper,direct_switch=self.direct_switch)


from .body_support import force_audit, audit_plan


def rank_plans(rows):
    fields=['geometric_failure_frames','motion_warning_frames','nonconverged','support_failure_frames','preunload_failure_frames','low_margin_frames','hand_load_integral_bw_s']
    def key(r):return tuple(float(r[k]) if np.isfinite(float(r[k])) else float('inf') for k in fields)
    ordered=sorted(rows,key=key);winner=ordered[0]
    for i,row in enumerate(ordered):
        row['rank']=i+1
        if i==0:row['selection_reason']='Fewest declared failures among tested complete plans; assumed angles are not identified measurements.'
        else:
            first=next((k for k in fields if row[k]!=winner[k]),None)
            row['selection_reason']='Numerical tie; deterministic candidate order.' if first is None else f"First differing criterion: {first} ({row[first]} vs selected {winner[first]})."
    return ordered,fields
