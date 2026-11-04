"""Joint native-frame trajectory and episode-contact optimization."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix,csr_matrix,kron,eye,vstack,diags,hstack
from .ik3d import SpatialProblem,kinematics,positive,smooth_positive,initial_pose,limb_basis
from .scene3d import project
from .contact_regions3d import homography,knot_basis,contacts_from_parameters
from .support_refinement import sparse_direct_least_squares,stationarity

PROFILES=('center','evidence_point','region','region_offset_half','region_offset')


class RegionProblem:
    def __init__(self,data,regions,profile='region_offset',height_cm=180.,tilt_deg=15.,depth_cm=25.,stride=4,ratios=None):
        if profile not in PROFILES:raise ValueError('Unknown region candidate.')
        self.data=data;self.regions=regions;self.profile=profile;self.n=len(data['frames'])
        self.base=SpatialProblem(data,tilt_deg,depth_cm,height_cm,force_weight=0,support_weight=0,ratios=ratios)
        self.lengths=self.base.lengths;self.known=np.asarray(data['known']);self.depth=depth_cm
        self.analytic=self.known.copy();self.direct_switch=np.zeros(self.known.shape,bool)
        # An instantaneous known-hold -> different-known-hold transition is not
        # a physically timed transfer. Keep its target and contact penalty, but
        # allow the joint endpoint to miss it instead of forcing a teleport.
        for i in range(4):
            changes=np.flatnonzero(self.known[1:,i]&self.known[:-1,i]&(data['labels'][1:,i]!=data['labels'][:-1,i]))+1
            for f in changes:self.direct_switch[max(0,f-6):min(self.n,f+7),i]=True
        self.analytic[self.direct_switch]=False
        basis_data={**data,'labels':np.char.add(np.asarray(data['labels'],str),self.analytic.astype(str))}
        self.knots,B=knot_basis(basis_data,stride);self.k=len(self.knots);self.e=len(regions['episodes'])
        episodes=regions['episodes'];mapping=regions['mapping'];ix=np.maximum(mapping,0)
        boxes=np.array([e['rectangle_px'] for e in episodes]);priors=np.array([e['prior_px'] for e in episodes])
        self.rect=boxes[ix];self.prior=priors[ix];self.H=regions['H']
        self.uv0=np.divide(priors-boxes[:,:2],boxes[:,2:]-boxes[:,:2],out=np.full((self.e,2),.5),where=(boxes[:,2:]-boxes[:,:2])>0)
        _,self.offset,_=contacts_from_parameters(self.uv0,regions,profile)
        count=np.bincount(mapping[mapping>=0],minlength=self.e)
        self.evidence_scale=1/np.sqrt(np.maximum(count[ix],1))
        self.prior_cm=homography(self.prior,self.H)
        self.lower=np.r_[self.base.lower[self.knots,:22].ravel(),np.zeros(self.e*2)]
        self.upper=np.r_[self.base.upper[self.knots,:22].ravel(),np.ones(self.e*2)]
        if profile in ('center','evidence_point'):
            self.lower[self.k*22:]=self.upper[self.k*22:]=self.uv0.ravel()
        else:
            frozen=np.array([not e['evidence_available'] for e in episodes])[:,None]|((boxes[:,2:]-boxes[:,:2])==0)
            self.lower[self.k*22:][frozen.ravel()]=self.uv0.ravel()[frozen.ravel()]
            self.upper[self.k*22:][frozen.ravel()]=self.uv0.ravel()[frozen.ravel()]
        # Linear expansion: pose knots plus shared episode UV -> each frame's
        # local pose and contact parameters. No frame gets a private grip point.
        pmap=kron(B,eye(22),format='coo')
        rows=(pmap.row//22)*30+pmap.row%22
        cr=np.arange(self.n*4*2).reshape(self.n,4,2);cc=ix[:,:,None]*2+np.arange(2)
        valid=np.repeat((mapping>=0)[:,:,None],2,axis=2)
        rr=np.broadcast_to(np.arange(self.n)[:,None,None]*30+22+np.arange(8).reshape(1,4,2),cr.shape)
        self.expand=coo_matrix((np.r_[pmap.data,np.ones(valid.sum())],
            (np.r_[rows,rr[valid]],np.r_[pmap.col,self.k*22+cc[valid]])),shape=(self.n*30,self.k*22+self.e*2)).tocsr()
        self.initial=np.r_[self.base.seed[self.knots].ravel(),self.uv0.ravel()]
        if 'initial_reference' in data:
            q=initial_pose(data['initial_reference'],depth_cm,self.lengths)
            initial_joints=kinematics(q,self.lengths)[0]
            contact,offset,_=contacts_from_parameters(self.uv0,regions,profile)
            delta=contact+offset-initial_joints[:,:,0]
            _,axis,u,v=limb_basis(delta,self.base.basis_helper)
            elbow=initial_joints[:,:,1]-initial_joints[:,:,0]
            swivel=np.arctan2((elbow*v).sum(-1),(elbow*u).sum(-1))
            for i in range(4):q[:,6+4*i]=np.where(self.analytic[:,i],swivel[:,i],q[:,6+4*i])
            q[:,6:]=np.unwrap(q[:,6:],axis=0)
            self.initial[:self.k*22]=q[self.knots].ravel()
        self.motion=[]
        for order,limit,weight in [(1,6.,50.),(2,3.,30.)]:
            if self.n<=order:continue
            coefficients=[-1.,1.] if order==1 else [1.,-2.,1.]
            D=diags([np.full(self.n-order,v) for v in coefficients],np.arange(order+1),shape=(self.n-order,self.n))
            self.motion.append((kron(D,eye(36),format='csr'),limit,weight))

    def geometry(self,y):
        if self.profile=='center':contact=np.nan_to_num(self.data['targets']).astype(y.dtype)
        else:
            uv=y[:,22:].reshape(self.n,4,2)
            xy=self.prior if self.profile=='evidence_point' else self.rect[:,:,:2]+uv*(self.rect[:,:,2:]-self.rect[:,:,:2])
            xy=homography(xy,self.H);contact=np.concatenate([xy,np.zeros((self.n,4,1),dtype=y.dtype)],axis=-1)
        contact=contact*self.known[:,:,None]
        target=contact+self.offset
        j,com,head=kinematics(y[:,:22],self.lengths,target,self.analytic,self.base.basis_helper)
        return j,com,head,contact,target

    def local(self,y):
        j,com,head,contact,target=self.geometry(y)
        pixels,z=project(j,self.data['K'],self.data['R'],self.data['t'])
        guide=((pixels-self.data['observed_px'])/40*np.sqrt(self.data['visibility'][...,None])).reshape(self.n,-1)
        gap=(20*(j[:,:,2]-target)*self.known[:,:,None]).reshape(self.n,-1)
        wall=(smooth_positive(-j[:,:,:,2],.01)*100).reshape(self.n,-1)
        behind=positive(10-z).reshape(self.n,-1)*.5
        l1=np.array([self.lengths['upper_arm']]*2+[self.lengths['thigh']]*2)
        l2=np.array([self.lengths['forearm']]*2+[self.lengths['shank']]*2)
        cos=((j[:,:,1]-j[:,:,0])*(j[:,:,2]-j[:,:,1])).sum(-1)/(l1*l2)
        flex=positive(np.cos(np.deg2rad(155))-cos)*100
        depth=(y[:,2]-self.depth)/10
        pose=.08*(y[:,3:22]-self.base.seed[:,3:22])
        prior=((contact[:,:,:2]-self.prior_cm)/6*self.evidence_scale[:,:,None]*self.known[:,:,None]).reshape(self.n,-1)
        if self.profile=='center':prior=prior*0
        return np.c_[guide,gap,wall,behind,flex,depth,pose,prior]

    def evaluate(self,flat):
        y=(self.expand@flat).reshape(self.n,30);r=self.local(y);nr=r.shape[1]
        deriv=np.empty((self.n,nr,30));jd=np.empty((self.n,36,30));cy=y.astype(complex)
        joints=self.geometry(y)[0].reshape(self.n,36)
        for i in range(30):
            cy[:,i]+=1e-25j;deriv[:,:,i]=self.local(cy).imag/1e-25
            jd[:,:,i]=self.geometry(cy)[0].imag.reshape(self.n,36)/1e-25;cy[:,i]-=1e-25j
        def sparse(d):
            a,b,c=d.shape
            rows=np.broadcast_to(np.arange(a*b).reshape(a,b,1),d.shape).ravel()
            cols=np.broadcast_to(np.arange(a)[:,None,None]*30+np.arange(30),d.shape).ravel()
            m=coo_matrix((d.ravel(),(rows,cols)),shape=(a*b,a*30)).tocsr();m.eliminate_zeros()
            return m@self.expand
        J=sparse(deriv);jj=sparse(jd)
        residuals=[r.ravel(),self.base.joint_smooth@joints.ravel()]
        jacobians=[J,self.base.joint_smooth@jj]
        for D,limit,weight in self.motion:
            delta=(D@joints.ravel()).reshape(-1,3);norm=np.linalg.norm(delta,axis=1)
            residuals.append(weight*np.maximum(norm-limit,0))
            grad=weight*delta/np.maximum(norm[:,None],1e-12)*(norm>limit)[:,None]
            G=coo_matrix((grad.ravel(),(np.repeat(np.arange(len(norm)),3),np.arange(len(norm)*3))),shape=(len(norm),len(norm)*3)).tocsr()
            jacobians.append(G@D@jj)
        return np.concatenate(residuals),vstack(jacobians,format='csr')

    def solve(self,max_iterations=600,initial=None,progress=None):
        started=time.monotonic();initial=self.initial if initial is None else initial
        fit=sparse_direct_least_squares(self.evaluate,initial,self.lower,self.upper,
            settings={'max_inner_iterations':int(max_iterations)},progress=progress)
        stages=[]
        if not fit['success']:
            stages.append({k:v for k,v in fit.items() if k not in ('x','history')})
            scale=np.r_[np.tile(np.r_[np.full(3,40.),np.ones(19)],self.k),np.ones(self.e*2)]
            evaluations=[0]
            def objective(y):
                r,J=self.evaluate(y*scale);evaluations[0]+=1
                if progress and evaluations[0]%100==0:progress(f'L-BFGS evaluation {evaluations[0]}: cost {r@r/2:.9g}')
                return float(r@r/2),np.asarray(J.T@r).ravel()*scale
            opt=minimize(objective,fit['x']/scale,jac=True,method='L-BFGS-B',
                bounds=list(zip(self.lower/scale,self.upper/scale)),
                options={'maxiter':int(max_iterations),'maxls':40,'maxcor':30,'ftol':1e-14,'gtol':1e-8})
            if np.isfinite(opt.fun) and opt.fun<fit['cost']:
                stages.append(dict(method='L-BFGS-B',cost=float(opt.fun),nfev=int(opt.nfev),message=str(opt.message)))
                # A final direct solve supplies the same stationarity and cost
                # stabilization check as the first stage, not SciPy's ftol alone.
                fit=sparse_direct_least_squares(self.evaluate,opt.x*scale,self.lower,self.upper,
                    settings={'max_inner_iterations':min(300,int(max_iterations))},progress=progress)
        x=fit.pop('x');r,J=self.evaluate(x)
        fit['optimality']=stationarity(x,r,J,self.lower,self.upper)[-1]
        fit['success']=bool(fit['success'] and fit['optimality']<=1e-5)
        fit['seconds']=time.monotonic()-started
        fit['earlier_stages']=stages
        y=(self.expand@x).reshape(self.n,30);j,com,head,c,t=self.geometry(y)
        return dict(parameters=x,values=np.c_[y[:,:22],np.zeros((self.n,12))],joints=j,com=com,head=head,
            contacts=c,targets=t,offsets=self.offset,solver=fit,basis_helper=self.base.basis_helper,
            candidate_forces_fitted=False,knots=self.knots,direct_switch=self.direct_switch)
