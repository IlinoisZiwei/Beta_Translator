"""Global contact/force/COM optimization with prospective unloading objectives."""
import numpy as np
from scipy.optimize import least_squares, linprog
from scipy.sparse import coo_matrix, vstack

from .gravity_sequence import GravityProblem, batch_com_and_jacobian, diagnose_gravity_sequence
from .motion_ik import batch_joints
from .contact_geometry import contact_points, support_ids
from .full_sequence import endpoint_jacobian
from .support_audit import audit_settings, release_events, audit_support_sequence, attach_audit

DEFAULT_OPTIMIZATION_SETTINGS = dict(contact_gain=30., robust_contact_cm=1.,
    force_contact_weight=30., reserve_weight=80., unloading_weight=100.,
    unload_force_weight=30., motion_limit_weight=200., rounds=3, max_nfev=800)


def optimization_settings(settings=None):
    c={**DEFAULT_OPTIMIZATION_SETTINGS,**(settings or {})}
    if (set(c)!=set(DEFAULT_OPTIMIZATION_SETTINGS) or any(not np.isfinite(v) or v<=0 for v in c.values())
            or any(type(c[k]) is not int for k in ('rounds','max_nfev'))):
        raise ValueError('Invalid support optimization settings.')
    return c


def support_interval(targets,active,limit_bw,mu,reserve):
    """Exact COM-x interval for fixed planar contacts under an assumed reserve.

    With sum fx=0 and sum fy=1, moment balance gives COM.x=sum(x*fy-y*fx).
    Thus two LPs project the bounded contact-force polytope onto COM.x.
    """
    ids=np.flatnonzero(active)
    if not len(ids):return dict(status='no_support',bounds=None)
    m=len(ids);cap=limit_bw*(1-reserve)
    eq=np.zeros((2,2*m));eq[0,::2]=1;eq[1,1::2]=1
    target=np.asarray(targets)[ids]
    objective=np.stack([-target[:,1],target[:,0]],axis=1).ravel()
    bounds=[];rows=[]
    for k,i in enumerate(ids):
        bounds.extend([(-cap,cap),(0 if i>=2 else -cap,cap)])
        if i>=2:
            for sign in (-1,1):
                row=np.zeros(2*m);row[2*k]=sign;row[2*k+1]=-mu;rows.append(row)
    extremes=[]
    for sign in (1,-1):
        fit=linprog(sign*objective,A_eq=eq,b_eq=[0,1],
            A_ub=np.asarray(rows) if rows else None,b_ub=np.zeros(len(rows)) if rows else None,
            bounds=bounds,method='highs')
        if fit.status==2:return dict(status='infeasible_capacity',bounds=None)
        if not fit.success or not np.isfinite(fit.x).all():return dict(status='not_evaluated',bounds=None)
        if np.max(np.abs(eq@fit.x-[0,1]))>1e-7:return dict(status='not_evaluated',bounds=None)
        extremes.append(float(objective@fit.x))
    return dict(status='available',bounds=extremes)


def prospective_constraints(schedule,frames,limit_bw,mu,config):
    """Desired contacts, not claims of valid support; final audit rechecks reach."""
    cache={};rows=[];n=len(frames);unload=np.zeros((n,4))
    eligible=schedule['known'] & schedule['supported'] & ~schedule['low_confidence']
    def append(t,mask,kind,event_id='',moving=None,phase=1.):
        groups=[('eligible',mask&eligible[t]),('conditional',mask&schedule['known'][t])]
        for group,active in groups:
            # Tighten the interval for the same +/- COM sensitivity used in audit.
            key=(tuple(schedule['targets'][t].ravel()),tuple(active),float(limit_bw),float(mu),config['margin_warning'])
            if key not in cache:cache[key]=support_interval(schedule['targets'][t],active,limit_bw,mu,min(.999999,config['margin_warning']+.02))
            v=cache[key];bounds=v['bounds']
            row=dict(frame=int(frames[t]),index=t,kind=kind,event_id=event_id,group=group,
                limbs=np.flatnonzero(active).tolist(),status=v['status'],lower_cm=None,upper_cm=None,phase=phase)
            if bounds is not None:
                lo,hi=bounds;delta=config['com_perturbation_cm']+.1
                if lo+delta<=hi-delta:lo+=delta;hi-=delta
                else:row['status']='sensitivity_interval_empty';lo=hi=(lo+hi)/2
                row.update(lower_cm=lo,upper_cm=hi)
            rows.append(row)
            if bounds is not None:break
    for t in range(n):append(t,np.ones(4,bool),'reserve')
    for e in release_events(schedule,frames):
        release=e['release_index'];moving=e['moving_indices'];window=[]
        for t in range(release-1,max(-1,release-config['unload_lookahead_frames']-1),-1):
            if not np.array_equal(support_ids(schedule)[t,moving],support_ids(schedule)[release-1,moving]):break
            window.append(t)
        for t in sorted(window):
            active=np.ones(4,bool);active[moving]=False
            phase=(len(window)-(release-1-t))/len(window)
            append(t,active,'unloading',e['event_id'],moving,phase)
            unload[t,moving]=np.maximum(unload[t,moving],phase)
    return rows,unload


class SupportProblem(GravityProblem):
    """Keep all frames coupled; add analytic contact/force/COM derivatives."""
    def configure(self,previous_poses,optimization,audit):
        self.opt=optimization_settings(optimization);self.audit=audit_settings(audit)
        ends=contact_points(batch_joints(previous_poses,self.lengths)[:,:,2],self.schedule)
        gap=np.linalg.norm(ends-self.schedule['targets'],axis=2)
        # Frozen IRLS weights favor retaining attainable contacts instead of
        # spreading an impossible span error across every supporting limb.
        robust=1/np.sqrt(1+(gap/self.opt['robust_contact_cm'])**2)
        self.contact_weight*=self.opt['contact_gain']*robust
        self.constraints,self.unload=prospective_constraints(self.schedule,self.frames,self.limit_bw,self.mu,self.audit)
        self.extra=[r for r in self.constraints if r['lower_cm'] is not None]
        self.by_frame=[[] for _ in self.frames]
        for r in self.extra:self.by_frame[r['index']].append(r)
        self.max_regions=max(map(len,self.by_frame))
        self.residual_count=40+16+8+2*self.max_regions

    def local(self,x,indices):
        base=super().local(x,indices)
        q=x[:,:11];raw=x[:,11:].reshape(-1,4,2);known=self.schedule['known'][indices]
        force=raw*known[:,:,None]
        ends=contact_points(batch_joints(q,self.lengths)[:,:,2],self.schedule,indices);delta=ends-self.schedule['targets'][indices]
        com,_=batch_com_and_jacobian(q,self.lengths)
        # Candidate equilibrium is evaluated at intended hold centres. The
        # force-contact product penalizes carrying force across an endpoint gap.
        lever=(self.schedule['targets'][indices]-com[:,None])/self.lengths['torso']
        base[:,29]=self.config['balance_weight']*np.sum(lever[:,:,0]*force[:,:,1]-lever[:,:,1]*force[:,:,0],axis=1)
        coupled=self.opt['force_contact_weight']*(delta[:,:,:,None]*force[:,:,None,:]).reshape(-1,16)
        unloading=self.opt['unload_force_weight']*(force*self.unload[indices,:,None]).reshape(-1,8)
        region=np.zeros((len(x),2*self.max_regions))
        ts=np.arange(len(self.frames))[indices]
        for j,t in enumerate(ts):
            for k,r in enumerate(self.by_frame[t]):
                weight=self.opt['reserve_weight' if r['kind']=='reserve' else 'unloading_weight']/self.lengths['torso']
                region[j,2*k]=weight*max(0,r['lower_cm']-com[j,0])
                region[j,2*k+1]=weight*max(0,com[j,0]-r['upper_cm'])
        return np.c_[base,coupled,unloading,region]

    def local_jacobian(self,x,indices):
        original=super().local_jacobian(x,indices);n=len(x)
        jac=np.zeros((n,self.residual_count,19));jac[:,:40]=original
        q=x[:,:11];known=self.schedule['known'][indices]
        force=x[:,11:].reshape(-1,4,2)*known[:,:,None]
        delta=contact_points(batch_joints(q,self.lengths)[:,:,2],self.schedule,indices)-self.schedule['targets'][indices]
        end_jac=endpoint_jacobian(q,self.lengths);com,cj=batch_com_and_jacobian(q,self.lengths)
        lever=(self.schedule['targets'][indices]-com[:,None])/self.lengths['torso']
        jac[:,29,:11]=self.config['balance_weight']*(-cj[:,0]*force[:,:,1].sum(axis=1)[:,None]+cj[:,1]*force[:,:,0].sum(axis=1)[:,None])/self.lengths['torso']
        for i in range(4):
            scale=self.config['balance_weight']*known[:,i]
            jac[:,29,11+2*i]=-scale*lever[:,i,1];jac[:,29,12+2*i]=scale*lever[:,i,0]
            for a in range(2):
                for b in range(2):
                    row=40+4*i+2*a+b
                    jac[:,row,:11]=self.opt['force_contact_weight']*end_jac[:,i,a]*force[:,i,b,None]
                    jac[:,row,11+2*i+b]=self.opt['force_contact_weight']*delta[:,i,a]*known[:,i]
                jac[:,56+2*i+a,11+2*i+a]=self.opt['unload_force_weight']*self.unload[indices,i]*known[:,i]
        ts=np.arange(len(self.frames))[indices]
        for j,t in enumerate(ts):
            for k,r in enumerate(self.by_frame[t]):
                w=self.opt['reserve_weight' if r['kind']=='reserve' else 'unloading_weight']/self.lengths['torso']
                if com[j,0]<r['lower_cm']:jac[j,64+2*k,:11]=-w*cj[j,0]
                if com[j,0]>r['upper_cm']:jac[j,65+2*k,:11]=w*cj[j,0]
        return jac


def motion_limit_terms(x,config,weight):
    """Soft limits below the unchanged per-frame warning thresholds."""
    n=len(x);step=np.diff(x[:,:11],axis=0);hip=np.linalg.norm(step[:,:2],axis=1)
    hip_limit=config['hip_step_warning_cm'];angle_limit=np.deg2rad(config['angle_step_warning_deg'])
    r=np.zeros((n-1,10));r[:,0]=weight*np.maximum(0,hip/hip_limit-.95)
    r[:,1:]=weight*np.maximum(0,np.abs(step[:,2:])/angle_limit-.95)
    local=np.zeros((n-1,10,11))
    local[:,0,:2]=weight/hip_limit*step[:,:2]/np.maximum(hip[:,None],1e-12)*(hip[:,None]>.95*hip_limit)
    local[:,1+np.arange(9),2+np.arange(9)]=weight/angle_limit*np.sign(step[:,2:])*(np.abs(step[:,2:])>.95*angle_limit)
    rr=np.broadcast_to(np.arange((n-1)*10).reshape(n-1,10,1),(n-1,10,11)).ravel()
    cc=np.broadcast_to(np.arange(n-1)[:,None,None]*19+np.arange(11),(n-1,10,11)).ravel()
    data=local.ravel()
    j=coo_matrix((np.r_[-data,data],(np.r_[rr,rr],np.r_[cc,cc+19])),shape=((n-1)*10,n*19)).tocsr()
    j.eliminate_zeros()
    return r.ravel(),j


def optimize_support_sequence(initial, fps, audit_config=None, optimization=None, reference_outside=None, progress=print):
    c=optimization_settings(optimization);audit=audit_settings(audit_config)
    current=initial;history=[];constraints=[]
    for iteration in range(c['rounds']):
        p=SupportProblem(initial['reference'],initial['visibility'],initial['reference_endpoints'],initial['schedule'],
            initial['body'],fps,initial['mass_kg'],initial['foot_friction'],initial['max_force_component_n'],initial['settings'])
        p.configure(current['poses'],c,audit)
        n=len(p.frames);nr=p.residual_count;smooth=p.smooth_matrix()
        x0=np.c_[current['poses'],current['contact_forces_n'].reshape(n,8)/p.weight_n]
        row=np.broadcast_to(np.arange(n*nr).reshape(n,nr,1),(n,nr,19)).ravel()
        col=np.broadcast_to(np.arange(n)[:,None,None]*19+np.arange(19),(n,nr,19)).ravel()
        evaluations=[0]
        def residual(flat):
            evaluations[0]+=1
            if progress and evaluations[0]%50==0:progress(f'Support optimization round {iteration+1}/{c["rounds"]}: global evaluation {evaluations[0]}')
            x=flat.reshape(n,19)
            return np.r_[p.local(x,slice(None)).ravel(),smooth@flat,motion_limit_terms(x,p.config,c['motion_limit_weight'])[0]]
        def jacobian(flat):
            local=coo_matrix((p.local_jacobian(flat.reshape(n,19),slice(None)).ravel(),(row,col)),shape=(n*nr,n*19)).tocsr()
            local.eliminate_zeros();return vstack([local,smooth,motion_limit_terms(flat.reshape(n,19),p.config,c['motion_limit_weight'])[1]],format='csr')
        fit=least_squares(residual,np.clip(x0,p.lower,p.upper).ravel(),jac=jacobian,
            bounds=(np.tile(p.lower,n),np.tile(p.upper,n)),x_scale='jac',
            tr_options={'atol':1e-7,'btol':1e-7,'maxiter':500},max_nfev=c['max_nfev'],ftol=1e-6,xtol=1e-7,gtol=1e-6)
        if not np.isfinite(fit.x).all():raise RuntimeError('Nonfinite support optimization; no export.')
        x=fit.x.reshape(n,19);com,_=batch_com_and_jacobian(x[:,:11],p.lengths)
        current={**initial,'poses':x[:,:11],'joints':batch_joints(x[:,:11],p.lengths),
            'center_of_mass_cm':com,'contact_forces_n':x[:,11:].reshape(n,4,2)*p.schedule['known'][:,:,None]*p.weight_n,
            'optimizer':dict(success=bool(fit.success),message=str(fit.message),nfev=int(fit.nfev),cost=float(fit.cost),optimality=float(fit.optimality)),
            'method':'Global IRLS pose/force optimization with contact-force coupling, COM reserve intervals and prospective unloading; all source frames remain coupled.'}
        d=diagnose_gravity_sequence(current,reference_outside);a=audit_support_sequence(current,d,audit);attach_audit(d,a)
        record=dict(round=iteration+1,optimizer=current['optimizer'],counts=d['counts'],
            numerical_pass_frames=int(d['numerical_checks_pass'].sum()),max_contact_error_cm=float(d['max_contact_error_cm'].max()),
            max_hip_step_cm=float(d['hip_step_cm'].max()),max_angle_step_deg=float(d['angle_step_deg'].max()))
        history.append(record);constraints=p.constraints
        if progress:progress(str(record))
    current.update(optimization_settings=c,optimization_history=history,optimization_constraints=constraints)
    return current
