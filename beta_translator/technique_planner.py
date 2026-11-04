"""Conditional technique candidates; teaching principles are not action labels.

Candidate poses share contact evidence, body, timing, audit thresholds and force
allocation rules. Technique terms are soft preferences, never feasibility gates.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from .inverse_kinematics import GRAVITY, MASS_FRACTIONS, CENTER_OF_MASS_FRACTIONS
from .motion_ik import batch_joints
from .gravity_sequence import batch_com_and_jacobian
from .contact_geometry import support_ids
from .support_refinement import (RefinementObjective, physical_coordinates,
    sparse_direct_least_squares, refinement_settings, stationarity)
from .support_audit import audit_support_sequence
from .trial1_ground_pipeline import ground_diagnose


PROFILES = {
    'control': dict(shift=0., lift=0., flag=0., straight=0., hand=0., torque=0.),
    'weight_shift': dict(shift=2., lift=0., flag=0., straight=0., hand=0., torque=0.),
    'leg_drive': dict(shift=2., lift=2., flag=0., straight=0., hand=0., torque=0.),
    'flag_assist': dict(shift=2., lift=2., flag=2., straight=0., hand=0., torque=0.),
    'arm_economy': dict(shift=2., lift=2., flag=0., straight=.6, hand=3., torque=2.),
    'combined': dict(shift=2., lift=2., flag=2., straight=.6, hand=3., torque=2.),
}


def episodes(schedule):
    """Free intervals between located contacts; retain automatic timing provenance."""
    rows=[];frames=schedule['frames'];known=schedule['known'];labels=schedule['labels'];ids=support_ids(schedule)
    for limb in range(4):
        free=(labels[:,limb]=='free') & ~known[:,limb]
        changes=np.diff(np.r_[False,free,False].astype(int))
        for lo,hi in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            if lo==0 or hi==len(frames) or not known[lo-1,limb] or not known[hi,limb]:continue
            rows.append(dict(limb=limb,release_index=int(lo),arrival_index=int(hi),
                release_frame=int(frames[lo]),arrival_frame=int(frames[hi]),
                from_hold=str(ids[lo-1,limb]),to_hold=str(ids[hi,limb]),
                same_support_recontact=bool(ids[lo-1,limb]==ids[hi,limb]),
                timing='automatic; not a reviewed technique annotation'))
    return sorted(rows,key=lambda r:r['release_frame'])


def templates(initial, first_active_frame=103):
    """Generate bounded, smooth posture guides without adding/changing contacts."""
    q=initial['poses'];s=initial['schedule'];frames=initial['frames'];n=len(q)
    com=initial['center_of_mass_cm'];j=initial['joints'];active=frames>=first_active_frame
    feet=s['known'][:,2:] & s['supported'][:,2:] & ~s['low_confidence'][:,2:]
    usable=active & feet.any(1)
    target_com=com[:,0].copy();foot_center=com[:,0].copy()
    for t in np.flatnonzero(usable):
        x=s['targets'][t,2:,0][feet[t]]
        # Project toward the feet's horizontal interval, not compulsorily above one foot.
        foot_center[t]=np.clip(com[t,0],x.min(),x.max())
        target_com[t]+=np.clip(foot_center[t]-com[t,0],-10.,10.)
    lift=np.zeros(n);flag=np.zeros((n,2));flag_target=j[:,2:,2,0].copy()
    phase=np.full(n,'Maintain / prepare next move',dtype='<U90')
    phase[~active]='Preserved approach / conditional ground support'
    for e in episodes(s):
        lo,hi,i=e['release_index'],e['arrival_index'],e['limb']
        before=max(0,lo-6);after=min(n,hi+7)
        phase[before:lo]='Transfer load / prepare release'
        phase[lo:hi]=('Unlocated interval / same-support recontact' if e['same_support_recontact'] else
            ('Reach for next handhold' if i<2 else 'Transfer foot to next foothold'))
        phase[hi:after]='Establish contact / stabilize'
        if i<2:
            ids=np.arange(before,after)
            wave=np.sin(np.pi*np.arange(len(ids))/max(1,len(ids)-1))**2
            lift[ids]=np.maximum(lift[ids],wave*usable[ids])
        else:
            ids=np.arange(lo,hi);wave=np.sin(np.pi*np.arange(len(ids))/max(1,len(ids)-1))**2
            other=3 if i==2 else 2
            ok=active[ids]&s['known'][ids,other]&s['supported'][ids,other]&~s['low_confidence'][ids,other]
            direction=np.sign(s['targets'][ids,other,0]-com[ids,0])
            direction=np.where(direction==0,-1 if i==2 else 1,direction)
            flag[ids,i-2]=wave*ok
            flag_target[ids,i-2]+=12.*direction*wave*ok
    # Automatic events must not relabel the frozen approach as a new technique.
    phase[~active]='Preserved approach / conditional ground support'
    return dict(active=active.astype(float),shift_mask=usable.astype(float),
        com_target=target_com,hip_target=q[:,1]+6.*lift,lift=lift,
        flag_mask=flag,flag_target=flag_target,phase=phase,events=episodes(s))


def arm_moments(joints, forces_bw):
    """Net quasi-static distal moments / BW, cm. Includes segment gravity.

    This omits muscle moment arms, co-contraction, fingers, 3D and fatigue.
    Forearm mass includes the hand, following the existing mass model.
    """
    j=joints[:,:2];f=forces_bw[:,:2];root,middle,end=j[:,:,0],j[:,:,1],j[:,:,2]
    upper=root+CENTER_OF_MASS_FRACTIONS['upper_arm']*(middle-root)
    lower=middle+CENTER_OF_MASS_FRACTIONS['forearm']*(end-middle)
    def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    elbow=cross(end-middle,f)-MASS_FRACTIONS['forearm']*(lower-middle)[...,0]
    shoulder=cross(end-root,f)-MASS_FRACTIONS['forearm']*(lower-root)[...,0]-MASS_FRACTIONS['upper_arm']*(upper-root)[...,0]
    return np.stack([shoulder,elbow],axis=-1)


class TechniqueObjective:
    def __init__(self,initial,fps,profile,guide,refinement=None,freeze_edges=2):
        self.base=RefinementObjective(initial,fps,initial['optimization_settings'],initial['audit_settings'],refinement)
        self.p=self.base.problem;self.n=len(initial['frames']);self.profile=dict(PROFILES[profile]);self.guide=guide
        self.lower=self.base.lower.copy();self.upper=self.base.upper.copy()
        coordinates=self.base.initial_coordinates(initial).reshape(self.n,19)
        fixed=(initial['frames']<103)
        fixed[:freeze_edges]=True;fixed[-freeze_edges:]=True
        self.lower[fixed,:11]=self.upper[fixed,:11]=coordinates[fixed,:11]
        self.count=self.extra(physical_coordinates(coordinates,self.p.lengths)[0]).shape[1]
        self.rows=np.broadcast_to(np.arange(self.n*self.count).reshape(self.n,self.count,1),(self.n,self.count,19)).ravel()
        self.cols=np.broadcast_to(np.arange(self.n)[:,None,None]*19+np.arange(19),(self.n,self.count,19)).ravel()

    def extra(self,x):
        p,g,w=self.p,self.guide,self.profile;q=x[:,:11];active=g['active']
        j=batch_joints(q,p.lengths);com=batch_com_and_jacobian(q,p.lengths)[0]
        forces=x[:,11:].reshape(self.n,4,2)*p.schedule['known'][:,:,None]
        shift=w['shift']*g['shift_mask']*(com[:,0]-g['com_target'])/10.
        lift=w['lift']*g['lift']*(q[:,1]-g['hip_target'])/6.
        flag=w['flag']*g['flag_mask']*(j[:,2:,2,0]-g['flag_target'])/12.
        straight=w['straight']*active[:,None]*p.schedule['known'][:,:2]*2*np.sin(q[:,[4,6]]/2)
        hand=w['hand']*active[:,None,None]*forces[:,:2]
        torque=w['torque']*active[:,None,None]*arm_moments(j,forces)/p.lengths['torso']
        return np.c_[shift,lift,flag,straight,hand.reshape(self.n,4),torque.reshape(self.n,4)]

    def evaluate(self,flat):
        r,j=self.base.evaluate(flat);x,tr=physical_coordinates(flat.reshape(self.n,19),self.p.lengths)
        extra=self.extra(x);ej=np.empty((self.n,self.count,19))
        for k in range(19):
            z=x.astype(complex);z[:,k]+=1e-25j;ej[:,:,k]=self.extra(z).imag/1e-25
        ej=ej@tr
        sparse=coo_matrix((ej.ravel(),(self.rows,self.cols)),shape=(self.n*self.count,self.n*19)).tocsr();sparse.eliminate_zeros()
        return np.r_[r,extra.ravel()],vstack([j,sparse],format='csr')


def fit_candidate(initial,fps,profile,guide=None,max_iterations=1600,max_outer=4,progress=print):
    guide=templates(initial) if guide is None else guide
    cfg=refinement_settings(dict(max_inner_iterations=max_iterations,max_outer_iterations=max_outer))
    current=initial;history=[];converged=False
    for outer in range(max_outer):
        obj=TechniqueObjective(current,fps,profile,guide,cfg)
        fit=sparse_direct_least_squares(obj.evaluate,obj.base.initial_coordinates(current),obj.lower.ravel(),obj.upper.ravel(),cfg)
        x,_=physical_coordinates(fit['x'].reshape(obj.n,19),obj.p.lengths)
        candidate={**initial,'poses':x[:,:11],'joints':batch_joints(x[:,:11],obj.p.lengths),
            'center_of_mass_cm':batch_com_and_jacobian(x[:,:11],obj.p.lengths)[0],
            'contact_forces_n':x[:,11:].reshape(obj.n,4,2)*obj.p.weight_n*initial['schedule']['known'][:,:,None]}
        fresh=TechniqueObjective(candidate,fps,profile,guide,cfg);values=fresh.base.initial_coordinates(candidate)
        rr,jj=fresh.evaluate(values);gradient=stationarity(values,rr,jj,fresh.lower.ravel(),fresh.upper.ravel())[-1]
        reweight=float(np.max(abs(fresh.p.contact_weight-obj.p.contact_weight)/np.maximum(1,abs(obj.p.contact_weight))))
        converged=bool(fit['success'] and gradient<=cfg['projected_gradient_tolerance'] and reweight<=cfg['reweight_tolerance'])
        history.append(dict(round=outer+1,inner={k:v for k,v in fit.items() if k!='x'},fresh_gradient=gradient,reweight=reweight))
        current=candidate
        if progress:progress(f'{profile} round {outer+1}: converged={converged}, gradient={gradient:.3g}, inner={fit["message"]}',flush=True)
        if converged:break
    current.update(optimizer=dict(success=converged,optimality=gradient,nfev=sum(r['inner']['nfev'] for r in history),
        message='Technique objective + refreshed IRLS stationarity passed.' if converged else 'Convergence not established; candidate is not eligible for recommendation.'),
        technique_profile=profile,technique_weights=dict(PROFILES[profile]),technique_history=history,
        singleton_com_constraints=obj.base.equalities)
    return current


def fair_load_allocation(targets,active,com,joints,limit_bw,mu,torso_cm):
    """Same two-stage LP for every candidate: min hand L1, then arm moments.

    All 8 force components, 4 hand absolute values and 4 moment absolute values.
    Uses the original capacity, friction and contact set; no phantom free-leg force.
    """
    active=np.asarray(active,bool);eq=np.zeros((3,16));eq[0,:8:2]=1;eq[1,1:8:2]=1
    lever=(targets-com)/torso_cm;eq[2,:8:2]=-lever[:,1];eq[2,1:8:2]=lever[:,0]
    bounds=[]
    for i in range(4):
        bounds.extend([(-limit_bw,limit_bw),(0 if i>=2 else -limit_bw,limit_bw)] if active[i] else [(0,0),(0,0)])
    bounds.extend([(0,None)]*8);rows=[];rhs=[]
    for k in range(4):
        for sign in (-1,1):
            a=np.zeros(16);a[k]=sign;a[8+k]=-1;rows.append(a);rhs.append(0.)
    for i in (2,3):
        for sign in (-1,1):
            a=np.zeros(16);a[2*i]=sign;a[2*i+1]=-mu;rows.append(a);rhs.append(0.)
    offset=arm_moments(joints[None],np.zeros((1,4,2)))[0].ravel()/torso_cm
    coeff=np.zeros((4,8))
    for k in range(8):
        f=np.zeros((1,4,2));f.reshape(-1)[k]=1
        coeff[:,k]=arm_moments(joints[None],f)[0].ravel()/torso_cm-offset
    for k in range(4):
        for sign in (-1,1):
            a=np.zeros(16);a[:8]=sign*coeff[k];a[12+k]=-1;rows.append(a);rhs.append(-sign*offset[k])
    a=np.array(rows);b=np.array(rhs);cost=np.zeros(16);cost[8:12]=1
    first=linprog(cost,A_ub=a,b_ub=b,A_eq=eq,b_eq=[0,1,0],bounds=bounds,method='highs')
    empty=dict(feasible=False,forces_bw=np.zeros((4,2)),hand_l1_bw=None,hand_l2_bw=None,arm_moment_bw_cm=None)
    if not first.success:return empty
    cap=cost.copy();cost2=np.zeros(16);cost2[12:]=1
    second=linprog(cost2,A_ub=np.vstack([a,cap]),b_ub=np.r_[b,first.fun+1e-8],A_eq=eq,b_eq=[0,1,0],bounds=bounds,method='highs')
    solution=second if second.success else first
    if np.max(abs(eq@solution.x-[0,1,0]))>1e-7 or np.max(a@solution.x-b)>1e-7:return empty
    f=solution.x[:8].reshape(4,2)
    return dict(feasible=True,forces_bw=f,hand_l1_bw=float(abs(f[:2]).sum()),
        hand_l2_bw=float(np.linalg.norm(f[:2],axis=1).sum()),arm_moment_bw_cm=float(abs(arm_moments(joints[None],f[None])).sum()),
        foot_vertical_bw=float(f[2:,1].sum()),equilibrium_residual=float(np.max(abs(eq@solution.x-[0,1,0]))))


def assess(result,fps,reference_outside=None):
    d=ground_diagnose(result,reference_outside,result['audit_settings']);a=d['support_audit'];rows=[];forces=[]
    for t,frame in enumerate(result['frames']):
        v=fair_load_allocation(result['schedule']['targets'][t],a['conditional'][t],result['center_of_mass_cm'][t],
            result['joints'][t],result['max_force_component_n']/(result['mass_kg']*GRAVITY),result['foot_friction'],result['lengths']['torso'])
        forces.append(v.pop('forces_bw'));rows.append(dict(frame=int(frame),**v))
    valid=np.array([r['feasible'] for r in rows]);loads=np.array([r['hand_l1_bw'] if r['feasible'] else np.nan for r in rows])
    moments=np.array([r['arm_moment_bw_cm'] if r['feasible'] else np.nan for r in rows])
    counts=d['counts'];flags=d['flags'];reliability=[r['balance_audit_status'] for r in a['frames']]
    hard_keys=['contact_residual','support_balance_failed','preunload_failed',
        'force_imbalance','moment_imbalance','foot_friction_violation','force_at_missed_contact']
    hard={k:int(np.asarray(flags.get(k,np.zeros(len(rows),bool))).sum()) for k in hard_keys}
    hard['rapid_hip_motion']=int((d['hip_step_cm']>result['settings']['hip_step_warning_cm']).sum())
    hard['rapid_angle_motion']=int((d['angle_step_deg']>result['settings']['angle_step_warning_deg']).sum())
    hard['joint_limit']=int(np.any(abs(result['poses'][:,[4,6,8,10]])>np.deg2rad(150)+1e-8,axis=1).sum())
    metrics=dict(profile=result.get('technique_profile','preserved_baseline'),converged=bool(result['optimizer']['success']),
        frames=len(rows),**hard,force_allocation_failed_frames=int((~valid).sum()),
        uncertain_support_frames=reliability.count('uncertain_support'),low_margin_frames=reliability.count('low_margin'),
        support_pass_frames=reliability.count('passes_model_checks'),
        preunload_uncertain_frames=counts.get('preunload_uncertain',0),
        max_contact_error_cm=float(d['max_contact_error_cm'].max()),max_hip_step_cm=float(d['hip_step_cm'].max()),
        max_angle_step_deg=float(d['angle_step_deg'].max()),
        hand_peak_l1_bw=float(np.nanmax(loads)) if valid.any() else None,
        hand_integral_l1_bw_s=float(loads.sum()/fps) if valid.all() else None,
        zero_hand_lower_bound_frames=int(np.sum(valid&(loads<1e-6))),
        arm_moment_integral_bw_cm_s=float(moments.sum()/fps) if valid.all() else None,
        all_geometry_uncertain=True,physical_validation=False)
    return metrics,rows,np.asarray(forces),d


def rank_key(metrics):
    """Feasibility, evidence, then load. Missing metrics can never win as zero."""
    hard=sum(metrics[k] for k in ['contact_residual','support_balance_failed','preunload_failed',
        'rapid_hip_motion','rapid_angle_motion','joint_limit','force_allocation_failed_frames',
        'force_imbalance','moment_imbalance','foot_friction_violation','force_at_missed_contact'])
    # Numerical ties are not evidence that one climbing technique is better.
    band=lambda k,width:round(metrics[k]/width) if metrics[k] is not None else float('inf')
    return (int(hard>0 or not metrics['converged']),hard,int(not metrics['converged']),
        metrics['uncertain_support_frames'],metrics['low_margin_frames'],metrics['preunload_uncertain_frames'],
        band('hand_integral_l1_bw_s',.005),band('hand_peak_l1_bw',.005),
        band('arm_moment_integral_bw_cm_s',.05),round(metrics['max_angle_step_deg'],2))
