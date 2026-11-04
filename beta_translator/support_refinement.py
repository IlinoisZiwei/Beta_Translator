"""Whole-sequence refinement with sparse-direct steps and explicit stationarity.

No audit threshold or source contact is changed. Singleton nominal support
intervals become exact COM-x constraints inside the parameterization. Their
zero robustness to COM perturbations remains visible in the independent audit.
"""
import numpy as np
from scipy.sparse import coo_matrix, vstack, diags, eye
from scipy.sparse.linalg import spsolve

from .support_optimization import SupportProblem, support_interval, motion_limit_terms, optimization_settings
from .support_audit import audit_settings
from .contact_geometry import contact_points
from .full_sequence import endpoint_jacobian
from .gravity_sequence import batch_com_and_jacobian
from .motion_ik import batch_joints

DEFAULT_REFINEMENT = dict(max_outer_iterations=6, max_inner_iterations=2000,
    projected_gradient_tolerance=1e-5, relative_cost_tolerance=1e-9,
    reweight_tolerance=1e-5, contact_restore_weight=100., contact_buffer_fraction=.8)


def refinement_settings(settings=None):
    c={**DEFAULT_REFINEMENT,**(settings or {})}
    if (set(c)!=set(DEFAULT_REFINEMENT) or any(not np.isfinite(v) or v<=0 for v in c.values())
            or any(type(c[k]) is not int for k in ('max_outer_iterations','max_inner_iterations'))
            or c['contact_buffer_fraction']>=1):
        raise ValueError('Invalid refinement settings.')
    return c


def stationarity(x, residual, jacobian, lower, upper):
    """KKT projected gradient, normalized by current residual-column norms."""
    gradient=np.asarray(jacobian.T@residual).ravel()
    norm=np.sqrt(np.asarray(jacobian.power(2).sum(axis=0)).ravel())
    scale=1/np.maximum(norm,1.)
    active=(lower==upper)|((x<=lower+1e-10)&(gradient>0))|((x>=upper-1e-10)&(gradient<0))
    projected=gradient*scale;projected[active]=0.
    return gradient,scale,active,float(np.max(np.abs(projected)))


def sparse_direct_least_squares(evaluate,x0,lower,upper,settings=None,progress=None):
    """Bounded damped Gauss-Newton with sparse-direct linear solves.

    Accept only actual cost-decreasing steps. Both projected first-order
    stationarity and relative cost stabilization are required for success.
    Iteration limits, failed factorization/descent and numerical stalls do not
    become successful termination just because the cost changes slowly.
    """
    c=refinement_settings(settings);x=np.clip(np.asarray(x0,float),lower,upper)
    lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    if x.shape!=lower.shape or x.shape!=upper.shape or np.any(lower>upper) or not np.isfinite(x).all():
        raise ValueError('Invalid starting point or bounds.')
    residual,jac=evaluate(x);evaluations=1;cost=float(residual@residual/2)
    if not np.isfinite(cost) or not np.isfinite(jac.data).all():raise ValueError('Nonfinite initial objective.')
    damping=1e-3;history=[];last_change=0.;success=False;message='Maximum inner iterations reached.'
    for iteration in range(c['max_inner_iterations']+1):
        gradient,scale,active,optimality=stationarity(x,residual,jac,lower,upper)
        if progress and iteration%50==0:
            progress(f'iteration {iteration}: cost {cost:.9g}, projected gradient {optimality:.3g}, relative cost change {last_change:.3g}')
        exact_stationary=np.max(abs(np.where(active,0.,gradient)))<=1e-12
        if optimality<=c['projected_gradient_tolerance'] and (last_change<=c['relative_cost_tolerance'] or exact_stationary):
            success=True
            message=('Exact projected stationarity.' if exact_stationary else
                     'Projected gradient and relative cost tolerances both satisfied.')
            break
        if iteration==c['max_inner_iterations']:break
        free=np.flatnonzero(~active);a=jac[:,free]@diags(scale[free]);hessian=(a.T@a).tocsc()
        accepted=False;attempts=0
        for attempts in range(24):
            step=spsolve(hessian+damping*eye(len(free),format='csc'),-gradient[free]*scale[free])
            candidate=x.copy();candidate[free]+=scale[free]*step;candidate=np.clip(candidate,lower,upper)
            displacement=candidate-x;linear=jac@displacement
            predicted=float(-gradient@displacement-linear@linear/2)
            if predicted<=0 or not np.isfinite(candidate).all():damping*=10;continue
            new_residual,new_jac=evaluate(candidate);evaluations+=1
            new_cost=float(new_residual@new_residual/2);ratio=(cost-new_cost)/predicted
            if np.isfinite(new_cost) and np.isfinite(new_jac.data).all() and ratio>1e-4 and new_cost<cost:
                last_change=(cost-new_cost)/max(1.,cost)
                history.append(dict(iteration=iteration+1,cost=new_cost,projected_gradient_before=optimality,
                    relative_cost_change=last_change,damping=damping,reduction_ratio=ratio,
                    max_coordinate_step=float(np.max(abs(displacement)))))
                x,residual,jac,cost=candidate,new_residual,new_jac,new_cost
                damping=max(1e-12,damping*(.3 if ratio>.75 else 3 if ratio<.25 else 1))
                accepted=True;break
            damping*=5
        if not accepted:
            message='No acceptable cost-decreasing step; stationarity not established.';break
    _,_,_,optimality=stationarity(x,residual,jac,lower,upper)
    return dict(x=x,success=success,message=message,nfev=evaluations,nit=iteration,cost=cost,
        optimality=optimality,relative_cost_change=last_change,history=history,
        gradient_definition='infinity norm of projected J.T r / max(1, column L2 norm of J)')


def com_coordinates(physical,lengths):
    values=np.asarray(physical,float).copy()
    values[:,0]=batch_com_and_jacobian(values[:,:11],lengths)[0][:,0]
    return values


def physical_coordinates(values,lengths):
    """COM-x replaces hip-x; return physical state and exact per-frame Jacobian."""
    values=np.asarray(values,float);physical=values.copy();physical[:,0]=0.
    com,cj=batch_com_and_jacobian(physical[:,:11],lengths)
    translation=cj[:,0,0]
    if np.any(abs(translation-1.)>1e-12):raise ValueError('COM translation must be affine with unit mass.')
    physical[:,0]=values[:,0]-com[:,0]
    transform=np.broadcast_to(np.eye(19),(len(values),19,19)).copy()
    transform[:,0,1:11]=-cj[:,0,1:]
    return physical,transform


def singleton_com_constraints(problem):
    """Enforce true nominal singleton intervals, never collapsed robust targets.

    A narrow interval cannot robustly cover +/- 2 cm, but it is not necessarily
    a point. Re-solve with zero reserve before identifying an exact equality.
    Conflicting equalities are recorded and left as soft objectives.
    """
    records=[];by_frame={};cache={}
    for row in problem.constraints:
        if row['lower_cm'] is None:continue
        t=row['index'];mask=np.zeros(4,bool);mask[row['limbs']]=True
        key=(tuple(problem.schedule['targets'][t].ravel()),tuple(mask))
        if key not in cache:cache[key]=support_interval(problem.schedule['targets'][t],mask,problem.limit_bw,problem.mu,0.)
        interval=cache[key]
        if interval['bounds'] is None:continue
        lo,hi=interval['bounds']
        if abs(hi-lo)>1e-8:continue
        record=dict(frame=int(problem.frames[t]),index=t,kind=row['kind'],group=row['group'],
            event_id=row['event_id'],nominal_com_x_cm=float((lo+hi)/2),limbs=row['limbs'],
            status='exact_nominal_singleton',robustness='Not robust to nonzero COM perturbation.')
        by_frame.setdefault(t,[]).append(record);records.append(record)
    fixed={}
    for t,rows in by_frame.items():
        values=[r['nominal_com_x_cm'] for r in rows]
        if max(values)-min(values)>1e-8:
            for row in rows:row['status']='conflicting_singletons_not_enforced'
        else:fixed[t]=float(np.mean(values))
    return fixed,records


class RefinementObjective:
    def __init__(self,result,fps,optimization=None,audit_config=None,refinement=None):
        self.refinement=refinement_settings(refinement)
        self.problem=SupportProblem(result['reference'],result['visibility'],result['reference_endpoints'],
            result['schedule'],result['body'],fps,result['mass_kg'],result['foot_friction'],
            result['max_force_component_n'],result['settings'])
        p=self.problem;p.configure(result['poses'],optimization,audit_config)
        self.n=len(p.frames);self.nr=p.residual_count+4;self.smooth=p.smooth_matrix()
        self.row=np.broadcast_to(np.arange(self.n*self.nr).reshape(self.n,self.nr,1),(self.n,self.nr,19)).ravel()
        self.column=np.broadcast_to(np.arange(self.n)[:,None,None]*19+np.arange(19),(self.n,self.nr,19)).ravel()
        self.tr=np.broadcast_to(np.arange(self.n*19).reshape(self.n,19,1),(self.n,19,19)).ravel()
        self.tc=np.broadcast_to(np.arange(self.n)[:,None,None]*19+np.arange(19),(self.n,19,19)).ravel()
        self.lower=np.tile(p.lower,(self.n,1));self.upper=np.tile(p.upper,(self.n,1))
        self.fixed,self.equalities=singleton_com_constraints(p)
        for t,value in self.fixed.items():self.lower[t,0]=self.upper[t,0]=value
        unknown=~p.schedule['known']
        self.lower[:,11:].reshape(self.n,4,2)[unknown]=0.
        self.upper[:,11:].reshape(self.n,4,2)[unknown]=0.

    def initial_coordinates(self,result):
        p=self.problem
        return np.clip(com_coordinates(np.c_[result['poses'],result['contact_forces_n'].reshape(self.n,8)/p.weight_n],p.lengths),self.lower,self.upper).ravel()

    def evaluate(self,flat):
        p=self.problem;cfg=self.refinement
        x,transform=physical_coordinates(flat.reshape(self.n,19),p.lengths)
        base=p.local(x,slice(None));bj=p.local_jacobian(x,slice(None))
        ends=contact_points(batch_joints(x[:,:11],p.lengths)[:,:,2],p.schedule)
        delta=ends-p.schedule['targets'];gap=np.linalg.norm(delta,axis=2)
        threshold=p.config['contact_tolerance_cm']*cfg['contact_buffer_fraction']
        active=p.schedule['known']&(gap>threshold)
        restore=cfg['contact_restore_weight']*np.maximum(0,gap-threshold)*p.schedule['known']
        ej=endpoint_jacobian(x[:,:11],p.lengths)
        rj=np.zeros((self.n,4,19));rj[:,:,:11]=cfg['contact_restore_weight']*np.einsum('nla,nlak->nlk',delta/np.maximum(gap[:,:,None],1e-12),ej)*active[:,:,None]
        local_jac=np.concatenate([bj,rj],axis=1)@transform
        local=coo_matrix((local_jac.ravel(),(self.row,self.column)),shape=(self.n*self.nr,self.n*19)).tocsr();local.eliminate_zeros()
        change=coo_matrix((transform.ravel(),(self.tr,self.tc)),shape=(self.n*19,self.n*19)).tocsr();change.eliminate_zeros()
        motion,mj=motion_limit_terms(x,p.config,p.opt['motion_limit_weight'])
        return np.r_[np.c_[base,restore].ravel(),self.smooth@x.ravel(),motion],vstack([local,self.smooth@change,mj@change],format='csr')


def refine_support_sequence(initial,fps,audit_config=None,optimization=None,refinement=None,progress=print):
    cfg=refinement_settings(refinement);opt=optimization_settings(optimization);audit=audit_settings(audit_config)
    current=initial;history=[];converged=False;total_nfev=0
    for outer in range(cfg['max_outer_iterations']):
        objective=RefinementObjective(current,fps,opt,audit,cfg);p=objective.problem
        if progress:progress(f'Global refinement {outer+1}/{cfg["max_outer_iterations"]}; {len(p.frames)} frames, {len(objective.fixed)} singleton COM constraints.')
        fit=sparse_direct_least_squares(objective.evaluate,objective.initial_coordinates(current),
            objective.lower.ravel(),objective.upper.ravel(),cfg,
            None if progress is None else lambda m:progress(f'Round {outer+1}: '+m))
        total_nfev+=fit['nfev']
        x,_=physical_coordinates(fit['x'].reshape(objective.n,19),p.lengths)
        if not np.isfinite(x).all():raise RuntimeError('Nonfinite refinement; no publication.')
        com,_=batch_com_and_jacobian(x[:,:11],p.lengths)
        candidate={**initial,'poses':x[:,:11],'joints':batch_joints(x[:,:11],p.lengths),
            'center_of_mass_cm':com,'contact_forces_n':x[:,11:].reshape(objective.n,4,2)*p.schedule['known'][:,:,None]*p.weight_n}
        # Check the actual reweighted objective too; convergence of only a frozen
        # IRLS surrogate does not establish outer fixed-point stationarity.
        refreshed=RefinementObjective(candidate,fps,opt,audit,cfg)
        a=p.contact_weight;b=refreshed.problem.contact_weight
        reweight=float(np.max(abs(b-a)/np.maximum(abs(a),1.)))
        new_x=refreshed.initial_coordinates(candidate);rr,jj=refreshed.evaluate(new_x)
        _,_,_,refreshed_gradient=stationarity(new_x,rr,jj,refreshed.lower.ravel(),refreshed.upper.ravel())
        equality_error=max((abs(com[t,0]-target) for t,target in objective.fixed.items()),default=0.)
        converged=bool(fit['success'] and reweight<=cfg['reweight_tolerance']
            and refreshed_gradient<=cfg['projected_gradient_tolerance'] and equality_error<=1e-8)
        record=dict(round=outer+1,inner={k:v for k,v in fit.items() if k!='x'},
            relative_weight_change=reweight,refreshed_projected_gradient=refreshed_gradient,
            max_singleton_com_error_cm=float(equality_error),outer_converged=converged)
        history.append(record);current=candidate
        if progress:progress(str({k:v for k,v in record.items() if k!='inner'}))
        if converged:break
    current.update(optimizer=dict(success=converged,
        message='Inner stationarity, cost stability, IRLS weight stability and refreshed stationarity satisfied.' if converged else 'Refinement limits reached; convergence not established.',
        nfev=total_nfev,cost=float(rr@rr/2),optimality=refreshed_gradient,
        gradient_definition=fit['gradient_definition'],inner_success=fit['success'],outer_iterations=len(history),
        relative_weight_change=reweight,max_singleton_com_error_cm=float(equality_error)),
        optimization_settings=opt,refinement_settings=cfg,optimization_history=history,
        optimization_constraints=p.constraints,singleton_com_constraints=objective.equalities,
        method='One full-sequence sparse-direct bounded pose/force optimization per IRLS iteration, in COM coordinates, with exact nominal singleton support constraints and soft contact restoration.')
    return current
