"""Sparse SQP with in-step feasibility restoration for full control trajectories.

OSQP supplies approximate quadratic steps; a separate convex projection restores
hard bounds before nonlinear line search. This never modifies exported poses.
"""
import time
import numpy as np
import osqp
from scipy.sparse import diags,triu
from .constrained_sequence_solver import constraints,project_seed


def warm_qp(H,g,C,lo,hi,dual):
    # Large dense-in-time fatigue couplings make exact local steps expensive.
    # These are inexact SQP steps; native hard bounds are restored and the true
    # nonlinear objective must decrease before any step can be accepted.
    budget=1000 if len(g)>9000 else 3000
    model=osqp.OSQP()
    model.setup(P=triu(H,format='csc'),q=g,A=C.tocsc(),l=lo,u=hi,
        verbose=False,eps_abs=1e-5,eps_rel=1e-5,max_iter=budget,
        polishing=True,adaptive_rho=True,adaptive_rho_interval=50,check_termination=25)
    model.warm_start(x=np.zeros(len(g)),y=dual)
    r=model.solve(raise_error=False)
    if r.x is None or not np.isfinite(r.x).all():raise RuntimeError('Full QP failed: '+r.info.status)
    return r.x,r.y,dict(iterations=r.info.iter,iteration_budget=budget,primal=float(r.info.prim_res),dual=float(r.info.dual_res),success=r.info.status_val==1,status=r.info.status)


def solve(evaluate,x0,lower,upper,rate,max_iterations=250,progress=None):
    if max_iterations<1:raise ValueError('At least one iteration required')
    start=time.monotonic();x,initial_projection=project_seed(x0,lower,upper,rate)
    A,lo,hi=constraints(lower,upper,rate);r,J=evaluate(x);cost=float(r@r/2)
    damping=.01;history=[];last_change=np.inf;success=False;message='Maximum constrained iterations reached.'
    lam=np.zeros(A.shape[0]);optimality=np.inf;qp_info={};evaluations=1
    for iteration in range(max_iterations):
        gradient=np.asarray(J.T@r).ravel();scale=1/np.maximum(1,np.sqrt(np.asarray(J.power(2).sum(axis=0)).ravel()))
        JS=J@diags(scale);CS=A@diags(scale)
        rownorm=np.maximum(np.sqrt(np.asarray(CS.power(2).sum(axis=1)).ravel()),1e-12)
        C=diags(1/rownorm)@CS;ax=A@x
        H=(JS.T@JS+diags(np.full(len(x),damping))).tocsr()
        qp_start=time.monotonic()
        step,dual,qp_info=warm_qp(H,gradient*scale,C,(lo-ax)/rownorm,(hi-ax)/rownorm,lam*rownorm)
        qp_seconds=time.monotonic()-qp_start;projection_seconds=0.
        physical=scale*step
        # Eliminate numerical drift of exact fixed coordinates (zero forces).
        fixed=upper==lower;physical[fixed]=lower[fixed]-x[fixed]
        proposed=x+physical
        if max(float(np.max(lo-A@proposed)),float(np.max(A@proposed-hi)))>1e-7:
            # Feasibility restoration is INSIDE optimization. The subsequent
            # line search checks the actual nonlinear cost at the restored pose.
            projection_start=time.monotonic()
            proposed,_=project_seed(proposed,lower,upper,rate)
            projection_seconds=time.monotonic()-projection_start
            physical=proposed-x
        alpha=1.
        accepted=False
        for backtrack in range(16):
            candidate=x+alpha*physical;rr,jj=evaluate(candidate);evaluations+=1;new_cost=float(rr@rr/2)
            feasible=max(float(np.max(lo-A@candidate)),float(np.max(A@candidate-hi)))<=1e-7
            if feasible and np.isfinite(new_cost) and new_cost<cost:
                accepted=True;break
            alpha*=.5
        if accepted:
            last_change=(cost-new_cost)/max(1.,cost);x,r,J,cost=candidate,rr,jj,new_cost
            lam=dual/rownorm;damping=max(1e-6,damping*.7)
        else:
            damping*=10
        fresh=np.asarray(J.T@r).ravel();fresh_scale=1/np.maximum(1,np.sqrt(np.asarray(J.power(2).sum(axis=0)).ravel()))
        optimality=float(np.max(abs((fresh+A.T@lam)*fresh_scale)))
        violation=max(0.,float(np.max(lo-A@x)),float(np.max(A@x-hi)))
        slack=np.where(lam>=0,hi-A@x,A@x-lo)
        complementarity=float(np.max(abs(lam*slack)))/max(1.,cost)
        history.append(dict(qp_iteration_budget=qp_info['iteration_budget'],qp_seconds=qp_seconds,projection_seconds=projection_seconds,qp_iterations=qp_info['iterations'],complementarity=complementarity,step_fraction=alpha,iteration=iteration+1,cost=cost,optimality=optimality,relative_cost_change=last_change if np.isfinite(last_change) else None,
            qp_primal=qp_info['primal'],qp_converged=qp_info['success'],linear_violation=violation,accepted=accepted))
        if progress:progress(f'iteration {iteration+1}/{max_iterations}: cost {cost:.7g}, KKT {optimality:.3g}, hard-rate excess {violation:.2g}, QP {qp_seconds:.1f}s, elapsed {time.monotonic()-start:.1f}s')
        if optimality<=1e-5 and complementarity<=1e-6 and last_change<=1e-9 and violation<=1e-7 and qp_info['success']:
            success=True;message='Fresh constrained stationarity, cost stability and linear feasibility satisfied.';break
        if damping>1e10:message='Constrained steps stalled; convergence not established.';break
    return dict(x=x,success=success,message=message,nit=iteration+1,nfev=evaluations,cost=cost,
        optimality=optimality,relative_cost_change=last_change if np.isfinite(last_change) else None,
        linear_violation=violation,complementarity=complementarity,qp_iteration_budget=qp_info['iteration_budget'],history=history,initial_projection=initial_projection,seconds=time.monotonic()-start,
        gradient_definition='Fresh Lagrangian gradient normalized by current Jacobian columns; explicit box and linear rate multipliers.')
