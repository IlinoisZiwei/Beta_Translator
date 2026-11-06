"""Sparse Gauss-Newton with box AND linear rate constraints.

Each local convex QP uses OSQP (scaled ADMM). A feasible line search verifies the real
nonlinear objective; failed/stalled QPs are not reported as convergence. Final
diagnostics include primal feasibility and fresh Lagrangian stationarity.
"""
import time
import numpy as np
from scipy.sparse import eye,diags,vstack


def qp(H,g,C,lo,hi,max_iterations=12000,tolerance=1e-8):
    import osqp
    from scipy.sparse import triu
    model=osqp.OSQP()
    model.setup(P=triu(H,format='csc'),q=g,A=C.tocsc(),l=lo,u=hi,
        verbose=False,eps_abs=tolerance,eps_rel=tolerance,
        max_iter=max_iterations,polishing=True,adaptive_rho=True,
        adaptive_rho_interval=50,check_termination=25)
    result=model.solve(raise_error=False)
    if result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError('Constraint QP failed: '+result.info.status)
    return result.x,result.y,dict(iterations=result.info.iter,
        primal=float(result.info.prim_res),dual=float(result.info.dual_res),
        success=result.info.status_val==1,status=result.info.status)


def constraints(lower,upper,rate):
    A=vstack([eye(len(lower),format='csr'),rate],format='csr')
    return A,np.r_[lower,-np.ones(rate.shape[0])],np.r_[upper,np.ones(rate.shape[0])]


def project_seed(x,lower,upper,rate):
    A,lo,hi=constraints(lower,upper,rate)
    change,_,info=qp(eye(len(x),format='csr'),np.zeros(len(x)),A,lo-A@x,hi-A@x,max_iterations=40000,tolerance=1e-9)
    candidate=x+change
    violation=max(float(np.max(lo-A@candidate)),float(np.max(A@candidate-hi)))
    # Large coordinates can leave a small cancellation residual after the QP.
    # Refine the residual in physical coordinates; keep the acceptance limit.
    for _ in range(3):
        if violation<=1e-7:break
        correction,_,info=qp(eye(len(x),format='csr'),np.zeros(len(x)),A,
                            lo-A@candidate,hi-A@candidate,
                            max_iterations=40000,tolerance=1e-10)
        candidate+=correction
        violation=max(float(np.max(lo-A@candidate)),float(np.max(A@candidate-hi)))
    if violation>1e-7:raise RuntimeError(f'Initial rate projection failed ({violation:g}); no unconstrained fallback.')
    return candidate,info


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
        step,dual,qp_info=qp(H,gradient*scale,C,(lo-ax)/rownorm,(hi-ax)/rownorm)
        physical=scale*step
        # Eliminate numerical drift of exact fixed coordinates (zero forces).
        fixed=upper==lower;physical[fixed]=lower[fixed]-x[fixed]
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
        history.append(dict(iteration=iteration+1,cost=cost,optimality=optimality,relative_cost_change=last_change if np.isfinite(last_change) else None,
            qp_primal=qp_info['primal'],qp_converged=qp_info['success'],linear_violation=violation,accepted=accepted))
        if progress and iteration%25==0:progress(f'iteration {iteration}: cost {cost:.7g}, KKT {optimality:.3g}, hard-rate excess {violation:.2g}')
        if optimality<=1e-5 and last_change<=1e-9 and violation<=1e-7 and qp_info['success']:
            success=True;message='Fresh constrained stationarity, cost stability and linear feasibility satisfied.';break
        if damping>1e10:message='Constrained steps stalled; convergence not established.';break
    return dict(x=x,success=success,message=message,nit=iteration+1,nfev=evaluations,cost=cost,
        optimality=optimality,relative_cost_change=last_change if np.isfinite(last_change) else None,
        linear_violation=violation,history=history,initial_projection=initial_projection,seconds=time.monotonic()-start,
        gradient_definition='Fresh Lagrangian gradient normalized by current Jacobian columns; explicit box and linear rate multipliers.')
