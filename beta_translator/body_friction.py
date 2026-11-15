"""Shared unilateral foot-friction pyramid in body-weight force units."""
import numpy as np
from scipy.sparse import coo_matrix
from .body_constraints import DEFAULT_LIMITS


def friction_coefficients(normals, mu=DEFAULT_LIMITS.foot_friction):
    """Return five rows per normal; every row dotted with force must be <= 0."""
    normals=np.asarray(normals,float)
    if normals.shape[-1]!=3 or not np.isfinite(normals).all() or not np.allclose(np.linalg.norm(normals,axis=-1),1,atol=1e-10,rtol=0):
        raise ValueError('Friction normals must be finite unit vectors')
    if not np.isfinite(mu) or mu<0:raise ValueError('Nonnegative finite friction coefficient required')
    tangent=np.broadcast_to([1.,0.,0.],normals.shape).copy()
    tangent-=normals*np.sum(tangent*normals,axis=-1,keepdims=True)
    length=np.linalg.norm(tangent,axis=-1,keepdims=True)
    if np.any(length<1e-8):raise ValueError('Contact normal parallel to tangent seed')
    tangent/=length;other=np.cross(normals,tangent);slope=mu/np.sqrt(2)*normals
    return np.stack([-normals,-tangent-slope,tangent-slope,-other-slope,other-slope],axis=-2)


def friction_excess(forces, normals, mu=DEFAULT_LIMITS.foot_friction):
    """Signed inequality values, shared by the soft objective and final audit."""
    return np.einsum('...kj,...j->...k',friction_coefficients(normals,mu),forces)


def foot_friction_matrix(normals, qdim, statedim, variables, mu=DEFAULT_LIMITS.foot_friction):
    """Sparse rows on native frame force variables, excluding the hands."""
    coeff=friction_coefficients(np.asarray(normals)[:,2:],mu)
    n=coeff.shape[0]
    rows=np.broadcast_to(np.arange(n*10).reshape(n,2,5,1),coeff.shape)
    columns=(np.arange(n)[:,None,None,None]*statedim+qdim
             +np.arange(2,4)[None,:,None,None]*3+np.arange(3)[None,None,None,:])
    columns=np.broadcast_to(columns,coeff.shape)
    matrix=coo_matrix((coeff.ravel(),(rows.ravel(),columns.ravel())),shape=(n*10,variables)).tocsr()
    matrix.eliminate_zeros()
    return matrix
