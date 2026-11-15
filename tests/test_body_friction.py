"""Independent cone checks, native force indexing, and solver restoration."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix,eye
from beta_translator.body_friction import friction_coefficients,friction_excess,foot_friction_matrix
from beta_translator.constrained_sequence_solver import project_seed
from beta_translator.restored_sequence_solver import solve


class FootFrictionTests(unittest.TestCase):
    def test_matrix_matches_independent_rotated_cone_and_ignores_hands(self):
        rng=np.random.default_rng(5810);n=7;qdim=32;statedim=44
        angle=np.linspace(-.4,.6,n*4).reshape(n,4)
        normals=np.stack([np.zeros_like(angle),np.sin(angle),np.cos(angle)],-1)
        x=rng.normal(size=n*statedim+12);forces=x[:n*statedim].reshape(n,statedim)[:,qdim:].reshape(n,4,3)
        matrix=foot_friction_matrix(normals,qdim,statedim,len(x))
        normal=(forces[:,2:]*normals[:,2:]).sum(-1)
        t1=forces[:,2:,0];t2=forces[:,2:,1]*np.cos(angle[:,2:])-forces[:,2:,2]*np.sin(angle[:,2:])
        expected=np.maximum.reduce([-normal,abs(t1)-.8/np.sqrt(2)*normal,abs(t2)-.8/np.sqrt(2)*normal])
        actual=(matrix@x).reshape(n,2,5)
        np.testing.assert_allclose(actual.max(-1),expected,atol=1e-14)
        np.testing.assert_allclose(actual,friction_excess(forces[:,2:],normals[:,2:]),atol=1e-14)
        changed=x.copy();changed[:n*statedim].reshape(n,statedim)[:,:qdim+6]+=100
        changed[n*statedim:]+=100
        np.testing.assert_array_equal(matrix@changed,matrix@x)

    def test_projection_and_solver_enforce_one_sided_cone(self):
        matrix=csr_matrix(friction_coefficients([0,0,1]))
        linear=(matrix,np.full(5,-np.inf),np.zeros(5));rate=csr_matrix((0,3))
        target=np.array([1.,0.,.1]);lower=np.full(3,-2.);upper=np.full(3,2.)
        slope=.8/np.sqrt(2);normal=(slope+.1)/(1+slope*slope)
        expected=np.array([slope*normal,0,normal])
        projected,_=project_seed(target,lower,upper,rate,linear)
        np.testing.assert_allclose(projected,expected,atol=1e-7)
        result=solve(lambda x:(x-target,eye(3,format='csr')),np.zeros(3),lower,upper,rate,30,linear=linear)
        np.testing.assert_allclose(result['x'],expected,atol=1e-6)
        self.assertTrue(result['success'],result['message'])
        self.assertTrue(np.isfinite(result['complementarity']))
        self.assertLessEqual(np.max(matrix@result['x']),1e-7)

    def test_inexact_step_restoration_preserves_extra_constraints(self):
        matrix=csr_matrix([[1.,1.]])
        linear=(matrix,np.array([-np.inf]),np.array([1.]))
        evaluated=[]
        def evaluate(x):
            evaluated.append(x.copy())
            return x-[2.,2.],eye(2,format='csr')
        def approximate_step(H,g,C,lo,hi,dual):
            return np.full(2,10.),np.zeros(C.shape[0]),dict(iterations=1,iteration_budget=1,primal=10.,success=False)
        with patch('beta_translator.restored_sequence_solver.warm_qp',side_effect=approximate_step):
            result=solve(evaluate,np.zeros(2),np.zeros(2),np.full(2,5.),csr_matrix((0,2)),1,linear=linear)
        self.assertTrue(result['history'][0]['accepted'])
        self.assertGreater(result['history'][0]['projection_seconds'],0)
        self.assertTrue(all(np.sum(x)<=1+1e-7 for x in evaluated))
        np.testing.assert_allclose(result['x'],[.5,.5],atol=1e-7)

    def test_free_force_equalities_and_control_transform(self):
        from tests.test_personalized_body import IntegratedMechanics
        from beta_translator.temporal_control_basis import control_map
        problem=IntegratedMechanics.problem('trial3',n=9)
        T,z,lower,upper,_=control_map(problem,4)
        linear=problem.friction_constraints(T)
        z,_=project_seed(z,lower,upper,problem.hard_rates@T,linear)
        native=T@z;forces=native[:problem.n*problem.statedim].reshape(problem.n,problem.statedim)[:,problem.qdim:].reshape(-1,4,3)
        np.testing.assert_allclose(linear[0]@z,friction_excess(forces[:,2:],problem.base.normals[:,2:]).ravel(),atol=1e-12)
        self.assertLessEqual(np.max(linear[0]@z),1e-7)
        np.testing.assert_array_equal(forces[~problem.known],0)

    def test_soft_objective_keeps_original_friction_penalty(self):
        rng=np.random.default_rng(7);forces=rng.normal(size=(5,2,3))
        normal=np.broadcast_to([0.,-.25881904510252074,.9659258262890683],forces.shape)
        nf=(forces*normal).sum(-1);t=np.broadcast_to([1.,0.,0.],normal.shape)
        old=[np.maximum(-nf,0)]
        for tangent in (t,np.cross(normal,t)):
            for sign in [-1,1]:old.append(np.maximum(sign*(forces*tangent).sum(-1)-.8/np.sqrt(2)*nf,0))
        new=np.maximum(friction_excess(forces,normal),0).transpose(0,2,1).reshape(5,-1)
        np.testing.assert_allclose(new,np.concatenate(old,axis=1),atol=1e-14)


if __name__=='__main__':unittest.main()
