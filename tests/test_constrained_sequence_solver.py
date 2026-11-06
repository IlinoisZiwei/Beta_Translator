import unittest
import numpy as np
from scipy.sparse import eye,diags,vstack
from beta_translator.constrained_sequence_solver import solve,project_seed

class ConstrainedSolverTests(unittest.TestCase):
    def test_velocity_bound_changes_optimum(self):
        target=np.array([0.,10.]);J=eye(2,format='csr')
        rate=diags([[-1.],[1.]],[0,1],shape=(1,2)).tocsr()
        result=solve(lambda x:(x-target,J),np.zeros(2),np.full(2,-20.),np.full(2,20.),rate,30)
        np.testing.assert_allclose(result['x'],[4.5,5.5],atol=1e-5)
        self.assertLessEqual(result['linear_violation'],1e-7)
        self.assertTrue(result['success'],result['message'])

    def test_fixed_coordinates_do_not_stop_other_coordinates(self):
        target=np.array([2.,10.]);J=eye(2,format='csr');rate=diags([np.ones(2)],[0])
        result=solve(lambda x:(x-target,J),np.zeros(2),np.array([0.,-20.]),np.array([0.,20.]),rate,30)
        np.testing.assert_allclose(result['x'],[0.,1.],atol=1e-6)

    def test_projection_enforces_both_rates_and_accelerations(self):
        n=100;D1=diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n))
        D2=diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n))
        rate=vstack([30*D1,900*D2],format='csr')
        x=np.sin(np.arange(n));p,info=project_seed(x,np.full(n,-2),np.full(n,2),rate)
        self.assertLessEqual(np.max(abs(rate@p)),1+1e-7)
        self.assertLessEqual(np.max(abs(p)),2+1e-7)

    def test_projection_keeps_absolute_tolerance_for_large_world_coordinates(self):
        n=78
        D1=diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n))
        D2=diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n))
        rate=vstack([30*D1,900*D2],format='csr')
        seed=400+25*np.sin(np.arange(n)/5)
        p,_=project_seed(seed,np.full(n,350),np.full(n,450),rate)
        self.assertLessEqual(np.max(abs(rate@p))-1,1e-7)
        self.assertTrue(np.all((p>=350-1e-7)&(p<=450+1e-7)))

if __name__=='__main__':unittest.main()
