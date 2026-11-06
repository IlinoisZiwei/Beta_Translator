import unittest
from types import SimpleNamespace
import numpy as np
from scipy.sparse import diags,vstack,eye
from beta_translator.temporal_control_basis import control_map
from beta_translator.anatomical_arms import bounds_and_rates
from beta_translator.restored_sequence_solver import solve

class ControlTests(unittest.TestCase):
    def test_controls_preserve_rom_and_remove_unobserved_forces(self):
        n=61;lo,hi,_,_=bounds_and_rates();lower=np.tile(np.r_[lo,[-1.]*12],n);upper=np.tile(np.r_[hi,[1.]*12],n)
        ids=np.arange(n)*40+28;lower[ids]=upper[ids]=0
        initial=(lower+upper)/2
        p=SimpleNamespace(n=n,lower=lower,upper=upper,initial=initial)
        T,z,l,u,info=control_map(p,8);rng=np.random.default_rng(10);v=rng.uniform(l,u);native=T@v
        self.assertTrue(np.all(native>=lower-1e-10));self.assertTrue(np.all(native<=upper+1e-10))
        np.testing.assert_allclose(native[ids],0)
        np.testing.assert_allclose(T@z,initial,atol=1e-7)
        np.testing.assert_allclose(np.asarray(T.sum(axis=1)).ravel()[np.arange(n)[:,None]*40+np.arange(28)],1)

    def test_restored_solver_feasibility_and_known_optimum(self):
        target=np.array([0.,10.]);J=eye(2,format='csr')
        rate=diags([[-1.],[1.]],[0,1],shape=(1,2)).tocsr()
        r=solve(lambda x:(x-target,J),np.zeros(2),np.full(2,-20.),np.full(2,20.),rate,30)
        np.testing.assert_allclose(r['x'],[4.5,5.5],atol=1e-5)
        self.assertLessEqual(r['linear_violation'],1e-7);self.assertTrue(r['success'])
        self.assertLessEqual(r['complementarity'],1e-6)

if __name__=='__main__':unittest.main()
