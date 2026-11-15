"""Performance paths must preserve the objective and constrained solution."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from beta_translator.restored_sequence_solver import solve
from tests import test_personalized_body as fixtures

class PreviewPerformance(unittest.TestCase):
    def test_residual_matches_full_objective_for_all_adapters(self):
        for trial in ('trial1','trial2','trial3'):
            p=fixtures.IntegratedMechanics.problem(trial,n=5)
            rng=np.random.default_rng(5120)
            for advanced in (True,False):
                p.dynamic_balance=advanced
                for x in (p.initial,p.initial+.0002*rng.normal(size=len(p.initial))):
                    full,J=p.evaluate(x)
                    np.testing.assert_array_equal(p.residual(x),full)
                    repeated,K=p.evaluate(x)
                    np.testing.assert_array_equal(repeated,full)
                    np.testing.assert_array_equal(K.toarray(),J.toarray())

    def test_lazy_derivatives_preserve_backtracking_result(self):
        def residual(x):return np.array([x[0]**2-4])
        def evaluate(x):return residual(x),csr_matrix([[2*x[0]]])
        args=(evaluate,np.array([.1]),np.array([.01]),np.array([10.]),csr_matrix((0,1)),8)
        old=solve(*args);new=solve(*args,residual=residual)
        np.testing.assert_array_equal(old['x'],new['x'])
        self.assertEqual(old['cost'],new['cost'])
        self.assertLess(new['derivative_evaluations'],old['derivative_evaluations'])

if __name__=='__main__':unittest.main()
