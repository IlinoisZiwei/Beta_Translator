import unittest
import numpy as np
from beta_translator.trial2_reach_diagnostics import target_reach_diagnostics

class ReachDiagnosticsTests(unittest.TestCase):
    def test_incompatible_targets_have_an_error_lower_bound(self):
        fit=dict(joints=np.zeros((2,4,3,3)),targets=np.zeros((2,4,3)),known=np.ones((2,4),bool),frames=np.arange(2))
        fit['targets'][1,0]=[0,500,0]
        r=target_reach_diagnostics(fit)
        self.assertFalse(r.target_reach_conflict.iloc[0]);self.assertTrue(r.target_reach_conflict.iloc[1])
        self.assertAlmostEqual(r.minimum_possible_max_endpoint_error_cm.iloc[1],r.target_span_excess_cm.iloc[1]/2)

    def test_unknown_targets_cannot_create_a_conflict(self):
        fit=dict(joints=np.zeros((1,4,3,3)),targets=np.full((1,4,3),1e6),known=np.zeros((1,4),bool))
        fit['targets'][0,0]=0
        self.assertFalse(target_reach_diagnostics(fit).target_reach_conflict.any())

if __name__=='__main__':unittest.main()
