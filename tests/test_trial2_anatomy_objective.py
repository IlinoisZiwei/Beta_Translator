"""Regression against shared episode-variable and temporal Jacobian mistakes."""
import unittest
from pathlib import Path
import numpy as np
from beta_translator.trial2_anatomy_pipeline import LEGACY,slice_arrays
from beta_translator.trial2_spatial_pipeline import load_fit
from beta_translator.trial2_spatial_data import load_trial2,episode_regions
from beta_translator.trial2_scene3d import Trial2Scene
from beta_translator.trial2_anatomy_optimizer import AnatomicalProblem
from beta_translator import anatomical_arms as arms
from beta_translator.inverse_kinematics import make_body

class ObjectiveTests(unittest.TestCase):
    def test_whole_sparse_jacobian_direction_including_shared_contacts(self):
        root=Path(__file__).resolve().parents[1]
        if not (root/LEGACY/'trajectory.npz').exists():self.skipTest('Saved Trial2 baseline unavailable')
        data=load_trial2(root);old=load_fit(root/LEGACY);n=len(data['frames']);ix=np.arange(625,629)
        scene=Trial2Scene(root,15,0,15);regions=episode_regions(data,scene,180)
        d=slice_arrays(data,ix,n);legacy=slice_arrays(old,ix,n)
        q=arms.initialize(legacy,make_body(180),180)
        problem=AnatomicalProblem(d,scene,{**regions,'mapping':regions['mapping'][ix]},legacy,q)
        x=problem.initial.copy();rng=np.random.default_rng(19);direction=rng.normal(size=len(x))
        direction[problem.lower==problem.upper]=0
        # Stay off exactly active max() kinks without changing the tested equations.
        x+=1e-4*rng.normal(size=len(x))
        r,J=problem.evaluate(x);epsilon=2e-7
        fd=(problem.evaluate(x+epsilon*direction)[0]-problem.evaluate(x-epsilon*direction)[0])/(2*epsilon)
        np.testing.assert_allclose(J@direction,fd,atol=1e-3,rtol=2e-5)

if __name__=='__main__':unittest.main()
