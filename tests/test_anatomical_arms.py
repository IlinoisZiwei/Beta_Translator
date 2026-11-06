import unittest
import numpy as np
from beta_translator import anatomical_arms as a
from beta_translator.inverse_kinematics import make_body
from beta_translator.trial2_anatomy_optimizer import rank_candidates


class AnatomyTests(unittest.TestCase):
    def setUp(self):
        lo,hi,_,_=a.bounds_and_rates();self.q=np.tile((lo+hi)/2,(7,1))
        self.q[:,:6]=0;self.lengths=make_body(180)

    def test_fixed_lengths_at_all_joint_limits(self):
        lo,hi,_,_=a.bounds_and_rates();rng=np.random.default_rng(31)
        q=rng.uniform(lo,hi,(70,28));g=a.kinematics(q,self.lengths,180)
        actual=np.linalg.norm(np.diff(g['joints'],axis=2),axis=-1)
        expected=np.array([[33.48,26.28]]*2+[[44.1,44.28]]*2)
        np.testing.assert_allclose(actual,np.broadcast_to(expected,actual.shape),atol=1e-10)

    def test_bilateral_mirror(self):
        g=a.kinematics(self.q,self.lengths,180)
        np.testing.assert_allclose(g['joints'][:,0]@a.MIRROR,g['joints'][:,1],atol=1e-12)
        np.testing.assert_allclose(g['palm_contacts'][:,0]@a.MIRROR,g['palm_contacts'][:,1],atol=1e-12)

    def test_wrist_changes_contact_but_not_elbow_or_wrist_position(self):
        before=a.kinematics(self.q,self.lengths,180);self.q[:,12]+=.3
        after=a.kinematics(self.q,self.lengths,180)
        np.testing.assert_allclose(before['joints'],after['joints'])
        self.assertGreater(np.linalg.norm(before['palm_contacts']-after['palm_contacts']),.1)

    def test_pronation_is_not_an_extra_elbow_swivel(self):
        before=a.kinematics(self.q,self.lengths,180);self.q[:,10]+=.8
        after=a.kinematics(self.q,self.lengths,180)
        np.testing.assert_allclose(before['joints'],after['joints'])
        self.assertGreater(np.linalg.norm(before['orientations']-after['orientations']),.1)

    def test_complex_derivative_and_proper_rotations(self):
        q=self.q.astype(complex);q[:,8]+=1e-25j
        analytic=a.kinematics(q,self.lengths,180)['joints'].imag/1e-25
        plus=self.q.copy();minus=self.q.copy();plus[:,8]+=1e-6;minus[:,8]-=1e-6
        fd=(a.kinematics(plus,self.lengths,180)['joints']-a.kinematics(minus,self.lengths,180)['joints'])/2e-6
        np.testing.assert_allclose(analytic,fd,atol=1e-7)
        R=a.kinematics(self.q,self.lengths,180)['orientations']
        np.testing.assert_allclose(np.linalg.det(R),1,atol=1e-12)

    def test_initialization_does_not_copy_an_impossible_elbow_branch(self):
        q=self.q[:3].copy();q[:,:3]=[0,100,30]
        q[:,6:13]=np.deg2rad([30,110,-40,100,0,0,0])
        q[:,13:20]=q[:,6:13]
        j=a.kinematics(q,self.lengths,180)['joints'].copy()
        S,E,W=j[:,0,0].copy(),j[:,0,1].copy(),j[:,0,2].copy()
        axis=W-S;axis/=np.linalg.norm(axis,axis=-1,keepdims=True)
        midpoint=S+np.sum((E-S)*axis,axis=-1,keepdims=True)*axis
        j[:,0,1]=2*midpoint-E  # same lengths/endpoints, wrong elbow side
        values=np.zeros((3,22));values[:,:6]=q[:,:6]
        initial=a.initialize(dict(joints=j,values=values),self.lengths,180)
        result=a.kinematics(initial,self.lengths,180)['joints']
        self.assertLess(np.linalg.norm(result[:,0,2]-W,axis=-1).max(),.1)

    def test_elbow_limit_is_signed_130_degrees(self):
        lo,hi,_,_=a.bounds_and_rates()
        for k in [9,16]:self.assertEqual(lo[k],0);self.assertAlmostEqual(np.rad2deg(hi[k]),130)

    def test_swivel_diagnostic_masks_straight_arm_singularity(self):
        j=a.kinematics(self.q,self.lengths,180)['joints']
        j[:,:2,1]=(j[:,:2,0]+j[:,:2,2])/2
        self.assertTrue(np.isnan(a.elbow_swivel_rates(j,30)).all())

    def test_rotation_rate_uses_seconds(self):
        r=a.axis_rotation(np.array([0,.1,.2]),np.array([0.,1,0]))[:,None]
        np.testing.assert_allclose(a.rotation_steps_deg(r,60),2*a.rotation_steps_deg(r,30))

    def test_feasible_gate_and_anatomy_precede_geometry(self):
        def row(name,anatomy,physical,feasible=False):
            return dict(candidate=name,numerically_feasible=feasible,anatomy_temporal_failure_frames=anatomy,
                worst_failure_fraction=physical,total_failed_frames=int(100*physical),grip_proxy_failure_frames=0,
                max_contact_gap_cm=physical,converged=feasible)
        rows=rank_candidates([row('geometry_only',10,0),row('bounded',0,.5)])
        self.assertEqual(rows[0]['candidate'],'bounded')
        self.assertEqual(rows[0]['selection_status'],'diagnostic_only_not_feasible')
        self.assertEqual(rank_candidates([row('no',0,0),row('yes',0,0,True)])[0]['candidate'],'yes')


if __name__=='__main__':unittest.main()
