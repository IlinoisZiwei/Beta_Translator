"""Independent source agreement, signed anatomy and contact-chain regressions."""
import unittest
import numpy as np
from beta_translator import body_legs as legs
from beta_translator.body_leg_reference import verify_leg_chain
from beta_translator.body_constraints import LEG_LIMITS_DEG
from beta_translator.inverse_kinematics import make_body

class LegChainTests(unittest.TestCase):
    def setUp(self):self.lengths=make_body(175)

    def test_source_transforms_and_coupled_splines(self):
        r=verify_leg_chain()
        self.assertTrue(r['passed']);self.assertLess(r['position_max_error_m'],1e-12)
        self.assertFalse(r['personal_anatomical_validation'])

    def test_ankle_subtalar_move_foot_without_orbiting_knee(self):
        q=np.deg2rad([40,-20,10,60,0,0.]);a=legs.local_chain(q,self.lengths,175,'r')
        for column in [4,5]:
            b=q.copy();b[column]=.2;c=legs.local_chain(b,self.lengths,175,'r')
            np.testing.assert_allclose(c['knee'],a['knee']);np.testing.assert_allclose(c['ankle'],a['ankle'])
            self.assertGreater(np.max(abs(c['foot_frame']-a['foot_frame'])),.01)

    def test_signed_knee_and_parent_local_axis(self):
        a=legs.local_chain(np.zeros(6),self.lengths,175,'r')
        q=np.zeros(6);q[3]=np.pi/3;b=legs.local_chain(q,self.lengths,175,'r')
        self.assertGreater((b['ankle']-b['knee'])[2],0.) # positive flexion moves heel backwards
        q[:3]=[.4,-.2,.3];c=legs.local_chain(q,self.lengths,175,'r')
        rel=b['thigh_frame'].T@b['shank_frame']
        np.testing.assert_allclose(c['thigh_frame'].T@c['shank_frame'],rel,atol=1e-14)
        self.assertGreater(np.linalg.norm(b['knee']-b['femur_distal']),.01)
        lo,hi,_,_=legs.bounds_and_rates();self.assertEqual(lo[23],0.)
        self.assertAlmostEqual(hi[23],np.deg2rad(120))

    def test_bilateral_reflection_and_rigid_bones(self):
        rng=np.random.default_rng(5);q=np.zeros((10,32));lo,hi=np.deg2rad(LEG_LIMITS_DEG).T
        q[:,20:26]=rng.uniform(lo,hi,(10,6));q[:,26:32]=q[:,20:26]
        g=legs.kinematics(q,self.lengths,175)
        M=np.diag([-1,1,1])
        np.testing.assert_allclose(g['joints'][:,2]@M,g['joints'][:,3],atol=1e-13)
        actual=np.linalg.norm(np.diff(g['rigid_segments'],axis=3)[...,0,:],axis=-1)
        expected=np.array([[self.lengths['upper_arm'],self.lengths['forearm']]]*2+[[self.lengths['thigh'],self.lengths['shank']]]*2)
        np.testing.assert_allclose(actual,np.broadcast_to(expected,actual.shape),atol=1e-12)
        R=g['orientations'];np.testing.assert_allclose(np.linalg.det(R),1,atol=1e-13)

    def test_complex_derivative_through_knee_curve_and_shoe(self):
        q=np.zeros((1,32));q[:,20:]=np.deg2rad(np.tile([40,-20,10,70,10,5],2))
        rng=np.random.default_rng(3);v=rng.normal(size=q.shape)
        def f(a):
            g=legs.kinematics(a,self.lengths,175)
            foot=g['foot_origins']+np.einsum('tsij,j->tsi',g['foot_frames'],[15,0,-1.4])
            return np.r_[g['joints'].ravel(),foot.ravel(),g['com'].ravel()]
        cs=f(q+1e-25j*v).imag/1e-25;eps=1e-6
        np.testing.assert_allclose(cs,(f(q+eps*v)-f(q-eps*v))/(2*eps),rtol=2e-6,atol=1e-7)

class GroundPatchTests(unittest.TestCase):
    def test_ground_patches_follow_ankles_and_free_forces_stay_zero(self):
        from pathlib import Path
        from beta_translator.trials import load_trial
        from beta_translator.body_pipeline import sliced
        from beta_translator.body_articulation import BodyProblem
        from beta_translator.body_contacts import foot_frames
        root=Path(__file__).resolve().parents[1]
        for trial in ['trial1','trial2']:
            with self.subTest(trial=trial):
                _,d,scene,r,old=load_trial(root,trial,175)
                n=len(d['frames']);start=int(np.flatnonzero(d['ground'].any(1))[0]);ix=np.arange(start,start+4)
                d=sliced(d,ix,n);old=sliced(old,ix,n);r={**r,'mapping':r['mapping'][ix]}
                q=legs.initialize(old,make_body(175),175,foot_frames(d,scene,r,175)[0])
                p=BodyProblem(d,scene,r,old,q,175);y=(p.expand@p.initial).reshape(-1,p.localdim)
                a=p.geometry(y);changed=y.copy();changed[:,24]+=.05;b=p.geometry(changed)
                np.testing.assert_allclose(a['joints'][:,2],b['joints'][:,2],atol=1e-12)
                self.assertGreater(np.max(abs(a['foot_contacts'][:,0]-b['foot_contacts'][:,0])),.01)
                point=a['foot_origins']+np.einsum('tsij,tsj->tsi',a['foot_frames'],a['foot_patch_local_cm'])
                np.testing.assert_allclose(point,a['foot_contacts'],atol=1e-12)
                force=y[:,p.qdim:p.statedim].reshape(-1,4,3)
                np.testing.assert_array_equal(force[~d['known']],0.)

if __name__=='__main__':unittest.main()
