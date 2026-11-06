"""Shared-chain regression and independent equation checks, not physiological validation."""
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from beta_translator import body_checks as c
from beta_translator.body_constraints import CONSTRAINTS,require_complete
from beta_translator.body_report import rank_candidates


class ConstraintChecks(unittest.TestCase):
    def test_unknown_cannot_be_accepted(self):
        rows=[dict(id=x.id,status='pass') for x in CONSTRAINTS]
        rows[-1]['status']='unknown';self.assertEqual(require_complete(rows),'unknown')
        with self.assertRaises(ValueError):require_complete(rows[:-1])
        self.assertEqual(c.torque_capacity()['status'],'unknown')

    def test_envelope_does_not_extrapolate(self):
        samples=[[0,0],[1,0],[0,1],[1,1]]
        self.assertEqual(c.shoulder_envelope([[.5,.5,0]],samples,[-1]*4,[1]*4,0)['status'],'pass')
        self.assertEqual(c.shoulder_envelope([[2,.5,0]],samples,[-1]*4,[1]*4,0)['status'],'unknown')
        self.assertEqual(c.shoulder_envelope([[.5,.5,2]],samples,[-1]*4,[1]*4,0)['status'],'fail')

    def test_so3_rates(self):
        R=Rotation.from_euler('z',np.array([[0],[.1],[.2]])).as_matrix()
        r=c.rotation_rates(R,[0,.1,.2],1.01,.001)
        self.assertEqual(r['status'],'pass');self.assertAlmostEqual(r['max_speed_rad_s'],1.)
        self.assertEqual(c.rotation_rates(R,[0,.1,.2],.9,.001)['status'],'fail')

    def test_collision_crossing_parallel_and_degenerate(self):
        self.assertAlmostEqual(c.segment_distance([-1,0,0],[1,0,0],[0,-1,0],[0,1,0]),0.)
        self.assertAlmostEqual(c.segment_distance([0,0,0],[1,0,0],[0,2,0],[1,2,0]),2.)
        self.assertAlmostEqual(c.segment_distance([0,0,0],[0,0,0],[0,2,0],[0,2,0]),2.)
        s=np.array([[[[-1,0,0],[1,0,0]],[[0,-1,0],[0,1,0]]]])
        self.assertEqual(c.self_collision(s,[.1,.1],[(0,1)],0)['status'],'fail')

    def test_directional_torque_and_fatigue_states(self):
        self.assertEqual(c.torque_capacity([-3],[-2],[5],0)['status'],'fail')
        r=c.fatigue([0,1,2],[0,0],.1,.05,[1,0,0]);states=np.array(r['states'])
        self.assertEqual(r['status'],'pass');np.testing.assert_allclose(states.sum(1),1)
        self.assertAlmostEqual(states[-1,0],np.exp(-.2))
        self.assertEqual(c.fatigue([0,1],[2],0,0,[0,1,0])['status'],'fail')

    def test_static_freefall_and_moment_dynamics(self):
        common=dict(mass_kg=10,com_m=[0,1,0],angular_momentum_rate_nm=[0,0,0],points_m=[[0,0,0]],moments_nm=[[0,0,0]],force_tolerance_n=1e-8,moment_tolerance_nm=1e-8)
        self.assertEqual(c.centroidal_dynamics(**common,com_acceleration_m_s2=[0,0,0],forces_n=[[0,98.1,0]])['status'],'pass')
        self.assertEqual(c.centroidal_dynamics(**common,com_acceleration_m_s2=[0,-9.81,0],forces_n=[[0,0,0]])['status'],'pass')
        common['points_m']=[[1,0,0]]
        self.assertEqual(c.centroidal_dynamics(**common,com_acceleration_m_s2=[0,0,0],forces_n=[[0,98.1,0]])['status'],'fail')

    def test_floating_root_not_actuated(self):
        S=np.zeros((1,7));S[0,6]=1
        kw=dict(mass_matrix=np.eye(7),bias=np.zeros(7),acceleration=np.r_[np.zeros(6),2.],actuation_map=S,torque=[2],contact_jacobian=np.zeros((3,7)),contact_wrench=np.zeros(3),passive=np.zeros(7),tolerance=1e-8)
        self.assertEqual(c.joint_dynamics(**kw)['status'],'pass')
        S[0,0]=1
        with self.assertRaises(ValueError):c.joint_dynamics(**kw)

    def test_passing_contact_noise_cannot_win_over_smoothness(self):
        a=dict(candidate='a',numerically_feasible=False,anatomy_temporal_failure_frames=0,worst_failure_fraction=.5,total_failed_frames=10,departing_force_failure_frames=0,unknown_constraints=18,converged=False,max_contact_gap_cm=.001,joint_acceleration_rms_cm_s2=100)
        b={**a,'candidate':'b','max_contact_gap_cm':.002,'joint_acceleration_rms_cm_s2':50}
        self.assertEqual(rank_candidates([a,b])[0]['candidate'],'b')


class TrialAdapters(unittest.TestCase):
    def test_pipeline_export_roundtrip(self):
        import tempfile
        from beta_translator.body_model import BodyModel
        from beta_translator.body_pipeline import verified_run
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            r=BodyModel(180).fit('trial1',project=root,frames=[113,115],iterations=1,
                candidates=('legacy_control','balanced'),output_root=temp,render=False,progress=lambda *a,**k:None)
            saved=verified_run(r['output_dir'])
            self.assertEqual(saved['frame_count'],3)
            self.assertFalse(saved['physical_validation'])
            self.assertEqual(len(saved['ranking']),1)
            self.assertEqual(saved['selected_candidate'],'balanced')
            self.assertEqual(saved['controls'][0]['candidate'],'legacy_control')
            self.assertFalse(saved['controls'][0]['eligible_for_selection'])
            cached=BodyModel(180).fit('trial1',project=root,frames=[113,115],iterations=1,
                candidates=('legacy_control','balanced'),output_root=temp,render=False,progress=lambda *a,**k:None)
            self.assertEqual(cached['output_dir'],r['output_dir'])

    def test_all_trial_schemas_and_shared_jacobians(self):
        from beta_translator.trials import load_trial,load_config
        from beta_translator.body_pipeline import sliced
        from beta_translator.body_legs import initialize
        from beta_translator.body_articulation import BodyProblem
        from beta_translator.inverse_kinematics import make_body
        root=Path(__file__).resolve().parents[1]
        for name,count in [('trial1',417),('trial2',976),('trial3',503)]:
            with self.subTest(trial=name):
                cfg=load_config(root,name);_,data,scene,regions,old=load_trial(root,cfg,180)
                self.assertEqual(len(data['frames']),count)
                self.assertEqual(data['observed_px'].shape,(count,4,3,2))
                ix=np.arange(cfg['verification_frames'][0],cfg['verification_frames'][0]+4)
                d=sliced(data,ix,count);legacy=sliced(old,ix,count);r={**regions,'mapping':regions['mapping'][ix]}
                q=initialize(legacy,make_body(180),180)
                problem=BodyProblem(d,scene,r,legacy,q,180)
                x=problem.initial.copy();rng=np.random.default_rng(91);x+=1e-4*rng.normal(size=len(x))
                direction=rng.normal(size=len(x));direction[problem.lower==problem.upper]=0
                residual,J=problem.evaluate(x);eps=2e-7
                fd=(problem.evaluate(x+eps*direction)[0]-problem.evaluate(x-eps*direction)[0])/(2*eps)
                np.testing.assert_allclose(J@direction,fd,atol=2e-3,rtol=3e-5)
                g=problem.geometry((problem.expand@problem.initial).reshape(4,problem.localdim))
                np.testing.assert_allclose(g['joints'][:,:,2]-g['targets'],g['contact_points']-g['contacts'],atol=1e-12)

if __name__=='__main__':unittest.main()
