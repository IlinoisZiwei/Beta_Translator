import unittest
import numpy as np
from tests.test_full_sequence import fixture
from beta_translator.full_sequence import build_schedule
from beta_translator.gravity_sequence import solve_gravity_sequence
from beta_translator.support_optimization import optimization_settings
from beta_translator.support_audit import audit_settings
from beta_translator.technique_planner import (templates,TechniqueObjective,arm_moments,
    fair_load_allocation,rank_key,episodes,fit_candidate)


def example():
    q,b,e,table,g=fixture(8);s=build_schedule(table,np.arange(8),g)
    r=solve_gravity_sequence(q,np.ones((8,9)),e,s,b,30,settings={'max_nfev':50})
    r['frames']=r['frames']+200;r['schedule']['frames']=r['frames']
    r.update(optimization_settings=optimization_settings(),audit_settings=audit_settings())
    return r


class TechniqueTests(unittest.TestCase):
    def test_distal_moments_include_forearm_and_upper_arm_weight(self):
        j=np.zeros((1,4,3,2));j[0,:2]=[[[0,0],[20,0],[40,0]]]*2;f=np.zeros((1,4,2))
        m=arm_moments(j,f)
        self.assertAlmostEqual(m[0,0,1],-.022*.69*20)
        self.assertAlmostEqual(m[0,0,0],-.022*(20+.69*20)-.028*.436*20)
        f[0,0,1]=.2;new=arm_moments(j,f)
        self.assertAlmostEqual(new[0,0,1]-m[0,0,1],4.)
        self.assertAlmostEqual(new[0,0,0]-m[0,0,0],8.)

    def test_fair_force_allocation_equilibrium_and_no_phantom_support(self):
        targets=np.array([[-20,100],[20,100],[-10,0],[10,0]],float);j=np.zeros((4,3,2))
        r=fair_load_allocation(targets,[False,False,True,True],[0,50],j,1,.8,50)
        self.assertTrue(r['feasible']);np.testing.assert_allclose(r['forces_bw'].sum(0),[0,1],atol=1e-7)
        self.assertFalse(r['forces_bw'][:2].any());self.assertLess(r['hand_l1_bw'],1e-7)
        self.assertTrue(np.all(abs(r['forces_bw'][2:,0])<=.8*r['forces_bw'][2:,1]+1e-7))
        self.assertFalse(fair_load_allocation(targets,[False]*4,[0,50],j,1,.8,50)['feasible'])

    def test_hand_load_bound_is_positive_when_feet_cannot_balance(self):
        targets=np.array([[-20,100],[20,100],[-10,0],[10,0]],float)
        r=fair_load_allocation(targets,[True,True,False,False],[0,50],np.zeros((4,3,2)),1,.8,50)
        self.assertTrue(r['feasible']);self.assertAlmostEqual(r['hand_l1_bw'],1,places=6)

    def test_technique_temporal_jacobian_with_com_transform(self):
        r=example();g=templates(r);g['shift_mask'][:]=1;g['lift'][:]=1;g['flag_mask'][:]=1
        obj=TechniqueObjective(r,30,'combined',g);x=obj.base.initial_coordinates(r)
        x.reshape(-1,19)[:,11:]=[.12,.31,-.12,.2,.2,.12,-.2,.37]
        a,j=obj.evaluate(x);rng=np.random.default_rng(42)
        for _ in range(3):
            v=rng.normal(size=len(x));v/=np.linalg.norm(v)
            numeric=(obj.evaluate(x+1e-6*v)[0]-obj.evaluate(x-1e-6*v)[0])/2e-6
            np.testing.assert_allclose(j@v,numeric,rtol=2e-4,atol=6e-5)

    def test_foot_transfer_does_not_acquire_contact_force_or_label(self):
        r=example();s=r['schedule'];s['known'][2:6,2]=False;s['supported'][2:6,2]=False;s['labels'][2:6,2]='free'
        g=templates(r);obj=TechniqueObjective(r,30,'combined',g)
        self.assertTrue(g['flag_mask'][3:5,0].any());self.assertFalse(obj.lower[2:6,15:17].any());self.assertFalse(obj.upper[2:6,15:17].any())
        self.assertTrue(all(e['timing'].startswith('automatic') for e in episodes(s)))
        self.assertFalse(g['flag_mask'][s['known'][:,2:]].any())

    def test_ground_identity_and_preserved_approach_are_not_new_techniques(self):
        r=example();s=r['schedule'];r['frames']=s['frames']=np.arange(8)
        s['labels'][:,2]='free';s['known'][:,2]=False;s['supported'][:,2]=False
        s['known'][[0,7],2]=True
        s['support_ids']=s['labels'].astype('<U40').copy();s['support_ids'][[0,7],2]='ground_left'
        records=[e for e in episodes(s) if e['limb']==2]
        self.assertEqual(records[0]['from_hold'],'ground_left');self.assertTrue(records[0]['same_support_recontact'])
        g=templates(r);self.assertTrue(all('Preserved approach' in p for p in g['phase']))
        self.assertFalse(g['flag_mask'].any())

    def test_failure_or_nonconvergence_cannot_win_for_lower_load(self):
        m={k:0 for k in ['contact_residual','support_balance_failed','preunload_failed','rapid_hip_motion','rapid_angle_motion',
          'joint_limit','force_allocation_failed_frames','force_imbalance','moment_imbalance','foot_friction_violation','force_at_missed_contact',
          'uncertain_support_frames','low_margin_frames','preunload_uncertain_frames']}
        m.update(converged=True,hand_integral_l1_bw_s=1.,hand_peak_l1_bw=1.,arm_moment_integral_bw_cm_s=1.,max_angle_step_deg=2.)
        for failure in [{'contact_residual':1},{'converged':False},{'force_imbalance':1}]:
            worse={**m,**failure,'hand_integral_l1_bw_s':0.};self.assertLess(rank_key(m),rank_key(worse))
        missing={**m,'hand_integral_l1_bw_s':None};self.assertLess(rank_key(m),rank_key(missing))
        tiny={**m,'hand_integral_l1_bw_s':1.+1e-8};self.assertEqual(rank_key(m),rank_key(tiny))

    def test_edge_pose_constraints_and_fixed_bones_after_fitting(self):
        r=example();old=r['poses'].copy();labels=r['schedule']['labels'].copy()
        candidate=fit_candidate(r,30,'weight_shift',max_iterations=10,max_outer=1,progress=None)
        np.testing.assert_allclose(candidate['poses'][[0,1,-2,-1]],old[[0,1,-2,-1]],atol=1e-10)
        np.testing.assert_array_equal(labels,candidate['schedule']['labels'])
        for i in range(4):
            for k,name in enumerate(('upper_arm','forearm') if i<2 else ('thigh','shank')):
                np.testing.assert_allclose(np.linalg.norm(candidate['joints'][:,i,k+1]-candidate['joints'][:,i,k],axis=1),r['lengths'][name],atol=1e-10)


if __name__=='__main__':unittest.main()
