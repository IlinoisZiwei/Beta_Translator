"""Support objectives must affect poses and retain honest failure semantics."""
import unittest
import numpy as np
from beta_translator.full_sequence import build_schedule
from beta_translator.gravity_sequence import solve_gravity_sequence,diagnose_gravity_sequence
from beta_translator.support_optimization import SupportProblem,support_interval,prospective_constraints,optimize_support_sequence,motion_limit_terms
from beta_translator.support_audit import audit_settings,force_reserve
from tests.test_full_sequence import fixture

class SupportOptimizationTests(unittest.TestCase):
    def test_interval_matches_analytic_feet_and_capacity(self):
        targets=np.array([[-20,0],[20,0],[-10,0],[10,0]],float)
        v=support_interval(targets,np.array([False,False,True,True]),1.,.8,.1)
        self.assertEqual(v['status'],'available')
        np.testing.assert_allclose(v['bounds'],[-8,8],atol=1e-8)
        for x in (-8,0,8):
            r=force_reserve(targets,np.array([False,False,True,True]),[x,50],100,100,.8,50)
            self.assertGreaterEqual(r['margin'],.1-1e-8)
        self.assertEqual(support_interval(targets,[True,False,False,False],1,.8,.1)['status'],'infeasible_capacity')
        self.assertEqual(support_interval(targets,[False]*4,1,.8,.1)['status'],'no_support')

    def test_analytic_augmented_jacobian(self):
        q,b,e,table,g=fixture(8);s=build_schedule(table,np.arange(8),g)
        s['labels'][6:,:2]='free';s['known'][6:,:2]=False;s['supported'][6:,:2]=False
        p=SupportProblem(q,np.ones((8,9)),e,s,b,30,70,.8,None,None);p.configure(q,None,None)
        x=np.c_[q+np.arange(11)*.01,np.tile([.1,.2,-.1,.3,.3,.1,-.2,.1],(8,1))]
        # Also exercise active COM-region derivatives away from hinge boundaries.
        for rows in p.by_frame:
            for r in rows:r['lower_cm']=300;r['upper_cm']=400
        j=p.local_jacobian(x,slice(None))
        for k in range(19):
            a=x.copy();b2=x.copy();a[:,k]+=1e-6;b2[:,k]-=1e-6
            expected=(p.local(a,slice(None))-p.local(b2,slice(None)))/2e-6
            np.testing.assert_allclose(j[:,:,k],expected,rtol=3e-5,atol=2e-5)

    def test_unload_removes_simultaneous_departures_without_future_contacts(self):
        q,b,e,table,g=fixture(10);s=build_schedule(table,np.arange(10),g)
        s['labels'][6:,:2]='free';s['known'][6:,:2]=False;s['supported'][6:,:2]=False
        rows,load=prospective_constraints(s,np.arange(10),1.,.8,audit_settings({'unload_lookahead_frames':3}))
        unloading=[r for r in rows if r['kind']=='unloading']
        self.assertEqual(set(r['index'] for r in unloading),{3,4,5})
        self.assertTrue(all(not {0,1}&set(r['limbs']) for r in unloading))
        np.testing.assert_allclose(load[3:6,:2],np.repeat([[1/3],[2/3],[1]],2,axis=1))
        self.assertFalse(load[6:].any())

    def test_full_sequence_optimization_changes_poses_and_keeps_fixed_lengths(self):
        q,b,e,table,g=fixture(6);s=build_schedule(table,np.arange(6),g)
        # Deliberately weak contact fit leaves a correctable endpoint error.
        baseline=solve_gravity_sequence(q+.05,np.ones((6,9)),e,s,b,30,settings={'contact_scale_cm':10.,'max_nfev':80})
        before=diagnose_gravity_sequence(baseline)
        r=optimize_support_sequence(baseline,30,optimization={'rounds':1,'max_nfev':100},progress=None)
        after=diagnose_gravity_sequence(r)
        self.assertGreater(np.max(abs(r['poses']-baseline['poses'])),1e-5)
        self.assertLess(after['max_contact_error_cm'].max(),before['max_contact_error_cm'].max())
        self.assertLess(after['max_segment_length_error_cm'],1e-10)
        np.testing.assert_array_equal(r['schedule']['labels'],baseline['schedule']['labels'])

    def test_temporal_limit_jacobian_and_zero_under_limits(self):
        x=np.zeros((4,19));x[1,0]=10;x[2,3]=.8
        cfg={'hip_step_warning_cm':5.,'angle_step_warning_deg':15.}
        r,j=motion_limit_terms(x,cfg,200)
        self.assertGreater(np.linalg.norm(r),0)
        for k in (0,19,41,63):
            a=x.copy().ravel();b=x.copy().ravel();a[k]+=1e-6;b[k]-=1e-6
            expect=(motion_limit_terms(a.reshape(4,19),cfg,200)[0]-motion_limit_terms(b.reshape(4,19),cfg,200)[0])/2e-6
            np.testing.assert_allclose(j[:,k].toarray().ravel(),expect,atol=2e-5)
        self.assertFalse(motion_limit_terms(np.zeros((4,19)),cfg,200)[0].any())

if __name__=='__main__':unittest.main()
