"""Independent equilibrium, support eligibility and release-window regressions."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from beta_translator.support_audit import (force_reserve, reserve_sensitivity,
    classify_balance, audit_support_sequence, audit_settings, attach_audit)


def fixture(n=10):
    targets=np.tile([[-1.,0.],[1.,0.],[-1.,0.],[1.,0.]],(n,1,1))
    return dict(frames=np.arange(n),schedule=dict(targets=targets,
        known=np.ones((n,4),bool),supported=np.ones((n,4),bool),
        low_confidence=np.zeros((n,4),bool),labels=np.tile(['H0','H1','H2','H3'],(n,1)).astype('<U20')),
        center_of_mass_cm=np.tile([0.,1.],(n,1)),mass_kg=100/9.81,
        max_force_component_n=100.,foot_friction=.8,lengths={'torso':50.},
        settings={'contact_tolerance_cm':.1},contact_forces_n=np.zeros((n,4,2)))


class SupportAuditTests(unittest.TestCase):
    def force(self,active,com=(0.,1.),limit=100.,mu=.8):
        return force_reserve(fixture(1)['schedule']['targets'][0],active,np.array(com),100.,limit,mu,50.)

    def test_known_capacity_and_equilibrium(self):
        v=self.force([False,False,True,True])
        self.assertEqual(v['status'],'feasible'); self.assertAlmostEqual(v['margin'],.5)
        np.testing.assert_allclose(v['forces_n'].sum(axis=0),[0,100],atol=1e-10)
        np.testing.assert_allclose(v['forces_n'][:2],0)
        lever=fixture(1)['schedule']['targets'][0]-[0,1]
        self.assertAlmostEqual(np.sum(lever[:,0]*v['forces_n'][:,1]-lever[:,1]*v['forces_n'][:,0]),0.)

    def test_negative_capacity_is_not_structural_infeasibility(self):
        v=self.force([False,False,True,True],limit=40.)
        self.assertEqual(v['status'],'capacity_exceeded');self.assertAlmostEqual(v['margin'],-.25)
        impossible=self.force([False,False,True,False])
        self.assertEqual(impossible['status'],'structurally_infeasible')
        self.assertIsNone(impossible['margin'])

    def test_feet_cannot_pull_or_exceed_friction(self):
        v=self.force([False,False,True,True],com=(2,1))
        self.assertEqual(v['status'],'structurally_infeasible')
        v=self.force([True,False,False,True],com=(1.5,2),mu=.1)
        if v['margin'] is not None:
            self.assertGreaterEqual(v['forces_n'][3,1],-1e-8)
            self.assertLessEqual(abs(v['forces_n'][3,0]),.1*v['forces_n'][3,1]+1e-8)

    def test_missing_support_and_solver_failure_are_distinct(self):
        self.assertEqual(self.force([False]*4)['status'],'no_support')
        with patch('beta_translator.support_audit.linprog',return_value=SimpleNamespace(status=4,success=False,message='numerical failure')):
            self.assertEqual(self.force([True]*4)['status'],'not_evaluated')
        with patch('beta_translator.support_audit.linprog',return_value=SimpleNamespace(status=0,success=True,message='bad result',x=np.zeros(9))):
            self.assertEqual(self.force([True]*4)['status'],'not_evaluated')

    def test_contact_exclusions_do_not_count_missed_or_free_limbs(self):
        r=fixture(1);s=r['schedule']
        s['known'][0,0]=False;s['low_confidence'][0,2]=True;s['supported'][0,3]=False
        d={'contact_errors_cm':np.array([[np.nan,.2,0,0]])}
        a=audit_support_sequence(r,d,{'com_perturbation_cm':0.})
        np.testing.assert_array_equal(a['eligible'],[[False]*4])
        np.testing.assert_array_equal(a['conditional'],[[False,False,True,True]])
        self.assertEqual(a['contact_state'].tolist(),[['no_located_support','invalid_gap','uncertain_confidence','uncertain_geometry']])
        self.assertEqual(a['frames'][0]['balance_audit_status'],'uncertain_support')
        np.testing.assert_allclose(a['forces_n']['conditional'][0,:2],0)

    def test_horizontal_sensitivity_detects_failure(self):
        r=fixture(1);c=audit_settings({'com_perturbation_cm':2.})
        v=reserve_sensitivity(r,0,np.array([False,False,True,True]),c)
        self.assertEqual(v['nominal']['status'],'feasible');self.assertFalse(v['robust'])
        self.assertIsNone(v['worst_margin'])
        self.assertEqual(classify_balance(v,v,.1),'low_margin')

    def test_simultaneous_unload_uses_current_contacts_and_no_future_arrival(self):
        r=fixture();s=r['schedule'];s['labels'][6:,:2]='free';s['known'][6:,:2]=False
        # No feet before release; a future foot arrival must not rescue readiness.
        s['known'][:6,2:]=False;s['labels'][:6,2:]='free'
        a=audit_support_sequence(r,{'contact_errors_cm':np.zeros((10,4))},{'com_perturbation_cm':0.,'unload_lookahead_frames':3})
        self.assertEqual(len(a['events']),1)
        self.assertEqual(a['events'][0]['moving_indices'],[0,1])
        self.assertEqual([v['frame'] for v in a['unloading']],[3,4,5])
        self.assertTrue(all(v['remaining_conditional_count']==0 for v in a['unloading']))
        self.assertTrue(all(v['status']=='fails_located_support_model' for v in a['unloading']))
        self.assertEqual(a['frames'][2]['unload_audit_status'],'not_applicable')
        self.assertEqual(a['frames'][6]['unload_audit_status'],'not_applicable')

    def test_direct_jump_and_short_hold_dwell_remain_uncertain(self):
        r=fixture();s=r['schedule'];s['labels'][4:,0]='H9';s['labels'][6:,0]='H8'
        a=audit_support_sequence(r,{'contact_errors_cm':np.zeros((10,4))},{'com_perturbation_cm':0.})
        self.assertTrue(all(e['timing_uncertain'] for e in a['events']))
        second=[x for x in a['unloading'] if x['event_id']=='R002']
        self.assertEqual([v['frame'] for v in second],[4,5])
        self.assertTrue(all(v['status']=='uncertain_timing' for v in second))

    def test_new_failures_cannot_inherit_old_numeric_pass(self):
        r=fixture(1);s=r['schedule'];s['known'][:]=False
        a=audit_support_sequence(r,{'contact_errors_cm':np.zeros((1,4))})
        d=dict(numerical_checks_pass=np.ones(1,bool),flags={'low_pose_visibility':np.ones(1,bool)})
        attach_audit(d,a)
        self.assertTrue(d['legacy_numerical_checks_pass'][0])
        self.assertFalse(d['numerical_checks_pass'][0]);self.assertEqual(d['status'][0],'constraint_warning')

    def test_invalid_configuration_rejected(self):
        for c in ({'margin_warning':1.},{'com_perturbation_cm':float('nan')}, {'unload_lookahead_frames':0},{'new':1}):
            with self.assertRaises(ValueError):audit_settings(c)


if __name__=='__main__':unittest.main()
