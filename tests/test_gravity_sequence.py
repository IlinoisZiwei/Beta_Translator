"""Exact derivatives, changing supports, fixed lengths and explicit failure flags."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from beta_translator.inverse_kinematics import make_body, make_mass_model, center_of_mass
from beta_translator.full_sequence import build_schedule
from beta_translator.gravity_sequence import (
    GravityProblem, batch_com_and_jacobian, solve_gravity_sequence, diagnose_gravity_sequence,
)
from tests.test_full_sequence import fixture


class GravitySequenceTests(unittest.TestCase):
    def problem(self, n=5, **kwargs):
        q, body, ends, table, geometry = fixture(n)
        schedule = build_schedule(table,np.arange(n),geometry)
        p = GravityProblem(q,np.ones((n,9)),ends,schedule,body,30,70,.8,None,kwargs)
        return p,q,body,ends,schedule

    def test_com_and_analytic_jacobian_match_static_model_and_complex_step(self):
        p,q,body,_,_ = self.problem()
        lengths = make_body(**body)
        q += np.arange(11)[None] * .03
        com,jac = batch_com_and_jacobian(q,lengths)
        expected = np.array([center_of_mass(x,lengths,make_mass_model(70)) for x in q])
        np.testing.assert_allclose(com,expected,atol=1e-12)
        for j in range(11):
            perturbed = q.astype(complex);perturbed[:,j] += 1e-25j
            actual = batch_com_and_jacobian(perturbed,lengths)[0].imag / 1e-25
            np.testing.assert_allclose(jac[:,:,j],actual,atol=1e-10)

    def test_entire_local_jacobian_matches_finite_difference(self):
        p,q,_,_,schedule = self.problem()
        schedule['known'][0,0] = False
        x = np.c_[q,np.tile([.1,.2,-.1,.3,.3,.1,-.2,.1],(len(q),1))]
        jac = p.local_jacobian(x,slice(None))
        for j in range(19):
            a,b = x.copy(),x.copy();a[:,j]+=1e-6;b[:,j]-=1e-6
            expected = (p.local(a,slice(None))-p.local(b,slice(None)))/2e-6
            np.testing.assert_allclose(jac[:,:,j],expected,rtol=2e-5,atol=2e-6)

    def test_global_fit_variable_contacts_and_force_masks(self):
        p,q,body,ends,schedule = self.problem(8)
        schedule['known'][2:4,:2] = False
        schedule['supported'][2:4,:2] = False
        schedule['labels'][2:4,:2] = 'free'
        schedule['states'][2:4,:2] = 'free_support_unresolved'
        r = solve_gravity_sequence(q,np.ones((8,9)),ends,schedule,body,30,
                                   settings={'max_nfev':200})
        d = diagnose_gravity_sequence(r)
        self.assertTrue(r['optimizer']['success'])
        self.assertEqual(r['poses'].shape,(8,11))
        np.testing.assert_array_equal(r['contact_forces_n'][2:4,:2],0)
        self.assertLess(d['max_segment_length_error_cm'],1e-10)
        self.assertTrue(np.isfinite(r['contact_forces_n']).all())

    def test_missing_support_and_insufficient_force_are_not_successes(self):
        p,q,body,ends,schedule = self.problem()
        schedule['known'][2] = False;schedule['supported'][2] = False
        schedule['labels'][2] = 'free';schedule['states'][2] = 'free_support_unresolved'
        r = solve_gravity_sequence(q,np.ones((5,9)),ends,schedule,body,30,
            max_force_n=100,settings={'max_nfev':40})
        d = diagnose_gravity_sequence(r)
        self.assertTrue(d['flags']['force_imbalance'].all())
        self.assertTrue(d['flags']['no_known_support'][2])
        np.testing.assert_array_equal(r['contact_forces_n'][2],0)
        self.assertFalse(d['numerical_checks_pass'].any())
        self.assertTrue((d['status']=='constraint_warning').all())

    def test_nonconvergence_and_missed_contacts_remain_flagged(self):
        import beta_translator.gravity_sequence as module
        p,q,body,ends,schedule = self.problem()
        schedule['targets'][:,0,0] += 400
        actual = module.least_squares
        def stop_global(fun,x0,**kwargs):
            if len(x0)>19:
                return SimpleNamespace(x=x0,success=False,message='iteration limit',nfev=1,cost=1.,optimality=1.)
            return actual(fun,x0,**kwargs)
        with patch.object(module,'least_squares',side_effect=stop_global):
            r = solve_gravity_sequence(q,np.ones((5,9)),ends,schedule,body,30)
        d = diagnose_gravity_sequence(r)
        self.assertTrue(d['flags']['optimizer_unconverged'].all())
        self.assertTrue(d['flags']['span_conflict'].all())
        self.assertTrue(d['flags']['force_at_missed_contact'].any())
        self.assertFalse(d['numerical_checks_pass'].any())

    def test_height_configuration_changes_fixed_lengths(self):
        p,q,body,ends,schedule = self.problem()
        body = {**body,'height_cm':165}
        r = solve_gravity_sequence(q,np.ones((5,9)),ends,schedule,body,30,
                                   settings={'max_nfev':30})
        self.assertEqual(r['body']['height_cm'],165)
        for i in range(4):
            name = 'upper_arm' if i<2 else 'thigh'
            actual = np.linalg.norm(r['joints'][:,i,1]-r['joints'][:,i,0],axis=1)
            np.testing.assert_allclose(actual,165*body['ratios'][name],atol=1e-10)


if __name__=='__main__':
    unittest.main()
