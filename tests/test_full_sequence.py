"""Whole-video constraints, uncertainty, and continuity regression checks."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

from beta_translator.inverse_kinematics import LIMBS, RATIOS, make_body
from beta_translator.motion_ik import batch_joints
from beta_translator.full_sequence import (
    build_schedule, endpoint_jacobian, solve_full_sequence, diagnose_sequence,
)


def fixture(count=12):
    q = np.tile([0., 100., .1, 1.7, .7, -1.2, -.7, .2, .8, -.2, -.8], (count, 1))
    body = dict(height_cm=180, ratios=dict(RATIOS))
    ends = batch_joints(q, make_body(**body))[:, :, 2]
    holds = {f'H{i}':ends[0, i].tolist() for i in range(4)}
    holds['H4'] = holds['H0']
    rows = [dict(frame=t, limb=l, candidate_hold=f'H{i}', confidence=.99)
            for t in range(count) for i,l in enumerate(LIMBS)]
    geometry = dict(supported_hold_ids=list(holds), hold_positions_cm=holds)
    return q, body, ends, pd.DataFrame(rows), geometry


class FullSequenceTests(unittest.TestCase):
    def test_multiple_free_limbs_wall_contact_and_extrapolation_remain_distinct(self):
        q, body, ends, table, geometry = fixture()
        table.loc[(table.frame == 4) & table.limb.isin(LIMBS[:2]), 'candidate_hold'] = 'free'
        table.loc[(table.frame == 5) & (table.limb == 'left_foot'), 'candidate_hold'] = 'wall_contact'
        geometry['supported_hold_ids'].remove('H3')
        s = build_schedule(table, np.arange(len(q)), geometry)
        self.assertFalse(s['known'][4, :2].any())
        self.assertEqual(s['states'][5, 2], 'wall_contact_location_unknown')
        self.assertTrue(s['known'][:, 3].all())
        self.assertFalse(s['supported'][:, 3].any())
        self.assertEqual(s['states'][4, 0], 'free_support_unresolved')

    def test_missing_duplicate_or_invalid_frames_rejected(self):
        q, _, _, table, geometry = fixture()
        for bad in (table.iloc[1:], pd.concat([table, table.iloc[:1]])):
            with self.assertRaises(ValueError):
                build_schedule(bad, np.arange(len(q)), geometry)
        with self.assertRaises(ValueError):
            build_schedule(table, np.arange(len(q))*2, geometry)
        table.loc[0, 'confidence'] = np.nan
        with self.assertRaises(ValueError):
            build_schedule(table, np.arange(len(q)), geometry)

    def test_analytic_endpoint_jacobian_matches_complex_step(self):
        q, body, _, _, _ = fixture(3)
        q[1] += np.arange(11) * .07
        lengths = make_body(**body)
        actual = endpoint_jacobian(q, lengths)
        for i in range(11):
            perturbed = q.astype(complex)
            perturbed[:, i] += 1e-25j
            expected = batch_joints(perturbed, lengths)[:, :, 2].imag / 1e-25
            np.testing.assert_allclose(actual[..., i], expected, atol=1e-10)

    def test_joint_fit_all_frames_changes_support_and_keeps_fixed_lengths(self):
        q, body, ends, table, geometry = fixture()
        table.loc[table.frame.between(3, 6) & table.limb.isin(LIMBS[:2]), 'candidate_hold'] = 'free'
        table.loc[(table.frame >= 7) & (table.limb == 'left_hand'), 'candidate_hold'] = 'H4'
        table.loc[(table.frame < 2) & (table.limb == 'right_foot'), 'candidate_hold'] = 'free'
        s = build_schedule(table, np.arange(len(q)), geometry)
        q[7:, 3] += 2*np.pi
        result = solve_full_sequence(q, np.ones((len(q), 9)), ends, s, body, 30)
        d = diagnose_sequence(result)
        self.assertTrue(result['optimizer']['success'])
        self.assertEqual(len(result['poses']), len(q))
        self.assertLess(d['max_segment_length_error_cm'], 1e-10)
        self.assertLess(d['max_contact_error_cm'].max(), .1)
        self.assertLess(d['angle_step_deg'].max(), 1.)
        self.assertTrue(d['flags']['unresolved_foot_support'][:2].all())
        self.assertFalse(d['flags']['unresolved_foot_support'][2:].any())

    def test_impossible_span_is_exportable_only_with_explicit_warnings(self):
        q, body, ends, table, geometry = fixture(5)
        geometry['hold_positions_cm']['H0'][0] += 350
        s = build_schedule(table, np.arange(len(q)), geometry)
        result = solve_full_sequence(q, np.ones((len(q), 9)), ends, s, body, 30,
                                     settings={'max_nfev':10})
        d = diagnose_sequence(result)
        self.assertTrue(d['flags']['span_conflict'].all())
        self.assertTrue(d['flags']['large_contact_residual'].all())
        self.assertTrue((d['status'] == 'constraint_warning').all())
        self.assertLess(d['max_segment_length_error_cm'], 1e-10)

    def test_optimizer_limit_never_becomes_a_clean_success(self):
        import beta_translator.full_sequence as module
        actual = module.least_squares
        def fail_final(fun, x0, **kwargs):
            if len(x0) > 11:
                return SimpleNamespace(x=x0, success=False, message='iteration limit',
                                       nfev=1, cost=0., optimality=1.)
            return actual(fun, x0, **kwargs)
        q, body, ends, table, geometry = fixture(4)
        s = build_schedule(table, np.arange(len(q)), geometry)
        with patch.object(module, 'least_squares', side_effect=fail_final):
            result = solve_full_sequence(q, np.ones((len(q), 9)), ends, s, body, 30)
        d = diagnose_sequence(result)
        self.assertTrue(d['flags']['optimizer_unconverged'].all())
        self.assertTrue((d['status'] == 'constraint_warning').all())


if __name__ == '__main__':
    unittest.main()
