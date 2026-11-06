"""Mass conservation, coordinate invariance, independent formulas and derivatives."""
import json
import unittest

import numpy as np

from beta_translator.inverse_kinematics import (
    ANTHROPOMETRY_PROFILES, CENTER_OF_MASS_FRACTIONS,
    anthropometry_profile, combined_distal_center_of_mass, weighted_center_of_mass,
    make_body, make_mass_model, center_of_mass, forward_kinematics,
    solve_inverse_kinematics_with_gravity, validate_com_fractions,
)
from beta_translator.gravity_sequence import batch_com_and_jacobian
from beta_translator.ik3d import kinematics, rotation


class AnthropometryTests(unittest.TestCase):
    def setUp(self):
        self.body = make_body(180)
        self.q = np.array([4., 100., .08, 2.7, .4, 3.5, -.6, .3, .7, -.2, -.5])

    def spatial_pose(self, q):
        out = np.zeros((1, 22))
        out[0, :2] = q[:2]
        out[0, 3] = q[2]
        angles = q[3:].reshape(4, 2)
        out[0, 6::4] = angles[:, 0]
        out[0, 8::4] = angles.sum(axis=-1)
        return out

    def test_profile_mass_conservation_and_explicit_evidence(self):
        for name in ANTHROPOMETRY_PROFILES:
            p = anthropometry_profile(name)
            json.dumps(p, allow_nan=False)
            self.assertAlmostEqual(sum(p['mass_fractions'][k] * n
                                       for k, n in p['segment_counts'].items()), 1.)
            self.assertFalse(p['validated_for_individual'])
            self.assertEqual(p['rules']['head']['com_evidence'], 'project_proxy')
            self.assertEqual(p['rules']['torso']['com_evidence'], 'project_proxy')
        self.assertEqual(anthropometry_profile()['com_fractions'], CENTER_OF_MASS_FRACTIONS)
        with self.assertRaises(ValueError):
            anthropometry_profile('personal_measured_model')

    def test_profiles_do_not_mutate_defaults_or_each_other(self):
        p = anthropometry_profile()
        p['com_fractions']['forearm'] = .1
        p['rules']['head']['com_evidence'] = 'measured'
        p['distal_components']['forearm']['bone_mass'] = 1.
        fresh = anthropometry_profile()
        self.assertEqual(fresh['com_fractions']['forearm'], .69)
        self.assertEqual(fresh['rules']['head']['com_evidence'], 'project_proxy')
        self.assertEqual(fresh['distal_components']['forearm']['bone_mass'], .016)

    def test_derived_lump_coefficients_match_explicit_split(self):
        p = anthropometry_profile('dempster_endpoint_proxy_v1')
        self.assertAlmostEqual(p['com_fractions']['forearm'], (.016 * .430 + .006) / .022)
        self.assertAlmostEqual(p['com_fractions']['shank'], (.0465 * .433 + .0145) / .061)
        # Independent full-body sum with separate forearm, hand, shank and foot.
        j = forward_kinematics(self.q, self.body)
        points = [j.shoulder_center + .4 * (j.shoulder_center - j.hip_center),
                  (j.hip_center + j.shoulder_center) / 2]
        masses = [.081, .497]
        for name, limb in j.limbs.items():
            arm = name.endswith('hand')
            points.extend([limb.root + (.436 if arm else .433) * (limb.middle - limb.root),
                           limb.middle + (.430 if arm else .433) * (limb.end - limb.middle), limb.end])
            masses.extend([.028, .016, .006] if arm else [.100, .0465, .0145])
        expected = sum(m * point for m, point in zip(masses, points)) / sum(masses)
        actual = center_of_mass(self.q, self.body, make_mass_model(70), p['com_fractions'])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_distal_orientation_changes_com_by_its_mass_share(self):
        elbow = np.array([0., 0., 0.]); wrist = np.array([30., 0., 0.])
        delta = np.array([0., 8., 3.])
        for name, fraction in [('forearm', .006 / .022), ('shank', .0145 / .061)]:
            base = combined_distal_center_of_mass(elbow, wrist, wrist, name)
            shifted = combined_distal_center_of_mass(elbow, wrist, wrist + delta, name)
            np.testing.assert_allclose(shifted - base, fraction * delta, atol=1e-12)

    def test_combined_com_is_rigid_transform_equivariant(self):
        rng = np.random.default_rng(10)
        a, b, c = rng.normal(size=(3, 5, 3))
        rot = rotation(np.array([[.4, -.3, .2]]))[0]
        offset = np.array([12., -20., 50.])
        value = combined_distal_center_of_mass(a, b, c, 'forearm')
        transformed = combined_distal_center_of_mass(a @ rot.T + offset, b @ rot.T + offset,
                                                     c @ rot.T + offset, 'forearm')
        np.testing.assert_allclose(transformed, value @ rot.T + offset, atol=1e-12)

    def test_invalid_points_masses_and_coefficients_rejected(self):
        np.testing.assert_allclose(weighted_center_of_mass([[0., 0.], [2., 4.]],
                                                          [1e308, 1e308]), [1., 2.])
        for masses in ([0., 1.], [-1., 2.], [1.], [1., np.nan], [1., 1j]):
            with self.assertRaises(ValueError):
                weighted_center_of_mass([[0., 0.], [1., 2.]], masses)
        with self.assertRaises(ValueError):
            weighted_center_of_mass([[0., np.nan]], [1.])
        with self.assertRaises(ValueError):
            combined_distal_center_of_mass([0., 0.], [1., 0.], [1., 0., 0.], 'forearm')
        with self.assertRaises(ValueError):
            combined_distal_center_of_mass([0., 0.], [1., 0.], [1., 0.], 'head')
        for value in (-.1, 1.01, np.nan, np.inf, 1j):
            with self.assertRaises(ValueError):
                validate_com_fractions({**CENTER_OF_MASS_FRACTIONS, 'thigh': value})
        validate_com_fractions({**CENTER_OF_MASS_FRACTIONS, 'thigh': 0., 'forearm': 1.})
        with self.assertRaises(ValueError):
            validate_com_fractions({**CENTER_OF_MASS_FRACTIONS, 'head': -.1})
        with self.assertRaises(ValueError):
            center_of_mass(self.q, self.body, {**make_mass_model(70), 'head': -1.})

    def test_two_and_three_dimensional_com_match_for_both_profiles(self):
        for name in ANTHROPOMETRY_PROFILES:
            c = anthropometry_profile(name)['com_fractions']
            scalar = center_of_mass(self.q, self.body, make_mass_model(70), c)
            batched, jac = batch_com_and_jacobian(self.q[None], self.body, c)
            _, spatial, _ = kinematics(self.spatial_pose(self.q), self.body, com_fractions=c)
            np.testing.assert_allclose(batched[0], scalar, atol=1e-12)
            np.testing.assert_allclose(spatial[0], [*scalar, 0.], atol=1e-12)
            for i in range(11):
                perturbed = self.q.astype(complex); perturbed[i] += 1e-25j
                actual = center_of_mass(perturbed, self.body, make_mass_model(70), c).imag / 1e-25
                np.testing.assert_allclose(actual, jac[0, :, i], atol=1e-10)

    def test_spatial_complex_step_derivative_with_experimental_profile(self):
        c = anthropometry_profile('dempster_endpoint_proxy_v1')['com_fractions']
        q = self.spatial_pose(self.q)
        q[0, 4:6] = [.2, -.1]; q[0, 7::4] = [.1, .2, -.1, .3]
        v = np.random.default_rng(12).normal(size=q.shape)
        analytical = kinematics(q + 1e-25j * v, self.body, com_fractions=c)[1].imag / 1e-25
        eps = 1e-5
        finite = (kinematics(q + eps * v, self.body, com_fractions=c)[1]
                  - kinematics(q - eps * v, self.body, com_fractions=c)[1]) / (2 * eps)
        np.testing.assert_allclose(analytical, finite, rtol=1e-7, atol=1e-7)

    def test_experimental_profile_is_used_by_static_gravity_solver(self):
        p = anthropometry_profile('dempster_endpoint_proxy_v1')
        targets = {k: limb.end for k, limb in forward_kinematics(self.q, self.body).limbs.items()}
        result = solve_inverse_kinematics_with_gravity(self.body, targets, self.q, 70.,
                    mass_fractions=p['mass_fractions'], com_fractions=p['com_fractions'])
        expected = center_of_mass(result.pose_vector, self.body, make_mass_model(70), p['com_fractions'])
        np.testing.assert_allclose(result.center_of_mass_cm, expected, atol=1e-12)
        self.assertTrue(result.balanced)
        self.assertLess(max(result.contact_errors_cm.values()), .05)


if __name__ == '__main__':
    unittest.main()
