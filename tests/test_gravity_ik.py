import unittest
import numpy as np

from beta_translator.inverse_kinematics import (
    LIMBS, make_body, forward_kinematics, solve_inverse_kinematics,
    make_mass_model, center_of_mass, static_balance_error,
    solve_inverse_kinematics_with_gravity, MASS_FRACTIONS,
)

HEIGHT, MASS = 175.0, 70.0
WEIGHT = MASS * 9.81


def climbing_pose():
    # hip_x, hip_y, torso, then (theta1, theta2) for LH, RH, LF, RF; 0 = pointing down, pi = up
    return np.array([0, 100, 0.05,
                     np.pi - 0.45, 0.4, np.pi + 0.35, 0.5,
                     0.25, 0.5, -0.2, 0.6])


def targets_from(pose, body, shift=(0.0, 0.0)):
    joints = forward_kinematics(pose, body)
    return {l: joints.limbs[l].end + np.array(shift) for l in LIMBS}


class MassModel(unittest.TestCase):
    def test_fractions_sum_to_one_and_scale(self):
        masses = make_mass_model(MASS)
        total = sum(c * masses[n] for n, c in
                    {"head": 1, "torso": 1, "upper_arm": 2, "forearm": 2, "thigh": 2, "shank": 2}.items())
        self.assertAlmostEqual(total, MASS)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError): make_mass_model(0)
        with self.assertRaises(ValueError): make_mass_model(MASS, {**MASS_FRACTIONS, "head": 0.2})
        with self.assertRaises(ValueError): make_mass_model(MASS, {"head": 1.0})

    def test_standing_com_height(self):
        body = make_body(HEIGHT)
        pose = np.zeros(11); pose[:2] = (0, body["thigh"] + body["shank"])
        com = center_of_mass(pose, body, make_mass_model(MASS))
        self.assertAlmostEqual(com[0], 0.0, places=6)
        self.assertTrue(0.52 * HEIGHT < com[1] < 0.57 * HEIGHT, com)


class Balance(unittest.TestCase):
    def test_hanging_under_hold_is_balanced(self):
        com = np.array([10.0, 120.0])
        net_force, net_moment = static_balance_error(com, [[10.0, 200.0]], [[0.0, WEIGHT]], MASS)
        self.assertLess(np.linalg.norm(net_force), 1e-9)
        self.assertAlmostEqual(net_moment, 0.0)
        _, off_axis = static_balance_error(com, [[40.0, 200.0]], [[0.0, WEIGHT]], MASS)
        self.assertGreater(abs(off_axis), 1.0)


class GravitySolver(unittest.TestCase):
    def setUp(self):
        self.body = make_body(HEIGHT)
        self.reference = climbing_pose()

    def test_four_contacts_balanced(self):
        targets = targets_from(self.reference, self.body, shift=(3.0, 2.0))
        result = solve_inverse_kinematics_with_gravity(self.body, targets, self.reference, MASS)
        self.assertTrue(result.balanced, (result.net_force_n, result.net_moment_ncm))
        self.assertLess(max(result.contact_errors_cm.values()), 0.05)
        self.assertLess(np.linalg.norm(result.net_force_n), 0.01 * WEIGHT)
        self.assertAlmostEqual(sum(f[1] for f in result.contact_forces_n.values()), WEIGHT, delta=0.01 * WEIGHT)
        for limb in ("left_foot", "right_foot"):
            fx, fy = result.contact_forces_n[limb]
            self.assertGreaterEqual(fy, -1e-9)
            self.assertLessEqual(abs(fx), 0.8 * fy + 0.01 * WEIGHT)

    def test_one_hand_hang_puts_com_under_hold(self):
        hold = (30.0, 210.0)
        result = solve_inverse_kinematics_with_gravity(
            self.body, {"left_hand": hold, "right_hand": None, "left_foot": None, "right_foot": None},
            self.reference, MASS)
        self.assertTrue(result.balanced)
        self.assertAlmostEqual(result.center_of_mass_cm[0], hold[0], delta=1.0)
        self.assertAlmostEqual(result.contact_forces_n["left_hand"][1], WEIGHT, delta=0.01 * WEIGHT)

    def test_pose_independent_of_mass_with_default_limit(self):
        targets = targets_from(self.reference, self.body, shift=(3.0, 2.0))
        a = solve_inverse_kinematics_with_gravity(self.body, targets, self.reference, 50.0)
        b = solve_inverse_kinematics_with_gravity(self.body, targets, self.reference, 90.0)
        np.testing.assert_allclose(a.pose_vector, b.pose_vector, atol=1e-4)
        np.testing.assert_allclose(b.contact_forces_n["left_hand"] / 90.0, a.contact_forces_n["left_hand"] / 50.0, atol=1e-4)

    def test_infeasible_force_limit_is_reported(self):
        targets = targets_from(self.reference, self.body)
        result = solve_inverse_kinematics_with_gravity(
            self.body, targets, self.reference, MASS, max_force_n=0.2 * WEIGHT)   # 4 limbs x 0.2 BW < 1 BW
        self.assertFalse(result.balanced)
        self.assertLess(max(result.contact_errors_cm.values()), 0.5)

    def test_free_limbs_and_validation(self):
        targets = targets_from(self.reference, self.body)
        with self.assertRaises(ValueError):
            solve_inverse_kinematics_with_gravity(self.body, {l: None for l in LIMBS}, self.reference, MASS)
        with self.assertRaises(ValueError):
            solve_inverse_kinematics_with_gravity(self.body, {"elbow": (0, 0)}, self.reference, MASS)
        with self.assertRaises(ValueError):
            solve_inverse_kinematics_with_gravity(self.body, targets, self.reference[:5], MASS)
        with self.assertRaises(ValueError):
            solve_inverse_kinematics_with_gravity(self.body, targets, self.reference, -1)

    def test_array_angle_weight_like_the_notebook(self):
        targets = targets_from(self.reference, self.body, shift=(2.0, 1.0))
        weights = 0.3 * np.linspace(0.5, 1.0, 9)
        result = solve_inverse_kinematics_with_gravity(self.body, targets, self.reference, MASS, angle_weight=weights)
        self.assertLess(max(result.contact_errors_cm.values()), 0.05)


class ExistingSolverUnchanged(unittest.TestCase):
    def test_still_returns_pose_and_errors(self):
        body = make_body(HEIGHT)
        ref = climbing_pose()
        pose, errors = solve_inverse_kinematics(body, targets_from(ref, body, (3.0, 2.0)), ref)
        self.assertEqual(pose.shape, (11,))
        self.assertLess(max(errors.values()), 0.05)


if __name__ == "__main__":
    unittest.main()