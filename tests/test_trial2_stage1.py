"""Source-chain, contact-equivalence, force gating and Jacobian regressions."""
import unittest
from pathlib import Path
import numpy as np
from beta_translator import anatomical_arms as arms
from beta_translator.trial2_stage1 import Stage1Problem, stage1_kinematics, verify_chain, audit_stage1
from beta_translator.trial2_anatomy_pipeline import slice_arrays
from beta_translator.trial2_spatial_pipeline import load_fit
from beta_translator.trial2_spatial_data import load_trial2, episode_regions
from beta_translator.trial2_scene3d import Trial2Scene
from beta_translator.inverse_kinematics import make_body


class ChainTests(unittest.TestCase):
    def test_independent_pinned_source_rotations(self):
        r = verify_chain()
        self.assertTrue(r['rotational_chain_pass'])
        self.assertGreater(r['previous_hand_orientation_max_difference_deg'], 1.)

    def test_bilateral_hinge_axis_and_distal_isolation(self):
        q = np.zeros((1, 28)); q[:, :3] = [0, 100, 30]
        q[:, 6:13] = q[:, 13:20] = np.deg2rad([20, 80, -25, 50, 20, 10, 30])
        L = make_body(180); before = stage1_kinematics(q, L, 180)
        for side in (0, 1):
            q1 = q.copy(); q1[:, 9+7*side] += .03
            after = stage1_kinematics(q1, L, 180)
            axis = arms.OSIM_TO_BODY@arms.AXES[2]
            if side == 0: axis = -arms.MIRROR@axis  # Axial vector under reflection.
            axis = before['orientations'][0, 1+3*side]@axis
            v = before['joints'][0, side, 2]-before['joints'][0, side, 1]
            expected = arms.axis_rotation(np.array(.03), axis)@v
            np.testing.assert_allclose(after['joints'][0, side, 2]-after['joints'][0, side, 1], expected, atol=1e-10)
        q[:, [10, 11, 12, 17, 18, 19]] += .05
        after = stage1_kinematics(q, L, 180)
        np.testing.assert_allclose(before['joints'], after['joints'], atol=1e-10)
        self.assertGreater(np.linalg.norm(before['palm_contacts']-after['palm_contacts']), .01)
        np.testing.assert_allclose(np.linalg.det(after['orientations']), 1, atol=1e-12)


class ContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.baseline = cls.root/'outputs/trial2/anatomy_plans/target_180cm_controls_db7b5a5de762'
        if not cls.baseline.exists():raise unittest.SkipTest('Trial2 assets unavailable')
        cls.data = load_trial2(cls.root); cls.scene = Trial2Scene(cls.root, 15, 0, 15)
        cls.regions = episode_regions(cls.data, cls.scene, 180)
        cls.old = load_fit(cls.baseline/'balanced')

    def problem(self, start=550):
        ix = np.arange(start, start+4); n = len(self.data['frames'])
        d = slice_arrays(self.data, ix, n); old = slice_arrays(self.old, ix, n)
        regions = {**self.regions, 'mapping':self.regions['mapping'][ix]}
        return Stage1Problem(d, self.scene, regions, old, old['anatomical_q'])

    def test_exact_contact_definition_and_ground_frame(self):
        for start in [0, 550, 615]:
            p = self.problem(start); y = (p.expand@p.initial).reshape(p.n, 48); g = p.geometry(y)
            np.testing.assert_allclose(g['joints'][:, :, 2]-g['targets'], g['contact_points']-g['contacts'], atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(g['foot_frames']), 1, atol=1e-12)
            active = p.data['known'][:, 2:]
            length = np.linalg.norm(g['foot_contacts']-g['joints'][:, 2:, 2], axis=-1)
            expected = np.where(p.data['ground'][:, 2:], .035, np.hypot(.045, .025))*180
            np.testing.assert_allclose(length[active], expected[active], atol=1e-10)

    def test_contact_audit_does_not_accept_wrist_when_palm_misses(self):
        p = self.problem(); g = p.geometry((p.expand@p.initial).reshape(p.n, 48))
        fit = dict(g, anatomical_q=p.initial[:p.n*40].reshape(p.n, 40)[:, :28],
                   episode_uv=p.initial[-2*p.e:].reshape(p.e, 2), solver={'success':False})
        # Spoof old-style exact wrist targets; the new physical patch must still fail.
        fit['targets'] = fit['joints'][:, :, 2].copy()
        fit['contact_points'] = fit['contacts']+10
        m, gaps, _ = audit_stage1(fit, p.data, p.scene, p.regions)
        self.assertTrue(m.contact_failed.all())
        self.assertFalse(m.conditional_balance.any())
        np.testing.assert_allclose(gaps, np.sqrt(300))

    def test_sparse_jacobian_including_distal_contacts_and_shared_regions(self):
        p = self.problem(); x = p.initial.copy(); rng = np.random.default_rng(14)
        direction = rng.normal(size=len(x)); direction[p.lower==p.upper] = 0
        x += 1e-4*rng.normal(size=len(x))
        _, J = p.evaluate(x); eps = 2e-7
        fd = (p.evaluate(x+eps*direction)[0]-p.evaluate(x-eps*direction)[0])/(2*eps)
        np.testing.assert_allclose(J@direction, fd, atol=1e-3, rtol=3e-5)


if __name__ == '__main__':unittest.main()
