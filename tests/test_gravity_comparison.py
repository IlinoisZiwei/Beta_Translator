"""Comparison checks: fair baseline, infeasible contacts, saved-input boundaries."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from beta_translator.gravity_comparison import (
    evaluate_fixed_pose_balance, compare_endpoint, comparison_metrics, run_gravity_comparison,
)
from beta_translator.inverse_kinematics import (
    LIMBS, RATIOS, make_body, forward_kinematics, GravityIKResult,
)
from beta_translator.pose_adapter import LIMB_LANDMARKS, wall_cm_to_image
from beta_translator.project_io import file_sha256, save_alignment, save_reviewed_holds


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.body = make_body(165)
        self.pose = np.array([180, 160, .05, 2.7, .4, 3.5, .5, .25, .5, -.2, .6])
        joints = forward_kinematics(self.pose, self.body)
        self.targets = {limb: joints.limbs[limb].end for limb in LIMBS}

    def test_baseline_force_fit_does_not_move_pose(self):
        original = self.pose.copy()
        result = evaluate_fixed_pose_balance(self.body, self.targets, self.pose, 70)
        np.testing.assert_array_equal(result.pose_vector, original)
        np.testing.assert_array_equal(self.pose, original)
        self.assertTrue(result.balanced)
        self.assertLess(np.linalg.norm(result.net_force_n), .01 * 70 * 9.81)

    def test_balanced_does_not_mean_contact_valid(self):
        result = GravityIKResult(self.pose, {'left_hand': 12.}, {'left_hand': np.array([0., 686.7])},
                                 np.array([0., 0.]), np.zeros(2), 0., True)
        metrics = comparison_metrics(result, self.pose, self.body, .8, .1)
        self.assertTrue(metrics['balanced'])
        self.assertFalse(metrics['contacts_within_tolerance'])
        self.assertFalse(metrics['model_checks_pass'])

    def test_force_limit_failure_visible_for_both_methods(self):
        fits = compare_endpoint(self.body, self.targets, self.pose, np.ones(9), 70, max_force_n=100.)
        for fit in fits.values():
            self.assertTrue(fit['converged'])
            self.assertFalse(fit['metrics']['balanced'])
            self.assertFalse(fit['metrics']['model_checks_pass'])

    def test_gravity_failure_preserves_ordinary_result(self):
        with patch('beta_translator.gravity_comparison.solve_inverse_kinematics_with_gravity',
                   side_effect=RuntimeError('iteration limit')):
            fits = compare_endpoint(self.body, self.targets, self.pose, np.ones(9), 70)
        self.assertTrue(fits['ordinary']['converged'])
        self.assertFalse(fits['gravity']['converged'])
        self.assertEqual(fits['gravity']['error'], 'iteration limit')


class SavedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/'outputs'
        (self.output/'state').mkdir(parents=True)
        (self.output/'tables').mkdir()
        self.video = self.root/'input.mp4'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 720))
        self.assertTrue(writer.isOpened())
        for _ in range(130):
            writer.write(np.zeros((720, 640, 3), np.uint8))
        writer.release()
        self.reference = self.root/'confirmed.json'
        scale = np.hypot(80, -21) / 46
        body = make_body(165)
        pose = np.array([180, 160, .05, 2.7, .4, 3.5, .5, .25, .5, -.2, .6])
        after = pose.copy(); after[3] -= .2
        contacts = {'before': dict(zip(LIMBS, ('H44', 'H33', 'H62', 'H65'))),
                    'after': dict(zip(LIMBS, ('H28', 'H33', 'H62', 'H65')))}
        rows, holds = [], {}
        for frame, phase, vector in [(107, 'before', pose), (129, 'after', after)]:
            joints = forward_kinematics(vector, body)
            row = {'frame': frame}
            for limb, chain in joints.limbs.items():
                pixels = wall_cm_to_image(chain, scale, 720)
                for name, xy in zip(LIMB_LANDMARKS[limb], pixels):
                    row[name+'_x_px'], row[name+'_y_px'] = xy
                    row[name+'_visibility'] = .9
                holds[contacts[phase][limb]] = pixels[-1]
            rows.append(row)
        pose_path = self.output/'tables/pose_coordinates.csv'
        pd.DataFrame(rows).to_csv(pose_path, index=False)
        records, masks = [], {}
        for hold, xy in holds.items():
            mask = np.zeros((720, 640), np.uint8)
            cv2.circle(mask, tuple(np.rint(xy).astype(int)), 5, 1, -1)
            masks[hold] = mask.astype(bool)
            records.append(dict(hold_id=hold, cx_px=float(xy[0]), cy_px=float(xy[1])))
        hold_ids = list(holds)
        save_reviewed_holds(self.output/'state/reviewed_hold_map.npz', self.video,
            np.zeros((720, 640, 3), np.uint8), records, masks, 'reviewed-fixture', hold_ids)
        save_alignment(self.output/'state/alignment_cache.npz',
            {f: (np.eye(3), 100, 'ok') for f in (107, 129)}, self.video, pose_path, {})
        annotation = dict(video_sha256=file_sha256(self.video), candidate_run_id='reviewed-fixture',
            selected_hold_ids=hold_ids, frames=dict(before=107, after=129), contacts=contacts,
            moving_limb='left_hand', body=dict(height_cm=165, ratios=RATIOS, px_per_cm=scale))
        self.reference.write_text(json.dumps(annotation))
        # Deliberately unrelated live annotation; the comparison must never replace it.
        (self.output/'state/endpoint_annotations.json').write_text('{"user_work": "keep"}')

    def run_comparison(self, **kwargs):
        return run_gravity_comparison(self.video, self.output, self.reference, height_cm=180,
                                      ratios={**RATIOS, 'forearm': .15}, **kwargs)

    def test_custom_body_reuse_preservation_and_tamper_guard(self):
        before = {p: file_sha256(p) for p in self.root.rglob('*') if p.is_file()}
        result = self.run_comparison()
        for p, digest in before.items():
            self.assertEqual(file_sha256(p), digest)
        self.assertEqual(result['configuration']['height_cm'], 180)
        body = make_body(180, {**RATIOS, 'forearm': .15})
        for endpoint in result['endpoints']:
            for fit in endpoint['fits'].values():
                self.assertTrue(fit['converged'])
                joints = forward_kinematics(fit['pose_vector'], body)
                arm = joints.limbs['left_hand']
                self.assertAlmostEqual(np.linalg.norm(arm.end - arm.middle), 27.)
        image = Path(result['output_dir'])/'comparison.png'
        stamp = image.stat().st_mtime_ns
        self.assertEqual(self.run_comparison()['output_dir'], result['output_dir'])
        self.assertEqual(image.stat().st_mtime_ns, stamp)
        image.write_bytes(b'changed by user')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.run_comparison()
        self.assertEqual(image.read_bytes(), b'changed by user')

    def test_stale_alignment_and_wrong_reference_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reference frame 0'):
            self.run_comparison(reference_frame_number=1)
        path = self.output/'tables/pose_coordinates.csv'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.run_comparison()
        self.assertFalse((self.output/'gravity_comparison').exists())

    def test_wrong_video_is_rejected(self):
        annotation = json.loads(self.reference.read_text())
        annotation['video_sha256'] = 'wrong'
        self.reference.write_text(json.dumps(annotation))
        with self.assertRaisesRegex(ValueError, 'different video'):
            self.run_comparison()


if __name__ == '__main__':
    unittest.main()
