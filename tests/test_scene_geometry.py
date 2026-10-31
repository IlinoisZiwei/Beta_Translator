"""Metric assumptions must not silently enable multi-plane IK."""

import copy
import unittest

import numpy as np

from beta_translator.scene_geometry import (
    analyze_geometry, local_scale_samples, parallel_line_vanishing_point,
)


class SceneGeometryTests(unittest.TestCase):
    def setUp(self):
        self.obs = dict(image_size=[1000, 800], hole_pairs=[dict(id='a',
            panel_id='wall', direction='row', endpoints_px=[[100, 100], [160, 100]],
            intervals=2, review_status='draft')], floor_lines=dict(
                relation='parallel_in_world', image_lines_reviewed=True,
                line_endpoints_px=[[[100, 300], [200, 200]], [[500, 300], [400, 200]]]))

    def test_intervals_and_spacing_change_local_scale_only(self):
        a = analyze_geometry(self.obs, 15, 173, 165)
        b = analyze_geometry(self.obs, 30, 173, 165)
        self.assertEqual(a['local_scale_samples'][0]['local_px_per_cm'], 2)
        self.assertEqual(b['local_scale_samples'][0]['local_px_per_cm'], 1)
        for result in (a, b):
            self.assertIsNone(result['global_px_per_cm'])
            self.assertIsNone(result['wall_floor_angle_deg'])
            self.assertFalse(result['ready_for_metric_ik'])

    def test_source_height_does_not_replace_target_or_calibrate_pixels(self):
        a = analyze_geometry(self.obs, 15, 173, 165)
        b = analyze_geometry(self.obs, 15, 180, 165)
        self.assertEqual(a['body_priors']['target_segments_cm'], b['body_priors']['target_segments_cm'])
        self.assertNotEqual(a['body_priors']['source_segments_cm'], b['body_priors']['source_segments_cm'])
        self.assertEqual(a['local_scale_samples'], b['local_scale_samples'])

    def test_known_vanishing_point_is_not_a_horizon(self):
        result = analyze_geometry(self.obs, 15, 173, 165)
        np.testing.assert_allclose(result['floor']['vanishing_point']['pixel_xy'], [300, 100])
        self.assertIsNone(result['floor_horizon'])

    def test_unreviewed_floor_lines_are_not_used(self):
        self.obs['floor_lines']['image_lines_reviewed'] = False
        self.assertIsNone(analyze_geometry(self.obs, 15, 173, 165)['floor']['vanishing_point'])

    def test_parallel_image_lines_have_vanishing_point_at_infinity(self):
        r = parallel_line_vanishing_point([[[10, 10], [20, 10]], [[10, 20], [20, 20]]], [100, 100])
        self.assertTrue(r['at_infinity']); self.assertIsNone(r['pixel_xy'])
        with self.assertRaises(ValueError):
            parallel_line_vanishing_point([[[10, 10], [20, 10]], [[30, 10], [40, 10]]], [100, 100])

    def test_invalid_measurements_fail_closed(self):
        for spacing in (0, -15, np.nan):
            with self.assertRaises(ValueError): local_scale_samples(self.obs['hole_pairs'], spacing, [1000, 800])
        for key, value in [('intervals', 0), ('endpoints_px', [[-10, 10], [20, 20]])]:
            pairs = copy.deepcopy(self.obs['hole_pairs']); pairs[0][key] = value
            with self.assertRaises(ValueError): local_scale_samples(pairs, 15, [1000, 800])


if __name__ == '__main__':
    unittest.main()
