"""Experimental acceptance must not invent contacts or metric geometry."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from beta_translator.direct_ik import build_motion_candidates, geometry_blockers, run_direct_ik
from beta_translator.inverse_kinematics import LIMBS, RATIOS
from beta_translator.project_io import file_sha256


class DirectIKTests(unittest.TestCase):
    def setUp(self):
        self.prediction = dict(video_sha256='video', candidate_run_id='map',
            selected_hold_ids=['A', 'B', 'C', 'D', 'E'], moves=[dict(
                before_frame=0, after_frame=5, release_frame=2, arrival_frame=4,
                moving_limb='left_hand', from_hold='A', to_hold='B',
                requires_review=True, eligible_for_ik=False)])
        self.contacts = pd.DataFrame([dict(frame=f, limb=l,
            candidate_hold=('A' if f < 2 else 'free' if f < 4 else 'B') if l == 'left_hand'
                else dict(right_hand='C', left_foot='D', right_foot='E')[l],
            hold_id='unknown', confidence=.1) for f in range(6) for l in LIMBS])
        self.body = dict(height_cm=190, ratios=dict(RATIOS), px_per_cm=2.)

    def build(self, poses=None):
        return build_motion_candidates(self.prediction, self.contacts, self.body,
            set(range(6)) if poses is None else poses, set(range(6)))[0]

    def test_user_enabled_path_accepts_decoded_states_below_model_threshold(self):
        row = self.build()
        self.assertTrue(row['solver_compatible'])
        self.assertEqual(row['annotation']['body']['height_cm'], 190)
        self.assertTrue(row['annotation']['automatic_evidence']['original_requires_review'])
        self.assertFalse(self.prediction['moves'][0]['eligible_for_ik'])
        self.assertTrue((self.contacts.hold_id == 'unknown').all())

    def test_free_foot_is_not_changed_to_a_hold(self):
        self.contacts.loc[self.contacts.limb == 'right_foot', 'candidate_hold'] = 'free'
        row = self.build()
        self.assertIsNone(row['annotation'])
        self.assertEqual(row['support_states']['right_foot'], ['free'])
        self.assertIn('right_foot: free', ' '.join(row['reasons']))

    def test_support_change_and_smear_are_not_fixed_supports(self):
        self.contacts.loc[(self.contacts.limb == 'left_foot') & (self.contacts.frame == 2), 'candidate_hold'] = 'wall_contact'
        self.assertFalse(self.build()['solver_compatible'])

    def test_missing_body_observation_is_not_interpolated_into_evidence(self):
        self.assertFalse(self.build(set([0, 1, 3, 4, 5]))['solver_compatible'])

    def test_moving_schedule_must_match_native_frame_labels(self):
        self.contacts.loc[(self.contacts.limb == 'left_hand') & (self.contacts.frame == 2), 'candidate_hold'] = 'A'
        self.assertFalse(self.build()['solver_compatible'])

    def test_missing_scale_and_multi_plane_geometry_remain_explicit(self):
        self.assertEqual(len(geometry_blockers(None, False)), 2)
        self.assertEqual(geometry_blockers(2., True), [])

    def test_direct_bridge_dispatches_solver_without_manual_annotation_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'state').mkdir(); (root/'tables').mkdir(); (root/'predictions').mkdir()
            video = root/'video.mp4'; video.write_bytes(b'test video')
            pred = copy.deepcopy(self.prediction); pred['video_sha256'] = file_sha256(video)
            (root/'predictions/automatic_contacts.json').write_text(json.dumps(pred))
            self.contacts.to_csv(root/'predictions/limb_contacts.csv', index=False)
            pd.DataFrame(dict(frame=range(6))).to_csv(root/'tables/pose_coordinates.csv', index=False)
            np.savez(root/'state/alignment_cache.npz', settings=json.dumps({}))
            (root/'state/reviewed_hold_map.npz').write_bytes(b'map copied without mutation')
            transforms = {f: (np.eye(3), 100, 'ok') for f in range(6)}
            with patch('beta_translator.direct_ik.load_review_bundle', return_value=(pred, self.contacts)), \
                 patch('beta_translator.direct_ik.load_alignment', return_value=transforms), \
                 patch('beta_translator.motion_pipeline.current_motion_outputs', return_value=[]), \
                 patch('beta_translator.motion_pipeline.run_motion_analysis') as solver:
                blocked = run_direct_ik(video, root, root/'pose', root/'package', root/'predictions',
                    190, RATIOS, None, False)
                solver.assert_not_called()
                self.assertEqual(blocked['moves'][0]['status'], 'pending_geometry')
                result = run_direct_ik(video, root, root/'pose', root/'package', root/'predictions',
                    190, RATIOS, 2., True)
                solver.assert_called_once()
                self.assertEqual(result['generated_count'], 1)
                self.assertFalse((root/'state/endpoint_annotations.json').exists())
                saved = json.loads((root/'automatic/ik/move_000/state/endpoint_annotations.json').read_text())
                self.assertEqual(saved['contacts']['after']['left_hand'], 'B')
                self.assertEqual(saved['motion']['release_frame'], 2)
                self.assertEqual(saved['body']['height_cm'], 190)


if __name__ == '__main__':
    unittest.main()
