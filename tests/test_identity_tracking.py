"""Identity continuity and display-only occlusion handling checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from beta_translator.identity_tracking import identity_match, interpolate_display, load_identity_guard
from beta_translator.project_io import file_sha256, save_npz


class IdentityTrackingTests(unittest.TestCase):
    def setUp(self):
        self.torso = np.array([[90.,80.],[110.,80.],[94.,140.],[106.,140.]])

    def test_distant_or_large_bystander_cannot_take_over(self):
        self.assertFalse(identity_match(self.torso + [200,0], self.torso)[0])
        center = self.torso.mean(0)
        self.assertFalse(identity_match((self.torso-center)*3+center, self.torso)[0])
        self.assertTrue(identity_match(self.torso+[2,1], self.torso, self.torso)[0])

    def test_anchor_alone_does_not_allow_a_large_temporal_jump(self):
        moved = self.torso + [200,0]
        self.assertFalse(identity_match(moved, moved, self.torso)[0])

    def test_short_gap_is_interpolated_in_wall_coordinates_without_changing_observations(self):
        points = np.array([self.torso, self.torso+5, self.torso+[20,0]])
        points[1] = np.nan
        matrices = {i:np.array([[1.,0.,10*i],[0.,1.,0.],[0.,0.,1.]]) for i in range(3)}
        display, inferred = interpolate_display(points, [True,False,True], matrices, [0,1,2,3], 30)
        np.testing.assert_allclose(display[1], self.torso+[10,0])
        self.assertTrue(inferred[1]);self.assertTrue(np.isnan(points[1]).all())

    def test_long_gap_or_large_endpoint_change_is_not_filled(self):
        points = np.full((64,4,2),np.nan);points[[0,-1]]=self.torso
        matrices = {i:np.eye(3) for i in range(len(points))}
        _, inferred = interpolate_display(points,np.isfinite(points).all((1,2)),matrices,[0,1,2,3],30)
        self.assertFalse(inferred.any())
        points = np.array([self.torso,self.torso,self.torso+[100,0]])
        points[1]=np.nan
        _, inferred=interpolate_display(points,[True,False,True],matrices,[0,1,2,3],30)
        self.assertFalse(inferred.any())

    def test_guard_rejects_changed_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'state').mkdir()
            video=root/'video';video.write_bytes(b'video')
            features=root/'features';features.write_bytes(b'features')
            self.assertIsNone(load_identity_guard(root,video,features))
            metadata=dict(video_sha256=file_sha256(video),wholebody_sha256=file_sha256(features),frame_count=2)
            save_npz(root/'state/identity_guard.npz',metadata=np.array(json.dumps(metadata)),
                     observed=np.array([True,False]),display_keypoints=np.zeros((2,133,2)),display_inferred=np.array([False,True]))
            self.assertEqual(load_identity_guard(root,video,features)['observed'].tolist(),[True,False])
            features.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'different video or whole-body'):
                load_identity_guard(root,video,features)


if __name__ == '__main__':unittest.main()
