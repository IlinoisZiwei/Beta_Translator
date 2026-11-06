"""Real mesh tests: frames, personalization, surface anchors and saved trials."""
from copy import deepcopy
from pathlib import Path
import json
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from beta_translator.avatar_model import AnnyAvatar,ANNY_TO_PROJECT,group_frames
from beta_translator.body_user import load_user


class AvatarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.avatar=AnnyAvatar()

    def test_bind_identity_and_true_mesh_dimensions(self):
        a=self.avatar
        np.testing.assert_allclose(a.pose(a.neutral)[0],a.vertices,atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(ANNY_TO_PROJECT),1)
        self.assertAlmostEqual(a.measurements['height_cm'],175,places=6)
        self.assertAlmostEqual(a.measurements['arm_span_cm'],176,delta=.1)
        np.testing.assert_allclose(a.weights.sum(1),1,atol=1e-10)
        self.assertGreater(len(a.faces),20000)
        self.assertIn('finger5-3.L',a.bone_labels)

    def test_global_rigid_transform_is_equivariant(self):
        a=self.avatar;g=deepcopy(a.neutral);Q=Rotation.from_euler('xyz',[.3,-.7,.2]).as_matrix();t=np.array([30,250,13])
        for key in ['joints','foot_origins']:
            g[key]=g[key]@Q.T+t
        for key in ['orientations','foot_frames']:
            g[key]=Q@g[key]
        np.testing.assert_allclose(a.pose(g)[0],a.vertices@Q.T+t,atol=1e-9)

    def test_wrist_twist_is_preserved_with_stationary_endpoints(self):
        a=self.avatar;g=deepcopy(a.neutral);Q=Rotation.from_euler('y',60,degrees=True).as_matrix()
        g['orientations'][:,3]=Q@g['orientations'][:,3]
        V=a.pose(g);p=a.contact_points(V,np.zeros((1,4),bool))
        H=g['orientations'][0,3];d=a.profile['dimensions_cm'];W=g['joints'][0,0,2]
        expected=W-H[:,1]*d['hand_patch_distance']-H[:,2]*d['hand_patch_depth']
        np.testing.assert_allclose(p[0,0],expected,atol=1e-8)
        unchanged=a.weights[:,4]==0
        np.testing.assert_allclose(V[0,unchanged],a.vertices[unchanged],atol=1e-9)
        self.assertGreater(np.linalg.norm(p[0,0]-a.neutral['palm_contacts'][0,0]),1)

    def test_ground_and_hold_use_different_persistent_sole_anchors(self):
        a=self.avatar;V=a.pose(a.neutral)
        ground=a.contact_points(V,np.ones((1,4),bool));hold=a.contact_points(V,np.zeros((1,4),bool))
        for s in range(2):
            expected=a.neutral['foot_frames'][0,s,:,0]*.2*a.profile['dimensions_cm']['foot_length']
            np.testing.assert_allclose(hold[0,s+2]-ground[0,s+2],expected,atol=.05)

    def test_all_trials_surface_anchors_and_no_target_snapping(self):
        root=Path(__file__).resolve().parents[1]
        paths=json.loads((root/'configs/personalized_results.json').read_text())
        for name,path in paths.items():
            with self.subTest(trial=name):
                run=root/path;summary=json.loads((run/'summary.json').read_text())
                with np.load(run/summary['selected_candidate']/'trajectory.npz',allow_pickle=False) as z:
                    g={k:z[k][:3] for k in z.files if z[k].ndim and len(z[k])==summary['frame_count']}
                V=self.avatar.pose(g);points=self.avatar.contact_points(V,g['ground'])
                np.testing.assert_allclose(points,g['contact_points'],atol=1e-7)
                g['contacts']+=30
                np.testing.assert_array_equal(V,self.avatar.pose(g))
                self.assertGreater(np.linalg.norm(points-g['contacts'],axis=-1).max(),20)

    def test_scaled_user_profile_changes_surface_and_anchors(self):
        a=AnnyAvatar(load_user(height_cm=185))
        self.assertAlmostEqual(a.measurements['height_cm'],185,places=6)
        self.assertAlmostEqual(a.measurements['arm_span_cm'],176*185/175,delta=.1)
        p=a.contact_points(a.pose(a.neutral),np.zeros((1,4),bool))
        np.testing.assert_allclose(p[:,:2],a.neutral['palm_contacts'],atol=1e-8)

    def test_bad_segment_rotation_is_rejected(self):
        g=deepcopy(self.avatar.neutral);g['orientations'][0,1,0,0]=2
        with self.assertRaises(ValueError):self.avatar.pose(g)

    def test_changed_video_is_rejected_before_output_creation(self):
        import tempfile
        from beta_translator.avatar_pipeline import run_avatar
        from beta_translator.project_io import file_sha256
        root=Path(__file__).resolve().parents[1]
        source=root/json.loads((root/'configs/personalized_results.json').read_text())['trial1']
        def changed(path):
            return 'different-video' if str(path).endswith('videos/trial1.mp4') else file_sha256(path)
        with tempfile.TemporaryDirectory() as tmp, patch('beta_translator.avatar_pipeline.file_sha256',side_effect=changed):
            out=Path(tmp)/'new'
            with self.assertRaisesRegex(ValueError,'video/alignment'):
                run_avatar(root,source,output_root=out,render=False)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
