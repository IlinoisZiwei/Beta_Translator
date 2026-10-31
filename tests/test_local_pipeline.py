"""Regression checks for the local helpers and saved-input boundaries."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
import pandas as pd

from beta_translator.inverse_kinematics import RATIOS, LIMBS, make_body, forward_kinematics, solve_inverse_kinematics
from beta_translator.pose_adapter import to_frame0, image_to_wall_cm, wall_cm_to_image
from beta_translator.contact_matching import CONTACT_LANDMARKS, match_limb_contacts
from beta_translator.project_io import (
    file_sha256, save_alignment, load_alignment, save_reviewed_holds,
    load_reviewed_holds, restore_reviewed_map, save_annotations, load_annotations,
    validate_reference_alignment,
)


class IKTests(unittest.TestCase):
    def test_custom_ratios_and_invalid_lengths(self):
        custom = {**RATIOS, 'forearm': 0.15}
        self.assertEqual(make_body(160, custom)['forearm'], 24)
        self.assertEqual(make_body(160)['forearm'], 160 * RATIOS['forearm'])
        for height, ratios in [(0, RATIOS), (160, {'forearm': 0.1}), (160, {**RATIOS, 'torso': -1})]:
            with self.assertRaises(ValueError):
                make_body(height, ratios)

    def test_matching_body_preserves_known_contacts(self):
        body = make_body(170)
        reference = np.array([10, 100, 0.1, 1, 0.4, -1, -0.4, 0.2, 0.3, -0.2, -0.3])
        joints = forward_kinematics(reference, body)
        targets = {limb: joints.limbs[limb].end for limb in LIMBS}
        _, errors = solve_inverse_kinematics(body, targets, reference)
        self.assertLess(max(errors.values()), 1e-7)

    def test_out_of_bounds_initial_bend_is_clipped(self):
        reference = np.zeros(11)
        reference[4] = np.deg2rad(170)
        pose, _ = solve_inverse_kinematics(make_body(160), {limb: None for limb in LIMBS}, reference)
        self.assertLessEqual(abs(pose[4]), np.deg2rad(150) + 1e-8)

    def test_nonconvergence_is_reported(self):
        with patch('beta_translator.inverse_kinematics.least_squares',
                   return_value=SimpleNamespace(success=False, message='iteration limit')):
            with self.assertRaisesRegex(RuntimeError, 'iteration limit'):
                solve_inverse_kinematics(make_body(160), {}, np.zeros(11))

    def test_adapter_import_and_coordinate_roundtrip(self):
        original = np.array([[10., 20.], [30., 50.]])
        H = np.array([[1., 0., 15.], [0., 1., -5.], [0., 0., 1.]])
        np.testing.assert_allclose(to_frame0(original + [15, -5], {0: (H, 100, 'ok')}, 0), original)
        np.testing.assert_allclose(wall_cm_to_image(image_to_wall_cm(original, 2, 200), 2, 200), original)


class SavedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video, self.poses = self.root / 'video.mp4', self.root / 'poses.csv'
        self.video.write_bytes(b'video identity')
        self.poses.write_text('frame,x\n0,10\n')
        self.cache = self.root / 'alignment.npz'
        self.transforms = {0: (np.eye(3), 100, 'ok')}
        self.settings = dict(nfeatures=4000, wall_roi=None)

    def test_cache_roundtrip_and_changed_inputs(self):
        save_alignment(self.cache, self.transforms, self.video, self.poses, self.settings)
        loaded = load_alignment(self.cache, self.video, self.poses, self.settings)
        np.testing.assert_array_equal(loaded[0][0], np.eye(3))
        self.assertIsNone(load_alignment(self.cache, self.video, self.poses, dict(nfeatures=2000)))
        self.poses.write_text('frame,x\n0,11\n')
        self.assertIsNone(load_alignment(self.cache, self.video, self.poses, self.settings))
        save_alignment(self.cache, self.transforms, self.video, self.poses, self.settings)
        self.video.write_bytes(b'another video')
        self.assertIsNone(load_alignment(self.cache, self.video, self.poses, self.settings))

    def test_legacy_cache_requires_recomputation(self):
        np.savez(self.cache, frames=[0], matrices=[np.eye(3)], inliers=[100], statuses=['ok'],
                 video_sha256=file_sha256(self.video), pose_sha256=file_sha256(self.poses))
        self.assertIsNone(load_alignment(self.cache, self.video, self.poses, self.settings))

    def test_reference_change_invalidates_alignment_but_legacy_zero_is_compatible(self):
        save_alignment(self.cache, self.transforms, self.video, self.poses, self.settings)
        self.assertIsNotNone(load_alignment(self.cache, self.video, self.poses,
            dict(self.settings, reference_frame_number=0)))
        self.assertIsNone(load_alignment(self.cache, self.video, self.poses,
            dict(self.settings, reference_frame_number=450)))

    def test_nonzero_reference_map_and_alignment_must_agree(self):
        path = self.root / 'map.npz'
        mask = np.ones((8, 8), dtype=bool)
        save_reviewed_holds(path, self.video, np.zeros((8, 8, 3), np.uint8),
            [{'hold_id': 'H01'}], {'H01': mask}, 'later-map', ['H01'], reference_frame_number=450)
        self.assertEqual(load_reviewed_holds(path, self.video, reference_frame_number=450)[2], 'later-map')
        with self.assertRaisesRegex(ValueError, 'another reference'):
            load_reviewed_holds(path, self.video, reference_frame_number=0)
        save_alignment(self.cache, self.transforms, self.video, self.poses, self.settings)
        with self.assertRaisesRegex(ValueError, 'different reference'):
            validate_reference_alignment(path, self.cache)
        save_alignment(self.cache, self.transforms, self.video, self.poses,
            dict(self.settings, reference_frame_number=450))
        self.assertEqual(validate_reference_alignment(path, self.cache), 450)

    def test_invalid_reference_indices_are_rejected(self):
        for value in (-1, 1.5, True, '450'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                save_reviewed_holds(self.root/'bad.npz', self.video, np.zeros((8, 8, 3), np.uint8),
                    [{'hold_id':'H01'}], {'H01':np.ones((8, 8), bool)}, 'map', ['H01'],
                    reference_frame_number=value)

    def test_hold_restore_preserves_ids_without_replacing_poses(self):
        hold_map = self.root / 'map.npz'
        mask = np.ones((8, 8), dtype=bool)
        save_reviewed_holds(hold_map, self.video, np.zeros((8, 8, 3), np.uint8),
                            [{'hold_id': 'H44'}], {'H44': mask}, 'review1', ['H44'])
        records, _, run, selected = load_reviewed_holds(hold_map, self.video)
        self.assertEqual((records[0]['hold_id'], run, selected), ('H44', 'review1', ['H44']))
        bundle = self.root / 'inputs.zip'
        with zipfile.ZipFile(bundle, 'w') as archive:
            archive.write(hold_map, 'reviewed_hold_map.npz')
            archive.writestr('pose_coordinates.csv', 'obsolete poses')
        before = self.poses.read_bytes()
        restore_reviewed_map(bundle, self.root / 'restored.npz', self.video)
        self.assertEqual(self.poses.read_bytes(), before)
        self.video.write_bytes(b'changed video')
        with self.assertRaises(ValueError):
            restore_reviewed_map(bundle, self.root / 'wrong.npz', self.video)
        self.assertFalse((self.root / 'wrong.npz').exists())

    def test_hold_restore_from_organized_bundle(self):
        hold_map = self.root / 'map.npz'
        save_reviewed_holds(hold_map, self.video, np.zeros((8, 8, 3), np.uint8),
                            [{'hold_id': 'H44'}], {'H44': np.ones((8, 8), bool)}, 'map1', ['H44'])
        bundle = self.root / 'organized.zip'
        with zipfile.ZipFile(bundle, 'w') as archive:
            archive.write(hold_map, 'state/reviewed_hold_map.npz')
        restored = self.root / 'outputs/state/reviewed_hold_map.npz'
        restore_reviewed_map(bundle, restored, self.video)
        self.assertEqual(restored.read_bytes(), hold_map.read_bytes())
        self.assertEqual(load_reviewed_holds(restored, self.video)[3], ['H44'])

    def test_manual_annotations_roundtrip_and_stale_route(self):
        data = dict(video_sha256=file_sha256(self.video), candidate_run_id='map1', selected_hold_ids=['H44'],
                    frames=dict(before=1, after=9),
                    contacts={phase: {limb: 'unknown' for limb in LIMBS} for phase in ('before', 'after')},
                    moving_limb='left_hand', notes='Occluded foot',
                    body=dict(height_cm=160, ratios=RATIOS, px_per_cm=1.8))
        data['contacts']['before']['left_hand'] = 'H44'
        path = self.root / 'annotations.json'
        save_annotations(path, data, self.video, 'map1', ['H44'], 10)
        self.assertEqual(load_annotations(path, self.video, 'map1', ['H44'], 10), data)
        with self.assertRaises(ValueError):
            load_annotations(path, self.video, 'map2', ['H44'], 10)
        data['frames']['after'] = 0
        with self.assertRaises(ValueError):
            save_annotations(path, data, self.video, 'map1', ['H44'], 10)


class ContactTests(unittest.TestCase):
    def test_visibility_gaps_short_runs_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'short.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(12):
                writer.write(np.zeros((240, 320, 3), np.uint8))
            writer.release()
            mask = np.zeros((240, 320), np.uint8)
            mask[95:106, 95:106] = 1
            rows = []
            for frame in range(10):
                row = {'frame': frame}
                for name in CONTACT_LANDMARKS.values():
                    row.update({name+'_x_px': 100., name+'_y_px': 100., name+'_visibility': 0.1 if frame == 5 else 1.})
                rows.append(row)
            transforms = {f: (np.eye(3), 100, 'ok') for f in range(12)}
            contacts, events, _ = match_limb_contacts(pd.DataFrame(rows), {'H01': mask}, transforms, video)
            self.assertEqual(len(events), 4)
            self.assertEqual(set(events.end_frame), {4})
            self.assertEqual(set(contacts.loc[contacts.frame == 5, 'status']), {'low_visibility'})
            self.assertEqual(set(contacts.loc[contacts.frame == 6, 'status']), {'tentative'})
            self.assertEqual(set(contacts.loc[contacts.frame == 10, 'status']), {'missing_pose'})
            contacts, events, _ = match_limb_contacts(pd.DataFrame(rows), {'H01': mask, 'H02': mask}, transforms, video)
            self.assertTrue(events.empty)
            self.assertEqual(set(contacts.loc[contacts.frame == 0, 'status']), {'ambiguous'})


if __name__ == '__main__':
    unittest.main()
