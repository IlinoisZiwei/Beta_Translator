"""Review loading rejects stale inference and preserves the manual IK gate."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from beta_translator.contact_review import load_review_bundle
from beta_translator.inverse_kinematics import LIMBS
from beta_translator.project_io import file_sha256


class ContactReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'outputs'
        self.package = self.root / 'package'
        self.destination = self.root / 'prediction'
        for folder in (self.output / 'state', self.package, self.destination):
            folder.mkdir(parents=True)
        self.video = self.root / 'video.mp4'
        self.pose = self.root / 'pose.npz'
        self.inputs = {'video_sha256': self.video, 'pose_cache_sha256': self.pose,
            'alignment_sha256': self.output / 'state/alignment_cache.npz',
            'reviewed_map_sha256': self.output / 'state/reviewed_hold_map.npz',
            'model_config_sha256': self.package / 'deployment.json'}
        for path in self.inputs.values(): path.write_bytes(b'test input')
        np.savez(self.inputs['reviewed_map_sha256'],
                 metadata=json.dumps({'reference_frame_number': 0}))
        np.savez(self.inputs['alignment_sha256'], settings=json.dumps({}))
        (self.package / 'model.pt').write_bytes(b'model')
        self.inputs['model_config_sha256'].write_text(json.dumps({'checkpoints': [
            {'file': 'model.pt', 'sha256': file_sha256(self.package / 'model.pt')}]}))
        pd.DataFrame([dict(frame=frame, limb=limb, hold_id='H44')
                      for frame in range(3) for limb in LIMBS]).to_csv(
                          self.destination / 'limb_contacts.csv', index=False)
        (self.destination / 'contact_events.csv').write_text('limb,hold_id\nleft_hand,H44\n')
        self.prediction = {key: file_sha256(path) for key, path in self.inputs.items()}
        self.prediction.update(candidate_run_id='map', selected_hold_ids=['H44'])
        self.seal()
        self.mock_map = patch('beta_translator.contact_review.load_reviewed_holds',
                              return_value=([], {}, 'map', ['H44']))
        self.mock_map.start(); self.addCleanup(self.mock_map.stop)

    def seal(self):
        self.prediction['artifact_sha256'] = {name: file_sha256(self.destination / name)
            for name in ('limb_contacts.csv', 'contact_events.csv')}
        (self.destination / 'automatic_contacts.json').write_text(json.dumps(self.prediction))

    def load(self):
        return load_review_bundle(self.video, self.output, self.pose, self.package, self.destination)

    def test_matching_inputs_load(self):
        _, contacts = self.load()
        self.assertEqual(len(contacts), 12)

    def test_wall_contact_is_a_valid_model_state(self):
        path = self.destination / 'limb_contacts.csv'
        rows = pd.read_csv(path)
        rows.loc[0, 'hold_id'] = 'wall_contact'
        rows.to_csv(path, index=False)
        self.seal()
        _, loaded = self.load()
        self.assertEqual(loaded.loc[0, 'hold_id'], 'wall_contact')

    def test_prediction_reference_must_match_map_and_alignment(self):
        self.prediction['reference_frame_number'] = 450
        self.seal()
        with self.assertRaisesRegex(ValueError, 'another reference'):
            self.load()

    def test_matching_nonzero_reference_loads(self):
        np.savez(self.inputs['reviewed_map_sha256'],
                 metadata=json.dumps({'reference_frame_number': 450}))
        np.savez(self.inputs['alignment_sha256'],
                 settings=json.dumps({'reference_frame_number': 450}))
        self.prediction.update({key: file_sha256(path) for key, path in self.inputs.items()})
        self.prediction['reference_frame_number'] = 450
        self.seal()
        prediction, _ = self.load()
        self.assertEqual(prediction['reference_frame_number'], 450)

    def test_changed_video_map_alignment_or_pose_rejected(self):
        for name in ('video_sha256', 'reviewed_map_sha256', 'alignment_sha256', 'pose_cache_sha256'):
            with self.subTest(name=name):
                path = self.inputs[name]; original = path.read_bytes(); path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Stale'): self.load()
                path.write_bytes(original)

    def test_changed_model_checkpoint_rejected(self):
        (self.package / 'model.pt').write_bytes(b'changed model')
        with self.assertRaisesRegex(ValueError, 'checkpoint'): self.load()

    def test_modified_contacts_rejected(self):
        with (self.destination / 'limb_contacts.csv').open('a') as stream:
            stream.write('3,left_hand,H44\n')
        with self.assertRaisesRegex(ValueError, 'Unverified'): self.load()

    def test_missing_limb_rows_rejected_even_with_matching_hash(self):
        path = self.destination / 'limb_contacts.csv'
        pd.read_csv(path).iloc[1:].to_csv(path, index=False); self.seal()
        with self.assertRaisesRegex(ValueError, 'four limbs'): self.load()

    def test_duplicate_rows_rejected(self):
        path = self.destination / 'limb_contacts.csv'; rows = pd.read_csv(path)
        pd.concat([rows, rows.iloc[:1]]).to_csv(path, index=False); self.seal()
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.load()


if __name__ == '__main__': unittest.main()
