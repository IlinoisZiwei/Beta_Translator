"""Critical data, model symmetry, frame indexing and review-gate checks."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import pandas as pd

torch.set_num_threads(1)

from beta_translator.contact_features import build_features
from beta_translator.contact_network import ContactTCN, predict_logits
from beta_translator.contact_decoding import export_predictions
from beta_translator.wayup_data import read_labels, participant_id, SPLITS
from beta_translator.wholebody_pose import WholebodyExtractor
from beta_translator.automatic_contacts import motion_annotation_from_prediction, review_annotation_from_prediction
from beta_translator.contact_baseline import geometric_probabilities


class LearnedContactTests(unittest.TestCase):
    def test_temporal_baseline_votes_on_nominal_hold_ids(self):
        d = np.ones((15, 4, 3), np.float32)
        d[:, :, 0] = .01; d[7, :, 0] = 1; d[7, :, 2] = .01
        original = geometric_probabilities(d, .2, 5)
        permuted = geometric_probabilities(d[:, :, [2, 0, 1]], .2, 5)
        np.testing.assert_allclose(permuted[:, :, [1, 2, 0, 3, 4]], original)
        self.assertTrue((original.argmax(-1) == 0).all())

    def test_people_never_overlap_splits(self):
        self.assertEqual(participant_id('p2a'), participant_id('p2b'))
        groups = list(SPLITS.values())
        for i, a in enumerate(groups):
            for b in groups[i+1:]: self.assertFalse(set(a) & set(b))

    def test_conflicts_and_unknown_holds_are_not_free(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'labels.csv'
            with path.open('w') as stream:
                stream.write('person,startFrame,endFrame,framesOccluded\nlh 0,1,5,2-3\n'
                             'lh 1,4,8,\nrf (wrong hold),3,4,\nrf 1,1,10,\n')
            labels, occluded, events, audit = read_labels(path, 10, 2)
        self.assertTrue((labels[4:6, 0] == -100).all())
        self.assertTrue((labels[3:5, 3] == -100).all())
        self.assertEqual(labels[9, 0], -100)
        self.assertEqual(labels[9, 3], 1)
        self.assertTrue(occluded[2:4, 0].all())
        self.assertEqual(audit['conflicting_limb_frames'], 2)

    def test_hold_permutation_and_chunk_invariance(self):
        rng = np.random.default_rng(4)
        points = rng.normal(100, 20, (180, 133, 2)).astype(np.float32)
        scores = np.full((180, 133), 3, np.float32)
        boxes = np.array([[70, 80, 90, 100], [100, 100, 120, 120]])
        features, _ = build_features(points, scores, boxes, 25)
        shuffled, _ = build_features(points, scores, boxes[::-1], 25)
        np.testing.assert_allclose(shuffled[:, :, [1, 0, 2, 3]], features, atol=1e-5)
        torch.manual_seed(2)
        model = ContactTCN(features.shape[-1], width=12, layers=3, dropout=0).eval()
        original = predict_logits(model, features, 'cpu', chunk=71)
        flipped = predict_logits(model, shuffled, 'cpu', chunk=180)
        np.testing.assert_allclose(flipped[:, :, [1, 0, 2, 3]], original, atol=2e-5)

    def test_release_and_arrival_and_unknown_support_gate(self):
        states = np.zeros((20, 4), int)
        states[:, 1:] = 1
        states[7:12, 0] = 2; states[12:, 0] = 1
        p = np.full((20, 4, 4), .01)
        np.put_along_axis(p, states[..., None], .97, axis=-1)
        rows, events, moves = export_predictions(p, states, ['H44', 'H28'], 25, .9)
        self.assertEqual(moves[0]['release_frame'], 7)
        self.assertEqual(moves[0]['arrival_frame'], 12)
        self.assertTrue(moves[0]['eligible_for_ik'])
        p[9, 2] = .25
        _, _, moves = export_predictions(p, states, ['H44', 'H28'], 25, .9)
        self.assertFalse(moves[0]['eligible_for_ik'])
        self.assertEqual(moves[0]['support_contacts']['left_foot'], 'unknown')

    def test_detector_miss_does_not_switch_to_distant_bystander(self):
        class Capture:
            def get(self, key):
                return {3: 720, 4: 1280, 5: 25, 7: 10}[key]
            def read(self): return True, np.zeros((1280, 720, 3), np.uint8)
            def release(self): pass
        detections = iter([np.array([[200., 100., 400., 400.]]),
                           np.array([[550., 900., 700., 1200.]])])
        seen = []
        def pose(image, bboxes):
            box = np.asarray(bboxes[0]); seen.append(box.copy())
            points = np.zeros((1, 133, 2), np.float32)
            points[0, :, 0] = np.linspace(box[0]+20, box[2]-20, 133)
            points[0, :, 1] = np.linspace(box[1]+20, box[3]-20, 133)
            return points, np.full((1, 133), 3., np.float32)
        extractor = WholebodyExtractor.__new__(WholebodyExtractor)
        extractor.detector = lambda image: next(detections)
        extractor.pose = pose; extractor.provenance = {'tracking_version': 2}
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory)/'video.mp4'; video.write_bytes(b'test video')
            with patch('beta_translator.wholebody_pose.cv2.VideoCapture', return_value=Capture()):
                extractor.extract(video, Path(directory)/'pose.npz')
        self.assertTrue(all(box[1] < 500 for box in seen))

    def test_ik_adapter_rejects_uncertain_and_mismatched_inputs(self):
        with self.assertRaisesRegex(ValueError, 'review'):
            motion_annotation_from_prediction({}, {'eligible_for_ik': False}, {})
        move = {'eligible_for_ik': True, 'requires_review': False}
        with self.assertRaisesRegex(ValueError, 'different'):
            motion_annotation_from_prediction({'video_sha256': 'a'}, move, {'video_sha256': 'b'})

    def test_review_draft_does_not_confirm_motion(self):
        from beta_translator.motion_ik import validate_motion_annotation
        template = {'video_sha256': 'a', 'candidate_run_id': 'b', 'selected_hold_ids': ['H0', 'H1'],
                    'motion': {'release_frame': 4}}
        prediction = {k: template[k] for k in ('video_sha256', 'candidate_run_id', 'selected_hold_ids')}
        move = dict(before_frame=1, after_frame=9, moving_limb='left_hand', release_frame=4, arrival_frame=8)
        contacts = pd.DataFrame([dict(frame=f, limb=limb, hold_id='H0') for f in (1, 9)
                                 for limb in ('left_hand', 'right_hand', 'left_foot', 'right_foot')])
        draft = review_annotation_from_prediction(template, move, prediction, contacts)
        self.assertNotIn('motion', draft)
        self.assertIn('motion', template)
        self.assertFalse(draft['suggested_motion']['support_contacts_maintained'])
        with self.assertRaises(ValueError): validate_motion_annotation(draft)


if __name__ == '__main__': unittest.main()
