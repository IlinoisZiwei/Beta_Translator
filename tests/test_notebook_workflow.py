"""Review exports preserve evidence and cannot silently confirm model drafts."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from notebooks.trial_workflow import Evidence, ContactReview
from beta_translator.trials import load_config
from beta_translator.project_io import file_sha256, load_reviewed_holds

ROOT = Path(__file__).resolve().parents[1]


class ReviewWorkflowTests(unittest.TestCase):
    def test_all_trial_route_exports_preserve_originals(self):
        for trial in ['trial1', 'trial2', 'trial3']:
            with self.subTest(trial=trial), tempfile.TemporaryDirectory() as tmp:
                evidence = Evidence(ROOT, load_config(ROOT, trial))
                original = file_sha256(evidence.map_path)
                evidence.selected = evidence.selected[:-1]
                folder = evidence.export_route(tmp)
                _, masks, _, selected = load_reviewed_holds(folder/'reviewed_hold_map.npz', evidence.video)
                self.assertEqual(selected, evidence.selected)
                self.assertEqual(set(masks), set(evidence.masks))
                self.assertEqual(original, file_sha256(evidence.map_path))

    def test_automatic_timing_remains_a_draft_for_all_trials(self):
        with patch('notebooks.trial_workflow.display'), redirect_stdout(StringIO()):
            for trial in ['trial1', 'trial2', 'trial3']:
                with self.subTest(trial=trial), tempfile.TemporaryDirectory() as tmp:
                    evidence = Evidence(ROOT, load_config(ROOT, trial))
                    review = ContactReview(evidence, 175)
                    review.prefill()
                    self.assertFalse(review.confirm.value)
                    path = review.export(tmp)
                    saved = json.loads(path.read_text())
                    self.assertNotIn('motion', saved)
                    self.assertFalse(saved['suggested_motion']['support_contacts_maintained'])
                    self.assertEqual(saved['body']['height_cm'], 175)
                    review.confirm.value = True
                    field = review.fields['before']['left_hand']
                    field.value = 'free' if field.value != 'free' else 'unknown'
                    self.assertFalse(review.confirm.value)
                    self.assertEqual(review.release.value, -1)
                    self.assertEqual(review.arrival.value, -1)

    def test_stale_route_cannot_use_automatic_draft(self):
        with patch('notebooks.trial_workflow.display'), redirect_stdout(StringIO()):
            e = Evidence(ROOT, load_config(ROOT, 'trial1'))
            r = ContactReview(e, 175)
            e.selected = e.selected[:-1]
            with self.assertRaisesRegex(ValueError, 'Route draft changed'):
                r.prefill()
            r = ContactReview(e, 175)
            with self.assertRaisesRegex(ValueError, 'Predictions belong to the saved route'):
                r.prefill()

    def test_loading_old_annotations_keeps_current_height(self):
        with patch('notebooks.trial_workflow.display'), redirect_stdout(StringIO()):
            e = Evidence(ROOT, load_config(ROOT, 'trial1'))
            r = ContactReview(e, 175)
            r.load_saved()
            self.assertEqual(r.state()['body']['height_cm'], 175)
            old = json.loads((e.root/'state/endpoint_annotations.json').read_text())
            self.assertEqual(r.frames, old['frames'])

    def test_invalid_draft_timing_creates_no_export(self):
        with patch('notebooks.trial_workflow.display'), redirect_stdout(StringIO()), tempfile.TemporaryDirectory() as tmp:
            e = Evidence(ROOT, load_config(ROOT, 'trial1'))
            r = ContactReview(e, 175)
            r.prefill();r.arrival.value = e.count + 1
            with self.assertRaisesRegex(ValueError, 'valid frame number'):
                r.export(tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
