"""Regression checks for independent-video timing and provenance."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from beta_translator.inverse_kinematics import RATIOS
from beta_translator.hold_mapping import filter_observed_candidates
from beta_translator.trial_pipeline import nearest_source_indices, checkpoint, valid_cache, geometry_status


class TrialPipelineTests(unittest.TestCase):
    def test_variable_rate_frames_are_selected_from_actual_timestamps(self):
        source=np.array([0.,.014,.035,.048,.071,.085,.102])
        target=np.arange(4)/30
        indices=nearest_source_indices(source,target)
        np.testing.assert_array_equal(indices,[0,2,4,6])
        self.assertLess(np.max(np.abs(source[indices]-target)), .008)

    def test_duplicate_or_nonmonotonic_frame_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            nearest_source_indices([0,.1,.2],np.arange(7)/30)
        with self.assertRaises(ValueError):
            nearest_source_indices([0,.1,.1],[0,.1])

    def test_modified_output_or_input_invalidates_resume_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);artifact=root/'result';artifact.write_text('original')
            checkpoint(root/'manifest.json',{'video':'a'},[artifact])
            self.assertIsNotNone(valid_cache(root/'manifest.json',{'video':'a'}))
            self.assertIsNone(valid_cache(root/'manifest.json',{'video':'b'}))
            artifact.write_text('changed')
            self.assertIsNone(valid_cache(root/'manifest.json',{'video':'a'}))

    def test_reconstruction_holes_cannot_become_hold_candidates(self):
        unresolved=np.zeros((20,20),bool);unresolved[:10,:10]=True
        candidates=[dict(bbox=[0,0,10,10],mask=np.ones((10,10),bool),area=100,score=.99),
                    dict(bbox=[10,10,20,20],mask=np.ones((10,10),bool),area=100,score=.8)]
        retained=filter_observed_candidates(candidates,unresolved)
        self.assertEqual(len(retained),1)
        self.assertEqual(retained[0]['bbox'],[10,10,20,20])
        self.assertEqual(retained[0]['observed_pixel_fraction'],1.)

    def test_new_trial_does_not_inherit_source_height_or_wall_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video=root/'video';video.write_bytes(b'new trial')
            result=geometry_status(video,root,190,RATIOS)
            self.assertIsNone(result['source_height_cm'])
            self.assertIsNone(result['px_per_cm'])
            self.assertFalse(result['inherited_trial2_geometry'])
            self.assertEqual(len(result['blockers']),2)
            with self.assertRaisesRegex(ValueError,'source of the wall scale'):
                geometry_status(video,root,190,RATIOS,2.,True)


if __name__=='__main__':unittest.main()
