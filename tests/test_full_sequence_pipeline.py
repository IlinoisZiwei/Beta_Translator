"""New runs must never overwrite the existing Trial 3 experiment."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from beta_translator.full_sequence import build_schedule
from beta_translator.full_sequence_pipeline import run_full_sequence, _intervals
from tests.test_full_sequence import fixture


class FullSequencePipelineTests(unittest.TestCase):
    def inputs(self):
        q, _, ends, table, geometry = fixture(4)
        return dict(reference=q, visibility=np.ones((4,9)), reference_endpoints=ends,
            schedule=build_schedule(table, np.arange(4), geometry), fps=30.,
            reference_outside=np.zeros(4,bool),
            geometry={**geometry,'candidate_run_id':'test-map','reference_frame_number':0},
            prediction={'accepted_frame_fraction':.8})

    @staticmethod
    def render(inputs, result, diagnostic, path):
        video_name = f'trial3_target_{result["body"]["height_cm"]:g}cm_full_sequence.mp4'
        files = [path/name for name in [video_name,'poster.png','storyboard.png']]
        for f in files:
            f.write_bytes(b'test artifact')
        return files

    def test_reuse_and_changed_inputs_preserve_previous_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            sentinel = project/'outputs/trial3/videos/old.mp4'
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b'old result')
            with patch('beta_translator.full_sequence_pipeline.load_full_sequence_inputs',return_value=self.inputs()), \
                 patch('beta_translator.full_sequence_pipeline.render_full_sequence',side_effect=self.render) as render, \
                 patch('beta_translator.full_sequence_pipeline.input_provenance',return_value={'data':'a'}) as provenance:
                first = run_full_sequence(project, progress=None)
                again = run_full_sequence(project, progress=None)
                self.assertEqual(render.call_count,1)
                self.assertEqual(first['output_dir'],again['output_dir'])
                provenance.return_value = {'data':'b'}
                second = run_full_sequence(project, progress=None)
                self.assertNotEqual(first['output_dir'],second['output_dir'])
                self.assertTrue(Path(first['video_file']).is_file())
                self.assertEqual(sentinel.read_bytes(),b'old result')
                # A damaged output also gets a new directory, not an overwrite.
                Path(second['video_file']).write_bytes(b'user changed this')
                third = run_full_sequence(project, progress=None)
                self.assertNotEqual(second['output_dir'],third['output_dir'])
                self.assertEqual(Path(second['video_file']).read_bytes(),b'user changed this')

    def test_height_changes_body_video_name_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            with patch('beta_translator.full_sequence_pipeline.load_full_sequence_inputs', return_value=self.inputs()), \
                 patch('beta_translator.full_sequence_pipeline.render_full_sequence', side_effect=self.render) as render, \
                 patch('beta_translator.full_sequence_pipeline.input_provenance', return_value={'data': 'a'}):
                first = run_full_sequence(project, height_cm=180, progress=None)
                changed = run_full_sequence(project, height_cm=170, progress=None)
                again = run_full_sequence(project, height_cm=170, progress=None)
                self.assertEqual(render.call_count, 2)
                self.assertEqual(changed['body']['height_cm'], 170)
                self.assertEqual(Path(changed['video_file']).name, 'trial3_target_170cm_full_sequence.mp4')
                self.assertEqual(changed['output_dir'], again['output_dir'])
                self.assertNotEqual(first['output_dir'], changed['output_dir'])
                self.assertTrue(Path(first['video_file']).is_file())
                self.assertTrue(Path(changed['video_file']).is_file())
                with np.load(Path(changed['output_dir']) / 'trajectory.npz') as z:
                    self.assertEqual(len(z['frames']), 4)

    def test_input_changes_during_solve_block_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            with patch('beta_translator.full_sequence_pipeline.load_full_sequence_inputs',return_value=self.inputs()), \
                 patch('beta_translator.full_sequence_pipeline.render_full_sequence',side_effect=self.render), \
                 patch('beta_translator.full_sequence_pipeline.input_provenance',side_effect=[{'data':'a'},{'data':'b'}]):
                with self.assertRaisesRegex(RuntimeError,'Inputs changed'):
                    run_full_sequence(project, progress=None)
            self.assertFalse((project/'outputs/trial3/full_sequence').exists())

    def test_interval_endpoints_are_inclusive(self):
        rows = _intervals({'flags':{'a':np.array([True,True,False,True])}},30)
        self.assertEqual([(r['start_frame'],r['end_frame']) for r in rows],[(0,1),(3,3)])
        self.assertAlmostEqual(rows[-1]['end_exclusive_sec'],4/30)


if __name__ == '__main__':
    unittest.main()
