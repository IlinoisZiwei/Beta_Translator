"""Gravity runs preserve older outputs and invalidate changed input/code/assets."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from beta_translator.full_sequence import build_schedule
from beta_translator.gravity_sequence_pipeline import run_trial3_gravity_sequence
from tests.test_full_sequence import fixture


class GravityPipelineTests(unittest.TestCase):
    def inputs(self):
        q,body,ends,table,geometry=fixture(4)
        return dict(reference=q,visibility=np.ones((4,9)),reference_endpoints=ends,
            schedule=build_schedule(table,np.arange(4),geometry),fps=30.,reference_outside=np.zeros(4,bool),
            geometry={**geometry,'candidate_run_id':'test-map','reference_frame_number':334},
            prediction={'accepted_frame_fraction':.8})

    def render(self,inputs,result,diagnostic,path):
        files=[path/f'trial3_gravity_{result["body"]["height_cm"]:g}cm_full_sequence.mp4',
               path/'poster.png',path/'storyboard.png']
        for f in files:f.write_bytes(b'test artifact')
        return files

    def test_configurable_height_cache_and_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory)
            old=project/'outputs/trial3/full_sequence/old.mp4'
            old.parent.mkdir(parents=True);old.write_bytes(b'original geometric result')
            with patch('beta_translator.gravity_sequence_pipeline.load_full_sequence_inputs',return_value=self.inputs()), \
                 patch('beta_translator.gravity_sequence_pipeline.render_gravity_sequence',side_effect=self.render) as render, \
                 patch('beta_translator.gravity_sequence_pipeline.gravity_provenance',return_value={'inputs':'a'}):
                first=run_trial3_gravity_sequence(project,progress=None)
                again=run_trial3_gravity_sequence(project,progress=None)
                self.assertEqual(first['output_dir'],again['output_dir'])
                self.assertEqual(render.call_count,1)
                other=run_trial3_gravity_sequence(project,height_cm=165,progress=None)
                self.assertEqual(other['body']['height_cm'],165)
                self.assertIn('165cm',other['video_file'])
                self.assertNotEqual(first['output_dir'],other['output_dir'])
                audit_change=run_trial3_gravity_sequence(project,progress=None,
                    audit_config={'com_perturbation_cm':3.})
                self.assertNotEqual(first['output_dir'],audit_change['output_dir'])
                self.assertEqual(audit_change['support_audit_settings']['com_perturbation_cm'],3.)
                Path(first['video_file']).write_bytes(b'user changed result')
                replacement=run_trial3_gravity_sequence(project,progress=None)
                self.assertNotEqual(first['output_dir'],replacement['output_dir'])
                self.assertEqual(Path(first['video_file']).read_bytes(),b'user changed result')
                self.assertEqual(old.read_bytes(),b'original geometric result')

    def test_input_change_blocks_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('beta_translator.gravity_sequence_pipeline.load_full_sequence_inputs',return_value=self.inputs()), \
                 patch('beta_translator.gravity_sequence_pipeline.render_gravity_sequence',side_effect=self.render), \
                 patch('beta_translator.gravity_sequence_pipeline.gravity_provenance',side_effect=[{'inputs':'a'},{'inputs':'b'}]):
                with self.assertRaisesRegex(RuntimeError,'changed'):
                    run_trial3_gravity_sequence(directory,progress=None)
            self.assertFalse((Path(directory)/'outputs/trial3/gravity_sequence').exists())


if __name__=='__main__':unittest.main()
