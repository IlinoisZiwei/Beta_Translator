"""Artifact integrity, baseline provenance and non-overwriting publication."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest,json
import numpy as np

from beta_translator.full_sequence import build_schedule
from beta_translator.gravity_sequence import solve_gravity_sequence
from beta_translator.support_optimization_pipeline import identity,publish,run_trial3_support_optimization,diagnose
from beta_translator.support_optimization import optimize_support_sequence
from tests.test_full_sequence import fixture


class SupportOptimizationPipelineTests(unittest.TestCase):
    def data(self,project):
        q,b,e,table,g=fixture(4);s=build_schedule(table,np.arange(4),g)
        initial=solve_gravity_sequence(q,np.ones((4,9)),e,s,b,30,settings={'max_nfev':30})
        inputs=dict(reference_outside=np.zeros(4,bool),fps=30.)
        old=project/'old';old.mkdir();(old/'summary.json').write_text('{"preserved":true}')
        baseline=dict(output_dir=str(old),body=b,physics=dict(mass_kg=70.,foot_friction=.8,max_force_component_n=70*9.81),settings=initial['settings'])
        return inputs,initial,baseline

    def render(self,inputs,result,d,path):
        files=[path/'candidate.mp4',path/'poster.png',path/'storyboard.png']
        for f in files:f.write_bytes(b'fixture')
        return files

    def test_cached_run_changes_with_settings_and_preserves_damaged_and_old_files(self):
        with TemporaryDirectory() as temp:
            project=Path(temp);inputs,initial,baseline=self.data(project)
            with patch('beta_translator.support_optimization_pipeline.load_initial',return_value=(inputs,initial,baseline)), \
                 patch('beta_translator.support_optimization_pipeline.provenance',return_value={'source':'a'}), \
                 patch('beta_translator.support_optimization_pipeline.render_gravity_sequence',side_effect=self.render):
                kwargs=dict(optimization={'rounds':1,'max_nfev':30},progress=None)
                first=run_trial3_support_optimization(project,**kwargs)
                again=run_trial3_support_optimization(project,**kwargs)
                self.assertEqual(first['output_dir'],again['output_dir'])
                Path(first['video_file']).write_bytes(b'edited artifact')
                other=run_trial3_support_optimization(project,**kwargs)
                self.assertNotEqual(first['output_dir'],other['output_dir'])
                self.assertEqual(Path(first['video_file']).read_bytes(),b'edited artifact')
                ident=identity(project,baseline,{'rounds':2,'max_nfev':30})
                self.assertNotEqual(first['provenance'],ident)
                self.assertEqual((project/'old/summary.json').read_text(),'{"preserved":true}')

    def test_input_change_blocks_publish(self):
        with TemporaryDirectory() as temp:
            project=Path(temp);inputs,initial,baseline=self.data(project)
            result=optimize_support_sequence(initial,30,optimization={'rounds':1,'max_nfev':20},progress=None)
            with patch('beta_translator.support_optimization_pipeline.provenance',side_effect=[{'source':'a'},{'source':'b'}]), \
                 patch('beta_translator.support_optimization_pipeline.render_gravity_sequence',side_effect=self.render):
                ident=identity(project,baseline)
                with self.assertRaisesRegex(RuntimeError,'changed'):publish(project,inputs,initial,result,ident,progress=None)
            self.assertFalse((project/'outputs/trial3/support_optimized').exists())

    def test_departure_load_is_separate_from_readiness_and_not_hidden(self):
        with TemporaryDirectory() as temp:
            inputs,r,_=self.data(Path(temp));s=r['schedule']
            s['labels'][3,0]='free';s['known'][3,0]=False;s['supported'][3,0]=False
            r['contact_forces_n'][2,0]=[0,100.]
            d=diagnose(r,inputs['reference_outside'],None)
            self.assertTrue(d['flags']['departing_load_remaining'][2])
            self.assertFalse(d['numerical_checks_pass'][2])
            self.assertTrue(np.isnan(d['departing_load_bw'][0]))

if __name__=='__main__':unittest.main()
