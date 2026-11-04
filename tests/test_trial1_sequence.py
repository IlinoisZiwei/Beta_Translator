"""Trial1 scale, native timing, reviewed-reference separation and export hooks."""
import json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from beta_translator.trial1_sequence import compare_confirmed_reference,load_inputs,TRIAL1_SCALE
from beta_translator.trial1_support_pipeline import trial1_diagnose,review_html
from beta_translator.gravity_sequence import solve_gravity_sequence
from beta_translator.full_sequence import build_schedule
from beta_translator.support_optimization import optimize_support_sequence
from beta_translator.support_optimization_pipeline import publish
from beta_translator.pose_adapter import image_to_wall_cm,project_plane
from tests.test_full_sequence import fixture

class Trial1Tests(unittest.TestCase):
    def test_scale_transform_matches_existing_trial1_convention(self):
        matrix=np.array([[1/TRIAL1_SCALE,0,0],[0,-1/TRIAL1_SCALE,1280/TRIAL1_SCALE],[0,0,1]])
        xy=np.array([[0,0],[100,300],[720,1280]],float)
        np.testing.assert_allclose(project_plane(xy,matrix),image_to_wall_cm(xy,TRIAL1_SCALE,1280),atol=1e-12)

    def test_different_video_rejects_scale(self):
        with TemporaryDirectory() as d,patch('beta_translator.trial1_sequence.file_sha256',return_value='different-video'):
            with self.assertRaisesRegex(ValueError,'supplied video'):load_inputs(d)

    def test_saved_review_respects_release_arrival_and_does_not_change_labels(self):
        q,b,e,table,g=fixture(6);s=build_schedule(table,np.arange(6),g)
        ref=dict(frames={'before':1,'after':5},moving_limb='left_hand',
            contacts={'before':dict(zip(['left_hand','right_hand','left_foot','right_foot'],['H0','H1','H2','H3'])),
                      'after':dict(zip(['left_hand','right_hand','left_foot','right_foot'],['H4','H1','H2','H3']))},
            motion=dict(release_frame=3,arrival_frame=5,support_contacts_maintained=True))
        original=s['labels'].copy();r=compare_confirmed_reference(s,ref)
        self.assertEqual(r[r.limb=='left_hand'].reviewed_contact.tolist(),['H0','H0','free','free','H4'])
        self.assertEqual(int((~r.agrees).sum()),3)
        np.testing.assert_array_equal(original,s['labels'])

    def test_geometry_and_review_uncertainty_cannot_become_clean_success(self):
        q,b,e,table,g=fixture(4);s=build_schedule(table,np.arange(4),g)
        r=solve_gravity_sequence(q,np.ones((4,9)),e,s,b,30,settings={'max_nfev':30})
        r['trial1_context']={'reference_disagreement':np.array([False,True,False,False])}
        d=trial1_diagnose(r,np.zeros(4,bool),None)
        self.assertEqual(d['counts']['uniform_scale_approximation'],4)
        self.assertEqual(d['counts']['reviewed_contact_disagreement'],1)
        self.assertFalse((d['status']=='within_numeric_tolerance').any())

    def test_trial1_publication_uses_separate_provenance_and_artifact_hook(self):
        q,b,e,table,g=fixture(4);s=build_schedule(table,np.arange(4),g)
        initial=solve_gravity_sequence(q,np.ones((4,9)),e,s,b,30,settings={'max_nfev':30})
        initial['trial1_context']={'reference_disagreement':np.zeros(4,bool)}
        result=optimize_support_sequence(initial,30,optimization={'rounds':1,'max_nfev':20},progress=None)
        inputs=dict(reference_outside=np.zeros(4,bool),fps=30.,trial_id='trial1')
        def render(a,b,c,path):
            files=[path/'raw.mp4',path/'poster.png',path/'storyboard.png']
            for f in files:f.write_bytes(b'fixture')
            return files
        def extra(path,*args):
            p=path/'review.html';p.write_text('review');return dict(files=[p],summary={'trial':'Trial1'})
        ident=dict(inputs={'trial1':'test'},physics={'mass_kg':70},support_audit_settings={})
        with TemporaryDirectory() as temp,patch('beta_translator.support_optimization_pipeline.render_gravity_sequence',side_effect=render):
            out=publish(temp,inputs,initial,result,ident,result_root=Path(temp)/'trial1',progress=None,
                input_provenance_fn=lambda _: {'trial1':'test'},diagnostic_fn=trial1_diagnose,artifact_builder=extra)
            self.assertIn('trial1_support_optimized',out['video_name']);self.assertEqual(out['trial'],'Trial1')
            self.assertIn('review.html',out['output_sha256'])
            self.assertEqual(out['frame_count'],4)

if __name__=='__main__':unittest.main()
