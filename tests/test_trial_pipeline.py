"""Regression checks for independent-video timing and provenance."""
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from beta_translator.inverse_kinematics import RATIOS
from beta_translator.project_io import save_reviewed_holds, load_reviewed_holds, file_sha256
from beta_translator.hold_mapping import filter_observed_candidates
from beta_translator.trial_pipeline import nearest_source_indices, checkpoint, valid_cache, geometry_status, select_route, add_manual_holds


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

    def test_route_selection_preserves_geometry_and_rejects_stale_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'state').mkdir()
            video=root/'video';video.write_bytes(b'video')
            reference=np.zeros((20,20,3),np.uint8)
            masks={'A':np.eye(20,dtype=bool),'B':np.fliplr(np.eye(20,dtype=bool))}
            records=[dict(hold_id=k,cx_px=10.,cy_px=10.) for k in masks]
            path=root/'state/reviewed_hold_map.npz'
            save_reviewed_holds(path,video,reference,records,masks,'map-one',['A','B'])
            manifest=root/'state/scene_manifest.json'
            checkpoint(manifest,{'video':'test'},[path],candidate_count=2)
            select_route(video,root,['B'],'map-one')
            _,loaded,run_id,selected=load_reviewed_holds(path,video)
            self.assertEqual(selected,['B']);self.assertEqual(run_id,'map-one')
            self.assertEqual(set(loaded),{'A','B'})
            for key in masks:np.testing.assert_array_equal(loaded[key],masks[key])
            self.assertIsNotNone(valid_cache(manifest,{'video':'test'}))
            digest=file_sha256(path)
            select_route(video,root,['B'],'map-one')
            self.assertEqual(digest,file_sha256(path))
            with self.assertRaisesRegex(ValueError,'another candidate map'):
                select_route(video,root,['A'],'old-map')
            with self.assertRaisesRegex(ValueError,'unique IDs'):
                select_route(video,root,['missing'],'map-one')
            self.assertEqual(digest,file_sha256(path))

    def test_manual_hold_preserves_original_map_and_rejects_stale_or_colliding_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'state').mkdir()
            video=root/'video';video.write_bytes(b'video')
            reference=np.zeros((20,20,3),np.uint8)
            original=np.eye(20,dtype=bool)
            path=root/'state/reviewed_hold_map.npz'
            save_reviewed_holds(path,video,reference,[dict(hold_id='H001',cx_px=10.,cy_px=10.)],
                                {'H001':original},'map-one',['H001'])
            manifest=root/'state/scene_manifest.json'
            checkpoint(manifest,{'video':'test'},[path],candidate_count=1)
            annotation=dict(video_sha256=file_sha256(video),candidate_run_id='map-one',
                reference_frame_number=0,image_size=[20,20],
                holds=[dict(hold_id='H002',polygon_xy=[[10,2],[15,2],[15,6],[10,6]])])
            patch=root/'manual.json';patch.write_text(json.dumps(annotation))
            result=add_manual_holds(video,root,patch)
            records,masks,run_id,selected=load_reviewed_holds(path,video)
            self.assertEqual((run_id,selected),('map-one',['H001']))
            np.testing.assert_array_equal(masks['H001'],original)
            self.assertEqual(result['candidate_count'],2)
            self.assertIsNone(records[-1]['score'])
            self.assertEqual(records[-1]['source'],'user_identified_manual_outline')
            self.assertIsNotNone(valid_cache(manifest,{'video':'test'}))
            digest=file_sha256(path)
            add_manual_holds(video,root,patch)
            self.assertEqual(file_sha256(path),digest)
            annotation['candidate_run_id']='old-map';patch.write_text(json.dumps(annotation))
            with self.assertRaisesRegex(ValueError,'another video, candidate map'):
                add_manual_holds(video,root,patch)
            annotation['candidate_run_id']='map-one'
            annotation['holds'][0]['hold_id']='H001';patch.write_text(json.dumps(annotation))
            with self.assertRaisesRegex(ValueError,'already exists'):
                add_manual_holds(video,root,patch)
            self.assertEqual(file_sha256(path),digest)

    def test_qualitative_line_plane_observation_does_not_supply_metric_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video=root/'video';video.write_bytes(b'video')
            observations=[dict(relation='line_parallel_to_plane',source='user_statement')]
            result=geometry_status(video,root,190,RATIOS,observations=observations)
            self.assertEqual(result['qualitative_observations'],observations)
            self.assertIsNone(result['px_per_cm'])
            self.assertFalse(result['single_panel_confirmed'])
            self.assertEqual(len(result['blockers']),2)

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
