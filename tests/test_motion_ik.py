import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from beta_translator.inverse_kinematics import RATIOS, LIMBS, make_body, forward_kinematics
from beta_translator.motion_ik import (
    batch_joints, contact_schedule, validate_motion_annotation, solve_motion_sequence,
)
from beta_translator.motion_pipeline import provenance, current_motion_outputs, SUMMARY_NAME, INPUT_NAMES
from beta_translator.project_io import file_sha256, save_annotations, load_annotations


def fixture():
    before = dict(zip(LIMBS, ['H01','H02','H03','H04']))
    after = {**before, 'left_hand':'H05'}
    annotation = dict(frames=dict(before=0,after=8),contacts=dict(before=before,after=after),
        moving_limb='left_hand',selected_hold_ids=['H01','H02','H03','H04','H05'],
        body=dict(height_cm=165,ratios=dict(RATIOS),px_per_cm=2.),
        motion=dict(release_frame=3,arrival_frame=6,support_contacts_maintained=True))
    reference = np.array([0.,100.,.1,1.7,.7,-1.2,-.7,.2,.8,-.2,-.8])
    body = make_body(165)
    joints = forward_kinematics(reference,body)
    holds = {before[limb]:joints.limbs[limb].end.copy() for limb in LIMBS}
    holds['H05'] = holds['H01'] + [4.,3.]
    return annotation,reference,holds


class MotionTests(unittest.TestCase):
    def test_release_and_arrival_boundary_semantics(self):
        annotation,_,_ = fixture()
        frames,labels = contact_schedule(annotation)
        self.assertEqual(frames.tolist(),list(range(9)))
        self.assertEqual([r['left_hand'] for r in labels],['H01']*3+['free']*3+['H05']*3)
        for limb in LIMBS[1:]:
            self.assertEqual({r[limb] for r in labels},{annotation['contacts']['before'][limb]})

    def test_reject_unconfirmed_or_inconsistent_motion(self):
        annotation,_,_ = fixture()
        variants=[]
        for key,value in [('release_frame',0),('arrival_frame',3),('arrival_frame',9),
                          ('release_frame',3.0),('support_contacts_maintained',False)]:
            a=copy.deepcopy(annotation);a['motion'][key]=value;variants.append(a)
        a=copy.deepcopy(annotation);a['contacts']['after']['right_foot']='H01';variants.append(a)
        a=copy.deepcopy(annotation);a['contacts']['before']['left_hand']='unknown';variants.append(a)
        for a in variants:
            with self.subTest(a=a), self.assertRaises(ValueError):validate_motion_annotation(a)

    def test_batch_matches_existing_body_geometry(self):
        _,q,_ = fixture();body=make_body(165)
        qs=np.stack([q,q+np.array([5,3,.2,0,.1,0,0,0,0,0,0])])
        actual=batch_joints(qs,body)
        for t,pose in enumerate(qs):
            expected=forward_kinematics(pose,body)
            for i,limb in enumerate(LIMBS):np.testing.assert_allclose(actual[t,i],expected.limbs[limb],atol=1e-12)

    def test_known_reachable_move_preserves_supports_and_avoids_angle_wrap_jump(self):
        a,q,holds=fixture();reference=np.tile(q,(9,1))
        reference[4:,5]+=2*np.pi
        result=solve_motion_sequence(reference,np.ones((9,9)),np.arange(9),holds,a,30)
        self.assertLess(result['metrics']['max_contact_error_cm'],.1)
        self.assertLess(result['metrics']['max_support_drift_cm'],.1)
        self.assertLess(result['metrics']['max_angle_step_deg'],15)
        self.assertFalse(any(result['metrics']['bend_direction_changes'].values()))
        self.assertTrue(result['metrics']['optimizer']['success'])

    def test_missing_frames_and_nonfinite_weights_rejected(self):
        a,q,holds=fixture()
        with self.assertRaisesRegex(ValueError,'consecutive'):
            solve_motion_sequence(np.tile(q,(8,1)),np.ones((8,9)),np.arange(8),holds,a,30)
        weights=np.ones((9,9));weights[4,2]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):
            solve_motion_sequence(np.tile(q,(9,1)),weights,np.arange(9),holds,a,30)

    def test_optimizer_failure_is_not_exportable_as_success(self):
        import beta_translator.motion_ik as module
        actual=module.least_squares
        def fail_final(fun,x0,**kwargs):
            if 'jac_sparsity' in kwargs:
                return SimpleNamespace(success=False,x=x0,message='iteration limit',cost=1.,optimality=1.)
            return actual(fun,x0,**kwargs)
        a,q,holds=fixture()
        with patch.object(module,'least_squares',side_effect=fail_final):
            with self.assertRaisesRegex(RuntimeError,'iteration limit'):
                solve_motion_sequence(np.tile(q,(9,1)),np.ones((9,9)),np.arange(9),holds,a,30)

    def test_motion_annotation_persistence_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            video=Path(d)/'video';video.write_bytes(b'identity')
            a,_,_=fixture();a.update(video_sha256=file_sha256(video),candidate_run_id='map',notes='')
            path=Path(d)/'annotations.json'
            save_annotations(path,a,video,'map',a['selected_hold_ids'],9)
            self.assertEqual(load_annotations(path,video,'map',a['selected_hold_ids'],9),a)
            a['motion']['arrival_frame']=10
            with self.assertRaises(ValueError):save_annotations(path,a,video,'map',a['selected_hold_ids'],9)

    def test_changed_annotations_invalidate_motion_export(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);video=root/'video';video.write_bytes(b'identity')
            for name in INPUT_NAMES:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'input')
            artifact=root/'tables/motion.csv';artifact.write_bytes(b'result')
            summary=dict(provenance=provenance(video,root),output_sha256={'tables/motion.csv':file_sha256(artifact)})
            (root/SUMMARY_NAME).write_text(json.dumps(summary))
            self.assertEqual(len(current_motion_outputs(video,root)),2)
            (root/'state/endpoint_annotations.json').write_bytes(b'changed body or timing')
            with self.assertRaisesRegex(ValueError,'outdated'):current_motion_outputs(video,root)


if __name__=='__main__':unittest.main()
