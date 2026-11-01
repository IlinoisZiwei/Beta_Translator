"""Projective IK must preserve metric geometry and reject unsupported contacts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from beta_translator.inverse_kinematics import LIMBS, RATIOS, make_body, forward_kinematics
from beta_translator.pose_adapter import (
    LIMB_LANDMARKS, image_to_wall_cm, wall_cm_to_image, pose_row_to_vector,
)
from beta_translator.plane_ik import plane_contact_blockers, contact_span_blockers, validate_saved_plane_geometry
from beta_translator.project_io import file_sha256


class ProjectiveIKTests(unittest.TestCase):
    def test_pose_and_limb_lengths_survive_perspective_projection(self):
        q=np.array([-20.,-150.,.15,1.7,.4,-1.8,-.5,.2,.7,-.4,-.8])
        body=make_body(180)
        joints=forward_kinematics(q,body)
        matrix=np.array([[.6,.03,-100.],[.05,-.55,100.],[.0002,.0003,1.]])
        row={}
        for limb in LIMBS:
            pixels=wall_cm_to_image(np.array(joints.limbs[limb]),None,1280,matrix)
            np.testing.assert_allclose(image_to_wall_cm(pixels,None,1280,matrix),joints.limbs[limb],atol=1e-9)
            for name,xy in zip(LIMB_LANDMARKS[limb],pixels):
                row[name+'_x_px'],row[name+'_y_px']=xy
        recovered=pose_row_to_vector(row,None,1280,matrix)
        np.testing.assert_allclose(recovered,q,atol=1e-9)

    def test_invalid_projection_is_not_treated_as_a_scalar_scale(self):
        with self.assertRaisesRegex(ValueError,'invertible'):
            image_to_wall_cm([[1,2]],2,100,np.zeros((3,3)))
        with self.assertRaisesRegex(ValueError,'horizon'):
            image_to_wall_cm([[1,2]],2,100,[[1,0,0],[0,1,0],[1,0,-1]])

    def test_reach_bound_accepts_a_known_pose_and_rejects_excess_span(self):
        body=dict(height_cm=180,ratios=dict(RATIOS))
        joints=forward_kinematics(np.array([0,0,.1,1.7,.4,-1.8,-.5,.2,.7,-.4,-.8]),make_body(180))
        contacts=dict(zip(LIMBS,['A','B','C','D']))
        annotation=dict(contacts=dict(before=contacts,after=dict(contacts)))
        geometry=dict(hold_positions_cm={contacts[l]:joints.limbs[l].end.tolist() for l in LIMBS},
                      supported_hold_ids=list(contacts.values()))
        self.assertEqual(contact_span_blockers(annotation,geometry,body),[])
        self.assertEqual(plane_contact_blockers(annotation,geometry),[])
        geometry['hold_positions_cm']['A']=[0,1000]
        self.assertTrue(contact_span_blockers(annotation,geometry,body))
        geometry['supported_hold_ids'].remove('A')
        self.assertIn('A',plane_contact_blockers(annotation,geometry)[0])

    def test_saved_calibration_cannot_be_changed_or_expand_contact_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'state').mkdir()
            video=root/'video';video.write_bytes(b'video')
            hold_map=root/'state/reviewed_hold_map.npz';hold_map.write_bytes(b'map')
            matrix=[[1,0,0],[0,-1,0],[0,0,1]]
            calibration=dict(video_sha256=file_sha256(video),candidate_run_id='run',reference_frame_number=5,
                image_to_wall_cm=matrix,selected_hold_positions=[
                    dict(hold_id=h,inside_plane=True,inside_grid_hull=h!='E') for h in 'ABCDE'])
            path=root/'state/plane_calibration.json';path.write_text(json.dumps(calibration))
            geometry=dict(video_sha256=file_sha256(video),hold_map_sha256=file_sha256(hold_map),
                calibration_sha256=file_sha256(path),candidate_run_id='run',reference_frame_number=5,
                image_to_wall_cm=matrix,supported_hold_ids=list('ABCD'))
            contacts=dict(zip(LIMBS,'ABCD'))
            annotation=dict(plane_geometry=geometry,candidate_run_id='run',contacts=dict(before=contacts,after=contacts))
            np.testing.assert_array_equal(validate_saved_plane_geometry(annotation,video,root,5),matrix)
            altered=copy.deepcopy(annotation);altered['plane_geometry']['supported_hold_ids'].append('E')
            with self.assertRaisesRegex(ValueError,'differs'):
                validate_saved_plane_geometry(altered,video,root,5)
            path.write_text(json.dumps({**calibration,'reference_frame_number':6}))
            with self.assertRaisesRegex(ValueError,'stale'):
                validate_saved_plane_geometry(annotation,video,root,5)


if __name__=='__main__':unittest.main()
