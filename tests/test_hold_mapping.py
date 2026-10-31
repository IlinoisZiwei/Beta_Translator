import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from beta_translator.hold_mapping import compact_mask, deduplicate, person_mask, reconstruct_wall


class HoldMappingTests(unittest.TestCase):
    def test_duplicate_masks_merge_but_neighboring_holds_remain(self):
        a=compact_mask(np.ones((8,8),bool),(10,10))
        b=compact_mask(np.ones((8,8),bool),(11,10))
        c=compact_mask(np.ones((8,8),bool),(25,10))
        for x,score in [(a,.8),(b,.9),(c,.7)]:x.update(score=score,clipped=False)
        result=deduplicate([a,b,c])
        self.assertEqual(len(result),2)
        self.assertIs(result[0],b)

    def test_missing_person_evidence_never_becomes_a_clean_frame(self):
        mask=person_mask(np.full((133,2),np.nan),np.zeros(133),(32,32,3))
        self.assertTrue(mask.all())

    def test_reconstruction_uses_observed_donor_pixels_and_tracks_holes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'video').write_bytes(b'test video')
            digest=hashlib.sha256(b'test video').hexdigest()
            np.savez(root/'pose.npz',keypoints=np.zeros((2,133,2)),scores=np.ones((2,133)),
                     metadata=json.dumps({'video_sha256':digest}))
            np.savez(root/'align.npz',frames=[0,1],matrices=[np.eye(3),np.eye(3)],
                     statuses=['ok','ok'],settings=json.dumps({'reference_frame_number':0}),video_sha256=digest)
            images=[np.full((32,32,3),50,np.uint8),np.full((32,32,3),180,np.uint8)]
            masks=[np.zeros((32,32),np.uint8),np.zeros((32,32),np.uint8)]
            masks[0][10:20,10:20]=1
            masks[1][10:12,10:12]=1
            class Capture:
                def set(self,prop,index):self.index=index
                def read(self):return True,images[self.index].copy()
                def release(self):pass
            with patch('beta_translator.hold_mapping.cv2.VideoCapture',return_value=Capture()), \
                 patch('beta_translator.hold_mapping.person_mask',side_effect=masks):
                image,report=reconstruct_wall(root/'video',root/'pose.npz',root/'align.npz',[0,1],0,root/'out')
            self.assertTrue((image[17,17]==180).all())
            self.assertTrue((image[5,5]==50).all())
            self.assertTrue((image[10,10]==128).all())
            self.assertGreater(report['unresolved_pixels'],0)
            with np.load(root/'out/pixel_provenance.npz') as z:
                self.assertEqual(z['source_frame'][17,17],1)
                self.assertEqual(z['source_frame'][10,10],-1)

    def test_reference_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'video').write_bytes(b'test video')
            digest=hashlib.sha256(b'test video').hexdigest()
            np.savez(root/'pose.npz',keypoints=np.zeros((1,133,2)),scores=np.ones((1,133)),
                     metadata=json.dumps({'video_sha256':digest}))
            np.savez(root/'align.npz',settings=json.dumps({'reference_frame_number':0}),video_sha256=digest)
            with self.assertRaisesRegex(ValueError,'share a reference'):
                reconstruct_wall(root/'video',root/'pose.npz',root/'align.npz',[0],450,root/'out')

    def test_changed_video_cannot_reuse_pose_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'video').write_bytes(b'new video')
            np.savez(root/'pose.npz',metadata=json.dumps({'video_sha256':'old video hash'}))
            with self.assertRaisesRegex(ValueError,'another video'):
                reconstruct_wall(root/'video',root/'pose.npz',root/'align.npz',[0],0,root/'out')


if __name__=='__main__':unittest.main()
