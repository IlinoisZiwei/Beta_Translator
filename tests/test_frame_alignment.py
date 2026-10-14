"""Run from the repository root: python -m unittest discover -s tests -v."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from beta_translator.frame_alignment import (
    align_pose_to_frame0, export_frame0_pose_data, plot_frame0_pose_preview,
)


def project(points, H):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ H.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def pose_table(frames, points, visibility=None):
    return pd.DataFrame({
        "frame": frames,
        "time_sec": np.array(frames) / 30,
        "left_wrist_x_px": np.asarray(points)[:, 0],
        "left_wrist_y_px": np.asarray(points)[:, 1],
        "left_wrist_visibility": np.ones(len(frames)) if visibility is None else visibility,
        "left_wrist_z": np.full(len(frames), -0.2),
    })


class FrameAlignmentTests(unittest.TestCase):
    def test_camera_motion_removed_real_limb_motion_preserved_and_sparse_frames(self):
        # An actual projective camera change, not just an easy translation.
        H = np.array([[1.03, .02, 26], [-.01, .97, -11], [.0002, -.0001, 1.]])
        reference_points = np.array([[100., 200.], [100., 200.], [115., 191.]])
        current_points = reference_points.copy()
        current_points[1:] = project(reference_points[1:], H)
        raw = pose_table([0, 8, 19], current_points)
        raw.index = [4, 20, 55]  # Neither the row number nor index is a frame ID.
        original = raw.copy(deep=True)
        aligned, diagnostics = align_pose_to_frame0(raw, {
            0: (H, 100, "ok"), 8: (H, 100, "ok"), 19: (H, 100, "ok"),
            7: (H, 100, "ok"),  # Video frame without a detected pose.
        })
        np.testing.assert_allclose(
            aligned[["left_wrist_x_frame0_px", "left_wrist_y_frame0_px"]], reference_points, atol=1e-10)
        pd.testing.assert_frame_equal(raw, original)
        pd.testing.assert_frame_equal(aligned[original.columns], original)
        self.assertEqual(aligned.frame.tolist(), [0, 8, 19])
        self.assertEqual(aligned.index.tolist(), [4, 20, 55])
        self.assertIn(7, diagnostics.frame.tolist())
        self.assertEqual(aligned.alignment_status.tolist(), ["reference", "ok", "ok"])

    def test_bad_and_reused_transforms_never_become_measurements(self):
        raw = pose_table(range(7), np.full((7, 2), 100.))
        H = np.eye(3)
        aligned, _ = align_pose_to_frame0(raw, {
            1: (H, 0, "reused"),
            2: (np.zeros((3, 3)), 100, "ok"),
            3: (H * np.nan, 100, "ok"),
            4: (H, 3, "ok"),
            6: (H, 100, "reference"),  # Only frame 0 may be a reference.
        })
        self.assertEqual(aligned.alignment_status.tolist(), [
            "reference", "reused", "invalid", "invalid", "insufficient_inliers", "missing", "untrusted_status"])
        self.assertTrue(aligned.left_wrist_x_frame0_px.iloc[1:].isna().all())
        self.assertFalse(aligned.alignment_usable.iloc[1:].any())

    def test_visibility_and_nonfinite_points_preserve_raw_but_hide_aligned(self):
        raw = pose_table([0, 1, 2, 3], [[10., 20.], [11., 21.], [np.nan, 22.], [12., np.inf]],
                         [1., .2, 1., 1.])
        original = raw.copy(deep=True)
        aligned, _ = align_pose_to_frame0(raw, {i: (np.eye(3), 100, "ok") for i in range(4)})
        self.assertEqual(aligned.left_wrist_frame0_usable.tolist(), [True, False, False, False])
        self.assertTrue(aligned.left_wrist_x_frame0_px.iloc[1:].isna().all())
        pd.testing.assert_frame_equal(aligned[raw.columns], original)

    def test_projective_horizon_is_missing_not_infinite(self):
        inverse = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., -100.]])
        aligned, _ = align_pose_to_frame0(pose_table([1], [[100., 50.]]), {
            1: (np.linalg.inv(inverse), 100, "ok")})
        self.assertTrue(np.isnan(aligned.left_wrist_x_frame0_px.iloc[0]))
        self.assertFalse(aligned.left_wrist_frame0_usable.iloc[0])

    def test_export_roundtrip_includes_inverse_and_coordinate_metadata(self):
        H = np.array([[1., 0., 20.], [0., 1., -15.], [0., 0., 1.]])
        with tempfile.TemporaryDirectory() as directory:
            original_path = Path(directory) / "pose_coordinates.csv"
            raw = pose_table([0, 5], [[100., 200.], [120., 185.]])
            raw.to_csv(original_path, index=False)
            original_bytes = original_path.read_bytes()
            _, _, paths = export_frame0_pose_data(
                raw, {5: (H, 100, "ok")}, Path(directory) / "aligned", source_video="videos/trial1.mp4")
            saved = pd.read_csv(paths["poses"])
            np.testing.assert_allclose(saved.left_wrist_x_frame0_px, [100., 100.])
            metadata = json.loads(paths["metadata"].read_text())
            self.assertEqual(metadata["reference_frame"], 0)
            self.assertEqual(metadata["units"], "pixels")
            self.assertEqual(metadata["axes"]["y"], "down")
            matrices = pd.read_csv(paths["homographies"])
            record = matrices[matrices.frame == 5].iloc[0]
            inverse = np.array([record[f"frame_to_ref_h{r}{c}"] for r in range(3) for c in range(3)]).reshape(3, 3)
            np.testing.assert_allclose(project(np.array([[120., 185.]]), inverse), [[100., 200.]])
            self.assertEqual(original_path.read_bytes(), original_bytes)

    def test_schema_and_frame_ids_rejected_instead_of_silent_misalignment(self):
        raw = pose_table([0, 2], [[1., 2.], [3., 4.]])
        for frames in ([0, 0], [0, 1.5], [0, np.nan], [0, -1]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                align_pose_to_frame0(raw.assign(frame=frames), {})
        with self.assertRaises(ValueError):
            align_pose_to_frame0(raw.drop(columns="left_wrist_y_px"), {})
        with self.assertRaises(ValueError):
            align_pose_to_frame0(raw, {}, min_visibility=1.5)
        once, _ = align_pose_to_frame0(raw, {})
        with self.assertRaises(ValueError):
            align_pose_to_frame0(once, {})

    def test_video_preview_uses_inverse_on_actual_sampled_frame(self):
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        H = np.array([[1., 0., 20.], [0., 1., -10.], [0., 0., 1.]])
        reference = np.zeros((120, 160, 3), dtype=np.uint8)
        reference[60:70, 30:40, 1] = 255
        moved = cv2.warpPerspective(reference, H, (160, 120))
        homographies = {1: (H, 100, "ok")}
        aligned, _ = align_pose_to_frame0(pose_table([1], [[55., 55.]]), homographies)
        mask = np.zeros((120, 160), dtype=bool)
        mask[60:70, 30:40] = True
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
            self.assertTrue(writer.isOpened())
            writer.write(reference)
            writer.write(moved)
            writer.release()
            fig = plot_frame0_pose_preview(reference, aligned, frames=[1],
                                           hold_masks={"H01": mask}, video_path=video,
                                           homographies=homographies)
            try:
                image = np.asarray(fig.axes[0].images[0].get_array())
                ys, xs = np.nonzero(image[:, :, 1] > 150)
                self.assertAlmostEqual(xs.mean(), 34.5, delta=1)
                self.assertAlmostEqual(ys.mean(), 64.5, delta=1)
            finally:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
