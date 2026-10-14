"""Export pose points in the hold detector's frame-0 pixel coordinates.

The notebook stores H_ref_to_frame, so points must use its inverse. This is
2D image registration, not metric wall calibration or 3D body reconstruction.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _frame_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid frame number: {value!r}") from None
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"Invalid frame number: {value!r}")
    return int(number)


def _landmarks(pose_df):
    if not pose_df.columns.is_unique:
        raise ValueError("Pose columns must be unique.")
    names = [name[:-5] for name in pose_df.columns if name.endswith("_x_px")]
    y_names = [name[:-5] for name in pose_df.columns if name.endswith("_y_px")]
    if not names or set(names) != set(y_names):
        raise ValueError("Expected paired <landmark>_x_px and <landmark>_y_px columns.")
    required = {f"{name}_visibility" for name in names}
    if not required.issubset(pose_df.columns):
        raise ValueError("Every landmark needs a visibility column.")
    return names


def _prepare_alignment(frame, entry, min_inliers):
    H = inverse = None
    source_status, status, inliers = "missing", "missing", None
    if entry is not None:
        try:
            matrix, count, source_status = entry
            inliers = _frame_number(count)
            H = np.asarray(matrix, dtype=float)
            if H.shape != (3, 3) or not np.isfinite(H).all() or np.max(np.abs(H)) == 0:
                raise ValueError("Invalid matrix")
            H = H / np.max(np.abs(H))
            if np.linalg.cond(H) > 1e12:
                raise ValueError("Ill-conditioned matrix")
            inverse = np.linalg.inv(H)
            if not np.isfinite(inverse).all():
                raise ValueError("Invalid inverse")
            inverse = inverse / np.max(np.abs(inverse))
            status = source_status if source_status in {"ok", "reused"} else "untrusted_status"
            if status == "ok" and inliers < min_inliers:
                status = "insufficient_inliers"
        except (TypeError, ValueError, np.linalg.LinAlgError):
            H = inverse = None
            status = "invalid"

    if frame == 0:
        # The reference coordinate system is exactly frame 0, regardless of a
        # numerically estimated self-match or the reference feature count.
        H = inverse = np.eye(3)
        status = "reference"
    usable = status in {"ok", "reference"}
    record = {
        "frame": frame, "alignment_status": status,
        "alignment_source_status": str(source_status),
        "alignment_inliers": inliers, "alignment_usable": usable,
    }
    for prefix, matrix in (("ref_to_frame", H), ("frame_to_ref", inverse)):
        for row in range(3):
            for col in range(3):
                record[f"{prefix}_h{row}{col}"] = np.nan if matrix is None else float(matrix[row, col])
    return record, inverse if usable else None


def _project(points, matrix):
    """Project finite points; points on the projective horizon stay missing."""
    result = np.full_like(points, np.nan, dtype=float)
    finite = np.isfinite(points).all(axis=1)
    positions = np.flatnonzero(finite)
    homogeneous = np.column_stack((points[finite], np.ones(finite.sum())))
    projected = homogeneous @ matrix.T
    denominator = projected[:, 2]
    scale = np.maximum(1.0, np.sum(np.abs(homogeneous * matrix[2]), axis=1))
    good = np.isfinite(projected).all(axis=1) & (np.abs(denominator) > 1e-12 * scale)
    result[positions[good]] = projected[good, :2] / denominator[good, None]
    result[~np.isfinite(result).all(axis=1)] = np.nan
    return result


def align_pose_to_frame0(pose_df, homographies, *, min_visibility=0.5, min_inliers=25):
    """Return (poses, alignment diagnostics) without changing the input table.

    homographies is the notebook's {frame: (H_ref_to_frame, inliers, status)}.
    Lookups use the actual `frame` value, not the row index. All original
    columns, timestamps and sparse frame IDs are retained. New columns are
    `<landmark>_x_frame0_px`, `<landmark>_y_frame0_px` and
    `<landmark>_frame0_usable`. Reused/missing/invalid transforms and low
    visibility produce blank aligned coordinates, never guessed contacts.
    "ok" means the notebook's matching gate passed, not verified accuracy.
    """
    if not np.isfinite(min_visibility) or not 0 <= min_visibility <= 1:
        raise ValueError("min_visibility must be between 0 and 1.")
    min_inliers = _frame_number(min_inliers)
    if "frame" not in pose_df or pose_df.empty:
        raise ValueError("Expected a nonempty pose table with a frame column.")
    names = _landmarks(pose_df)
    frames = [_frame_number(value) for value in pose_df["frame"]]
    if len(set(frames)) != len(frames):
        raise ValueError("Expected at most one pose row per frame.")
    entries = {_frame_number(frame): entry for frame, entry in homographies.items()}
    records, transforms = {}, {}
    for frame in sorted(set(frames) | set(entries) | {0}):
        records[frame], transforms[frame] = _prepare_alignment(frame, entries.get(frame), min_inliers)
    generated = [f"{name}_{suffix}" for name in names
                 for suffix in ("x_frame0_px", "y_frame0_px", "frame0_usable")]
    generated += ["alignment_status", "alignment_source_status", "alignment_inliers", "alignment_usable"]
    if set(generated).intersection(pose_df.columns):
        raise ValueError("Input already contains alignment columns; use the original pose_df.")

    aligned = pose_df.copy(deep=True)
    coordinates = np.full((len(aligned), len(names), 2), np.nan)
    raw = np.stack([pose_df[[f"{name}_x_px", f"{name}_y_px"]].apply(
        pd.to_numeric, errors="coerce").to_numpy(dtype=float) for name in names], axis=1)
    visibility = np.column_stack([pd.to_numeric(pose_df[f"{name}_visibility"], errors="coerce")
                                  for name in names])
    for position, frame in enumerate(frames):
        if transforms[frame] is not None:
            points = _project(raw[position], transforms[frame])
            visible = np.isfinite(visibility[position]) & (visibility[position] >= min_visibility)
            points[~visible] = np.nan
            coordinates[position] = points
    for index, name in enumerate(names):
        aligned[f"{name}_x_frame0_px"] = coordinates[:, index, 0]
        aligned[f"{name}_y_frame0_px"] = coordinates[:, index, 1]
        aligned[f"{name}_frame0_usable"] = np.isfinite(coordinates[:, index]).all(axis=1)
    for column in ("alignment_status", "alignment_source_status", "alignment_inliers", "alignment_usable"):
        aligned[column] = [records[frame][column] for frame in frames]
    return aligned, pd.DataFrame(records.values())


def export_frame0_pose_data(pose_df, homographies, output_dir, *, source_video=None,
                           min_visibility=0.5, min_inliers=25):
    """Write separate aligned CSVs and a description of their coordinate system.

    The files in output_dir are replaced on rerun; the original pose CSV is
    never written. Run pose extraction and homography estimation for the same
    video before calling this function.
    """
    aligned, diagnostics = align_pose_to_frame0(
        pose_df, homographies, min_visibility=min_visibility, min_inliers=min_inliers)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_path = output_dir / "pose_coordinates_frame0.csv"
    alignment_path = output_dir / "homographies.csv"
    metadata_path = output_dir / "coordinate_system.json"
    metadata = {
        "reference_frame": 0, "units": "pixels", "origin": "top-left of video frame 0",
        "axes": {"x": "right", "y": "down"},
        "source_video": None if source_video is None else str(source_video),
        "homography_input_direction": "frame 0 -> current frame",
        "pose_transform": "inverse homography: current frame -> frame 0",
        "min_visibility": float(min_visibility), "min_inliers": int(min_inliers),
        "pose_rows": len(aligned), "alignment_rows": len(diagnostics),
        "landmarks": _landmarks(pose_df),
        "accepted_statuses": ["reference", "ok"],
        "missing_values": "Unusable aligned coordinates are blank; original measurements are preserved.",
        "limitations": [
            "Matching inliers do not independently establish registration accuracy.",
            "Wall-plane registration does not recover off-wall body depth or metric distances.",
            "Original normalized coordinates and z values are not transformed.",
        ],
    }
    aligned.to_csv(pose_path, index=False)
    diagnostics.to_csv(alignment_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return aligned, diagnostics, {"poses": pose_path, "homographies": alignment_path, "metadata": metadata_path}


def plot_frame0_pose_preview(reference_bgr, aligned_poses, *, frames=None, hold_masks=None,
                            video_path=None, homographies=None):
    """Plot sampled transformed skeletons over the SAME frame-0 wall/hold masks.

    Pass wall_image and candidate_masks restricted to reviewed IDs. Supplying
    video_path and homographies warps only the sampled video frames for direct
    visual checks; no stabilized video is written. Without a video, the first
    frame is the background. This cannot establish contact or true 3D position.
    """
    import matplotlib.pyplot as plt

    names = _landmarks(aligned_poses)
    if frames is None:
        count = min(4, len(aligned_poses))
        frames = aligned_poses["frame"].iloc[np.linspace(0, len(aligned_poses) - 1, count, dtype=int)].tolist()
    frames = list(frames)
    if not frames:
        raise ValueError("Select at least one preview frame.")
    if reference_bgr.ndim != 3 or reference_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR reference image.")
    rows = []
    for frame in frames:
        matches = aligned_poses[aligned_poses["frame"] == frame]
        if len(matches) != 1:
            raise ValueError(f"No unique pose row for frame {frame}.")
        rows.append(matches.iloc[0])
    backgrounds = [reference_bgr] * len(frames)
    if video_path is not None:
        import cv2

        if homographies is None:
            raise ValueError("Video previews need the homographies used for the export.")
        cap = cv2.VideoCapture(str(video_path))
        try:
            backgrounds = []
            for frame, row in zip(frames, rows):
                if not row["alignment_usable"]:
                    backgrounds.append(np.zeros_like(reference_bgr))
                    continue
                _, inverse = _prepare_alignment(_frame_number(frame), homographies.get(frame), 0)
                if inverse is None:
                    raise ValueError(f"Preview and exported alignment disagree for frame {frame}.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                ok, image = cap.read()
                if not ok or image.shape != reference_bgr.shape:
                    raise ValueError(f"Cannot read matching video frame {frame}.")
                height, width = reference_bgr.shape[:2]
                backgrounds.append(cv2.warpPerspective(image, inverse, (width, height)))
        finally:
            cap.release()
    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 7), squeeze=False)
    connections = [(side + start, side + end) for side in ("left_", "right_")
                   for start, end in (("shoulder", "elbow"), ("elbow", "wrist"),
                                      ("shoulder", "hip"), ("hip", "knee"),
                                      ("knee", "ankle"), ("ankle", "heel"), ("ankle", "foot"))]
    connections += [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")]
    for ax, frame, row, background in zip(axes[0], frames, rows, backgrounds):
        ax.imshow(background[..., ::-1])
        for hold_id, mask in (hold_masks or {}).items():
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != reference_bgr.shape[:2]:
                plt.close(fig)
                raise ValueError("Hold masks must use the reference image dimensions.")
            if mask.any():
                ax.contour(mask.astype(float), levels=[0.5], colors="gold", linewidths=0.8)
                ys, xs = np.nonzero(mask)
                ax.text(xs.mean(), ys.mean(), hold_id, color="gold", fontsize=8)
        points = {name: np.array([row[f"{name}_x_frame0_px"], row[f"{name}_y_frame0_px"]])
                  for name in names}
        for start, end in connections:
            if start in points and end in points:
                pair = np.array([points[start], points[end]])
                if np.isfinite(pair).all():
                    ax.plot(pair[:, 0], pair[:, 1], color="cyan", linewidth=1.5)
        valid = np.array([point for point in points.values() if np.isfinite(point).all()])
        if len(valid):
            ax.scatter(valid[:, 0], valid[:, 1], c="cyan", s=10)
        ax.set_xlim(-0.5, reference_bgr.shape[1] - 0.5)
        ax.set_ylim(reference_bgr.shape[0] - 0.5, -0.5)
        ax.set_title(f"Frame {frame}: {row['alignment_status']}\n{len(valid)}/{len(names)} usable landmarks")
        ax.axis("off")
    fig.tight_layout()
    return fig
