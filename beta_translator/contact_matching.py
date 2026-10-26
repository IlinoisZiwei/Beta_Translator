"""Draft contacts from visibility, hold-mask proximity, and temporal persistence."""

import cv2
import numpy as np
import pandas as pd

CONTACT_LANDMARKS = {
    'left_hand': 'left_wrist', 'right_hand': 'right_wrist',
    'left_foot': 'left_foot', 'right_foot': 'right_foot',
}


def contact_mask_distance(mask, point, margin):
    """Return distance to the nearest hold pixel in the search window, or infinity."""
    x, y = point
    height, width = mask.shape
    x0, x1 = max(0, int(np.floor(x - margin))), min(width, int(np.ceil(x + margin)) + 1)
    y0, y1 = max(0, int(np.floor(y - margin))), min(height, int(np.ceil(y + margin)) + 1)
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if not len(xs):
        return np.inf
    return float(np.hypot(xs + x0 - x, ys + y0 - y).min())


def match_limb_contacts(poses, masks, transforms, video_file, *, hand_margin_px=20,
                        foot_margin_px=10, min_visibility=0.5, min_contact_sec=0.5,
                        ambiguity_gap_px=3, min_inliers=25):
    """Predict per-frame limb contacts and return them with persistent intervals and FPS."""
    if not masks:
        raise ValueError('Review at least one route hold first.')
    if not 0 <= min_visibility <= 1 or min_contact_sec <= 0:
        raise ValueError('Check the visibility and persistence settings.')
    if min(hand_margin_px, foot_margin_px) <= 0 or ambiguity_gap_px < 0:
        raise ValueError('Margins must be positive; ambiguity gap must be nonnegative.')

    fps, count, width, height, pose_rows, references = _prepare_contact_inputs(
        poses, masks, video_file)

    # Settings are expressed at 720px image width and scale with the input resolution.
    scale = width / 720
    margins = {limb: (hand_margin_px if 'hand' in limb else foot_margin_px) * scale
               for limb in CONTACT_LANDMARKS}
    records = []
    for frame in range(count):
        row = pose_rows.loc[frame] if frame in pose_rows.index else None
        H = _verified_homography(transforms.get(frame), min_inliers)
        frame_records = _frame_limb_records(
            row, frame, fps, width, height, H is not None, min_visibility)
        candidates = _find_hold_candidates(references, H, frame_records, margins, width, height)
        _assign_frame_contacts(frame_records, candidates, margins, ambiguity_gap_px * scale)
        records.extend(frame_records.values())

    contacts, events = group_contact_intervals(pd.DataFrame(records), fps, min_contact_sec)
    return contacts, events, fps


def _prepare_contact_inputs(poses, masks, video_file):
    """Validate video, pose rows, and hold masks; prepare frame lookups and hold boxes."""
    cap = cv2.VideoCapture(str(video_file))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = (int(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not np.isfinite(fps) or fps <= 0 or min(count, width, height) <= 0:
        raise ValueError('Could not read the video metadata.')
    required = ['frame'] + [f'{name}_{suffix}' for name in CONTACT_LANDMARKS.values()
                            for suffix in ('x_px', 'y_px', 'visibility')]
    if not set(required).issubset(poses.columns):
        raise ValueError('Run pose extraction first; required landmark columns are missing.')
    frame_ids = poses['frame'].to_numpy(dtype=float)
    if (not np.isfinite(frame_ids).all() or (frame_ids != np.floor(frame_ids)).any()
            or (frame_ids < 0).any() or (frame_ids >= count).any() or poses['frame'].duplicated().any()):
        raise ValueError('Pose frame IDs must be unique integers inside the video.')
    pose_rows = poses.set_index('frame')
    references = {}
    for hold_id, mask in masks.items():
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.shape != (height, width) or not mask.any():
            raise ValueError(f'{hold_id}: expected a nonempty mask at the original video size.')
        x, y, w, h = cv2.boundingRect(mask)
        corners = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
        references[hold_id] = (mask, corners)
    return fps, count, width, height, pose_rows, references


def _verified_homography(entry, min_inliers):
    """Return a reliable frame-0-to-current transform, or None when alignment is unverified."""
    H = np.asarray(entry[0], dtype=float) if entry is not None else None
    aligned = (entry is not None and entry[2] == 'ok' and entry[1] >= min_inliers
               and H.shape == (3, 3) and np.isfinite(H).all())
    if aligned:
        aligned = np.linalg.cond(H) < 1e12
    return H if aligned else None


def _frame_limb_records(row, frame, fps, width, height, aligned, min_visibility):
    """Create four limb observations, marking missing or unreliable evidence before matching."""
    records = {}
    for limb, name in CONTACT_LANDMARKS.items():
        point = np.array([row[f'{name}_x_px'], row[f'{name}_y_px']]) if row is not None else np.full(2, np.nan)
        visibility = float(row[f'{name}_visibility']) if row is not None else np.nan
        status = 'tentative'
        if row is None:
            status = 'missing_pose'
        elif not aligned:
            status = 'alignment_unverified'
        elif not np.isfinite(point).all() or not (0 <= point[0] < width and 0 <= point[1] < height):
            status = 'invalid_landmark'
        elif not np.isfinite(visibility) or visibility < min_visibility:
            status = 'low_visibility'
        records[limb] = dict(frame=frame, time_sec=frame/fps, limb=limb,
            x_px=point[0], y_px=point[1], visibility=visibility, candidate_hold=None,
            hold_id=None, distance_px=np.nan, match_score=np.nan, alternatives='', status=status)
    return records


def _find_hold_candidates(references, H, frame_records, margins, width, height):
    """Find nearby holds using masks warped into the same frame as the limb points."""
    candidates = {limb: [] for limb in CONTACT_LANDMARKS}
    if H is None:
        return candidates
    for hold_id, (mask, corners) in references.items():
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        if not np.isfinite(projected).all():
            continue
        lower, upper = projected.min(axis=0), projected.max(axis=0)
        nearby = []
        # Bounding boxes skip distant holds before the more expensive mask warp.
        for limb, record in frame_records.items():
            if record['status'] != 'tentative':
                continue
            point = np.array([record['x_px'], record['y_px']])
            box_distance = np.linalg.norm(np.maximum(np.maximum(lower-point, point-upper), 0))
            if box_distance <= margins[limb]:
                nearby.append((limb, point))
        if not nearby:
            continue
        warped = cv2.warpPerspective(mask, H, (width, height), flags=cv2.INTER_NEAREST)
        for limb, point in nearby:
            distance = contact_mask_distance(warped, point, margins[limb])
            if distance <= margins[limb]:
                candidates[limb].append((distance, hold_id))
    return candidates


def _assign_frame_contacts(frame_records, candidates, margins, ambiguity_gap_px):
    """Fill each eligible observation with its nearest candidate, flagging near-ties."""
    for limb, record in frame_records.items():
        if record['status'] != 'tentative':
            continue
        ranked = sorted(candidates[limb])
        if not ranked:
            record['status'] = 'no_candidate'
        else:
            distance, hold_id = ranked[0]
            record.update(candidate_hold=hold_id, distance_px=distance,
                          match_score=max(0, 1-distance/(margins[limb]+1)))
            record['alternatives'] = ','.join(hold for _, hold in ranked[1:])
            if len(ranked) > 1 and ranked[1][0] - distance <= ambiguity_gap_px:
                record['status'] = 'ambiguous'


def group_contact_intervals(contacts, fps, min_contact_sec):
    """Mark long enough runs on the same hold as likely contacts; uncertain frames split runs."""
    contacts = contacts.copy()
    events = []
    minimum_frames = max(1, int(np.ceil(min_contact_sec * fps)))
    for limb in CONTACT_LANDMARKS:
        group = contacts[contacts['limb'] == limb]
        keys = group['candidate_hold'].where(group['status'] == 'tentative', '').fillna('')
        runs = (keys.ne(keys.shift()) | group['frame'].diff().ne(1)).cumsum()
        for _, run in group.groupby(runs):
            if run['status'].iloc[0] != 'tentative' or len(run) < minimum_frames:
                continue
            hold_id = run['candidate_hold'].iloc[0]
            contacts.loc[run.index, 'hold_id'] = hold_id
            contacts.loc[run.index, 'status'] = 'likely_contact'
            start, end = int(run['frame'].iloc[0]), int(run['frame'].iloc[-1])
            events.append(dict(limb=limb, hold_id=hold_id, start_frame=start, end_frame=end,
                               start_sec=start/fps, end_sec=(end+1)/fps, duration_sec=len(run)/fps,
                               min_visibility=float(run['visibility'].min()),
                               min_match_score=float(run['match_score'].min())))
    # Missing/ambiguous observations break intervals; no hidden contact is filled in.
    event_columns = ['limb', 'hold_id', 'start_frame', 'end_frame', 'start_sec', 'end_sec',
                     'duration_sec', 'min_visibility', 'min_match_score']
    return contacts, pd.DataFrame(events, columns=event_columns)
