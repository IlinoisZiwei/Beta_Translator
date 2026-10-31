"""Keep one climbing identity; separate observed poses from display-only gap filling."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .project_io import file_sha256, save_npz, save_alignment, read_frame

LANDMARKS = {
    'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30, 'left_foot': 31, 'right_foot': 32,
}
NAMES = list(LANDMARKS)
TORSO = [0, 1, 6, 7]
LINKS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7),
         (6, 7), (6, 8), (8, 10), (7, 9), (9, 11), (10, 14), (11, 15)]
SETTINGS = dict(version=1, min_anchor_score=2.0, center_torso_limit=.75,
                min_torso_ratio=.5, max_torso_ratio=1.8, crop_padding=.15,
                guard_padding_frames=2, max_display_gap_sec=2.0, max_endpoint_motion_torsos=1.0)


def ensure_bootstrap_alignment(video, raw_pose_path, destination, reference=450):
    """Create a separate initial wall pass when restoring from raw inputs."""
    destination = Path(destination)
    if destination.exists():
        return destination  # Input hashes are checked by refine_target_poses.
    raw = pd.read_csv(raw_pose_path).set_index('frame')
    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def features(image, frame):
        mask = np.full(image.shape[:2],255,np.uint8)
        if frame in raw.index:
            row = raw.loc[frame]
            xs = [row[n+'_x_px'] for n in NAMES]
            ys = [row[n+'_y_px'] for n in NAMES]
            mask[max(0,int(min(ys))-60):int(max(ys))+60,
                 max(0,int(min(xs))-60):int(max(xs))+60] = 0
        return sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask)
    reference_points, reference_descriptors = features(read_frame(video,reference),reference)
    cap = cv2.VideoCapture(str(video)); transforms = {}; last = np.eye(3); frame = 0
    try:
        while True:
            ok,image = cap.read()
            if not ok:break
            H,n = None,0
            if frame == reference:
                H,n = np.eye(3),len(reference_points)
            else:
                points,descriptors = features(image,frame)
                if descriptors is not None and reference_descriptors is not None:
                    good = [pair[0] for pair in matcher.knnMatch(reference_descriptors,descriptors,k=2)
                            if len(pair)==2 and pair[0].distance < .75*pair[1].distance]
                    if len(good)>=25:
                        a = np.float32([reference_points[m.queryIdx].pt for m in good])[:,None]
                        b = np.float32([points[m.trainIdx].pt for m in good])[:,None]
                        H,inliers = cv2.findHomography(a,b,cv2.RANSAC,4.)
                        n = int(inliers.sum()) if inliers is not None else 0
                        if H is None or not np.isfinite(H).all() or n<25:H=None
            status = 'ok' if H is not None else 'reused'
            if H is None:H=last
            else:last=H
            transforms[frame]=(H,n,status);frame+=1
    finally:cap.release()
    settings=dict(reference_frame_number=reference,nfeatures=4000,min_inliers=25,
                  ratio_test=.75,ransac_threshold=4.,exclusion_pad=60,wall_roi=None)
    save_alignment(destination,transforms,video,raw_pose_path,settings)
    return destination


def ranges(mask):
    indices = np.flatnonzero(mask)
    return [[int(g[0]), int(g[-1])] for g in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1) if len(g)]


def transform(points, H):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def torso_geometry(points):
    p = np.asarray(points)
    return p.mean(axis=0), max(10., float(np.linalg.norm(p[:2].mean(0) - p[2:].mean(0))))


def identity_match(candidate, anchor, previous=None, elapsed_frames=1, settings=SETTINGS):
    """Reject distant or implausibly scaled torsos in stabilized wall coordinates."""
    if not np.isfinite(candidate).all() or not np.isfinite(anchor).all():
        return False, 'nonfinite_pose'
    center, size = torso_geometry(candidate)
    target, target_size = torso_geometry(anchor)
    if np.linalg.norm(center - target) > max(20., settings['center_torso_limit'] * target_size):
        return False, 'target_center_mismatch'
    if not settings['min_torso_ratio'] <= size / target_size <= settings['max_torso_ratio']:
        return False, 'target_scale_mismatch'
    if previous is not None:
        old_center, old_size = torso_geometry(previous)
        allowance = max(25., settings['center_torso_limit'] * old_size) * (1 + .15 * min(10, elapsed_frames - 1))
        if np.linalg.norm(center - old_center) > allowance:
            return False, 'wall_motion_jump'
    return True, 'matched_target'


def interpolate_display(points, observed, matrices, torso_indices, fps, settings=SETTINGS):
    """Fill only bounded short gaps in wall coordinates. Never change observed data."""
    display = np.asarray(points).copy()
    inferred = np.zeros(len(points), bool)
    for start, end in ranges(~np.asarray(observed, bool)):
        left, right = start - 1, end + 1
        if left < 0 or right >= len(points) or end - start + 1 > round(settings['max_display_gap_sec'] * fps):
            continue
        if not all(f in matrices for f in range(left, right + 1)):
            continue
        a = transform(points[left], np.linalg.inv(matrices[left]))
        b = transform(points[right], np.linalg.inv(matrices[right]))
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            continue
        _, size = torso_geometry(a[torso_indices])
        if np.max(np.linalg.norm(a - b, axis=1)) > settings['max_endpoint_motion_torsos'] * size:
            continue
        for frame in range(start, end + 1):
            alpha = (frame - left) / (right - left)
            display[frame] = transform((1 - alpha) * a + alpha * b, matrices[frame])
            inferred[frame] = True
    return display, inferred


def load_identity_guard(output_dir, video, wholebody):
    """An optional guard leaves older pipelines unchanged and rejects stale evidence."""
    path = Path(output_dir) / 'state/identity_guard.npz'
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata'].item()))
        if (metadata['video_sha256'] != file_sha256(video)
                or metadata['wholebody_sha256'] != file_sha256(wholebody)):
            raise ValueError('Identity guard belongs to different video or whole-body features. Rerun target tracking.')
        valid = saved['observed'].copy()
        if valid.dtype != bool or valid.ndim != 1 or len(valid) != metadata['frame_count']:
            raise ValueError('Invalid identity-guard frame mask.')
        return dict(observed=valid, display=saved['display_keypoints'].copy(),
                    display_inferred=saved['display_inferred'].copy(), metadata=metadata,
                    sha256=file_sha256(path))


def refine_target_poses(video, raw_pose_path, wholebody_path, anchor_alignment, model_path,
                        output_dir, pose_settings, settings=None):
    """Reuse matching full-frame poses; re-detect rejected frames inside the target ROI."""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import mediapipe as mp
    settings = {**SETTINGS, **(settings or {})}
    video, raw_pose_path, wholebody_path, anchor_alignment, model_path, output_dir = map(Path,
        (video, raw_pose_path, wholebody_path, anchor_alignment, model_path, output_dir))
    inputs = dict(video_sha256=file_sha256(video), raw_pose_sha256=file_sha256(raw_pose_path),
                  wholebody_sha256=file_sha256(wholebody_path), anchor_alignment_sha256=file_sha256(anchor_alignment),
                  model_sha256=file_sha256(model_path), code_sha256=file_sha256(Path(__file__)),
                  pose_settings=pose_settings, settings=settings)
    manifest_path = output_dir / 'state/pose_manifest.json'
    pose_path = output_dir / 'tables/pose_coordinates.csv'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('inputs') == inputs and all((output_dir / p).exists() and file_sha256(output_dir / p) == h
                                                  for p, h in manifest.get('artifacts', {}).items()) and manifest.get('artifacts'):
            print('Reused verified target-identity pose outputs.')
            return pd.read_csv(pose_path)
    raw = pd.read_csv(raw_pose_path).set_index('frame')
    with np.load(wholebody_path, allow_pickle=False) as saved:
        meta = json.loads(str(saved['metadata'].item()))
        if meta['video_sha256'] != inputs['video_sha256']:
            raise ValueError('Whole-body features belong to another video.')
        q, scores, boxes = saved['keypoints'], saved['scores'], saved['boxes']
    with np.load(anchor_alignment, allow_pickle=False) as saved:
        if (str(saved['video_sha256'].item()) != inputs['video_sha256']
                or str(saved['pose_sha256'].item()) != inputs['raw_pose_sha256']):
            raise ValueError('Bootstrap alignment must match the raw video and pose pass.')
        matrices = {int(f): H for f, H, status, inliers in zip(saved['frames'], saved['matrices'], saved['statuses'], saved['inliers'])
                    if str(status) == 'ok' and inliers >= 25 and np.isfinite(H).all() and np.linalg.cond(H) < 1e12}
    count, fps, width, height = len(q), meta['fps'], meta['width'], meta['height']
    anchor_valid = np.zeros(count, bool)
    anchor_wall = np.full((count, 4, 2), np.nan)
    last, last_frame = None, None
    for frame in range(count):
        if frame not in matrices or not np.isfinite(q[frame, [5, 6, 11, 12]]).all() or scores[frame, [5, 6, 11, 12]].min() < settings['min_anchor_score']:
            continue
        anchor = transform(q[frame, [5, 6, 11, 12]], np.linalg.inv(matrices[frame]))
        ok, _ = identity_match(anchor, anchor, last, frame - last_frame if last_frame is not None else 1, settings)
        if ok:
            anchor_wall[frame], anchor_valid[frame], last, last_frame = anchor, True, anchor, frame
    padding = settings['guard_padding_frames']
    if padding:
        anchor_valid &= np.convolve((~anchor_valid).astype(int), np.ones(2*padding+1), mode='same') == 0

    result_points = np.full((count, len(NAMES), 2), np.nan)
    rows, diagnostics = [], []
    last, last_frame = None, None
    cap = cv2.VideoCapture(str(video))
    options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1, **pose_settings)
    with vision.PoseLandmarker.create_from_options(options) as detector:
        try:
            for frame in range(count):
                ok, image = cap.read()
                if not ok:
                    raise ValueError('Video ended before the tracked feature cache.')
                row = raw.loc[frame].to_dict() if frame in raw.index else None
                reason, source, chosen = 'raw_pose_missing', 'unavailable', None
                if row is not None:
                    p = np.array([[row[n + '_x_px'], row[n + '_y_px']] for n in NAMES])
                raw_ok = False
                if anchor_valid[frame]:
                    elapsed = frame - last_frame if last_frame is not None else 1
                    if row is not None:
                        raw_ok, reason = identity_match(transform(p[TORSO], np.linalg.inv(matrices[frame])),
                            anchor_wall[frame], last, elapsed, settings)
                        if raw_ok:
                            chosen, source = row, 'observed_full_frame'
                    if chosen is None:
                        box = boxes[frame].copy()
                        pad = (box[2:] - box[:2]) * settings['crop_padding']
                        lo = np.maximum(0, np.floor(box[:2] - pad)).astype(int)
                        hi = np.minimum([width, height], np.ceil(box[2:] + pad)).astype(int)
                        if np.isfinite(box).all() and np.all(hi - lo > 20):
                            crop = image[lo[1]:hi[1], lo[0]:hi[0]]
                            crop = cv2.resize(crop, None, fx=512/crop.shape[0], fy=512/crop.shape[0])
                            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                               data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                            if result.pose_landmarks:
                                landmarks = result.pose_landmarks[0]
                                c = np.array([[landmarks[i].x * (hi[0]-lo[0]) + lo[0],
                                               landmarks[i].y * (hi[1]-lo[1]) + lo[1]] for i in LANDMARKS.values()])
                                crop_ok, _ = identity_match(transform(c[TORSO], np.linalg.inv(matrices[frame])),
                                    anchor_wall[frame], last, elapsed, settings)
                                if crop_ok:
                                    chosen = dict(frame=frame, time_sec=frame/fps, time_ms=round(frame*1000/fps))
                                    for i, (name, index) in enumerate(LANDMARKS.items()):
                                        landmark = landmarks[index]
                                        chosen.update({name+'_x_px':c[i, 0], name+'_y_px':c[i, 1],
                                            name+'_x_norm':c[i, 0]/width, name+'_y_norm':c[i, 1]/height,
                                            name+'_z':landmark.z*(hi[0]-lo[0])/width,
                                            name+'_visibility':landmark.visibility})
                                    p, source = c, 'observed_target_crop'
                else:
                    reason = 'target_identity_uncertain'
                if chosen is not None:
                    chosen.update(frame=frame, pose_source=source, identity_status='target_verified')
                    rows.append(chosen)
                    result_points[frame] = p
                    last = transform(p[TORSO], np.linalg.inv(matrices[frame])); last_frame = frame
                diagnostics.append(dict(frame=frame, time_sec=frame/fps, anchor_observed=bool(anchor_valid[frame]),
                                        raw_pose_present=frame in raw.index, raw_pose_accepted=raw_ok,
                                        reason=reason, pose_source=source))
        finally:
            cap.release()
    if not rows:
        raise ValueError('No target poses survived identity checks.')
    observed = np.isfinite(result_points).all(axis=(1, 2))
    display_points, inferred = interpolate_display(result_points, observed, matrices, TORSO, fps, settings)
    rtmw_points = q.copy(); rtmw_points[~anchor_valid] = np.nan
    rtmw_display, rtmw_inferred = interpolate_display(rtmw_points, anchor_valid, matrices, [5,6,11,12], fps, settings)
    # A confirmed display-only MediaPipe gap can supply the visible body skeleton
    # when RTMW feet were unstable at its endpoints. These values never enter ML.
    body_mapping = [5,6,7,8,9,10,11,12,13,14,15,16,19,22,17,20]
    for frame in np.flatnonzero(~anchor_valid & inferred):
        rtmw_display[frame] = np.nan
        rtmw_display[frame, body_mapping] = display_points[frame]
        rtmw_inferred[frame] = True
    poses = pd.DataFrame(rows)
    poses.to_csv(pose_path, index=False)
    display_rows = poses.to_dict('records')
    for frame in np.flatnonzero(inferred):
        row = dict(frame=int(frame), time_sec=frame/fps, time_ms=round(frame*1000/fps),
                   pose_source='interpolated_display_only', identity_status='unobserved')
        for i, name in enumerate(NAMES):
            x, y = display_points[frame, i]
            row.update({name+'_x_px':x, name+'_y_px':y, name+'_x_norm':x/width, name+'_y_norm':y/height,
                        name+'_z':np.nan, name+'_visibility':0.})
        display_rows.append(row)
    pd.DataFrame(display_rows).sort_values('frame').to_csv(output_dir/'tables/pose_display_coordinates.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir/'tables/identity_tracking.csv', index=False)
    with np.load(anchor_alignment, allow_pickle=False) as anchor_cache:
        reference_index = json.loads(str(anchor_cache['settings'].item())).get('reference_frame_number', 0)
    metadata = dict(inputs, frame_count=count, fps=fps, reference_frame=reference_index,
                    interpretation='observed gates RTMW evidence; display_keypoints may be interpolated and never feed contact inference')
    save_npz(output_dir/'state/identity_guard.npz', metadata=np.array(json.dumps(metadata)),
             observed=anchor_valid, display_keypoints=rtmw_display, display_inferred=rtmw_inferred)
    diagnostic = pd.DataFrame(diagnostics)
    report = dict(inputs=inputs, frame_count=count, observed_pose_frames=int(observed.sum()),
        original_pose_frames=len(raw), reused_raw_frames=int((diagnostic.pose_source=='observed_full_frame').sum()),
        crop_recovered_frames=int((diagnostic.pose_source=='observed_target_crop').sum()),
        raw_rejected_ranges=ranges(diagnostic.raw_pose_present & ~diagnostic.raw_pose_accepted),
        missing_observation_ranges=ranges(~observed), display_interpolated_ranges=ranges(inferred),
        wholebody_unobserved_ranges=ranges(~anchor_valid), wholebody_interpolated_ranges=ranges(rtmw_inferred),
        limitations='Rule-based identity checks for this video, not an independently measured pose-accuracy improvement. Display interpolation is an estimate, not proof of maintained contacts.')
    (output_dir/'state/identity_tracking_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    writer = cv2.VideoWriter(str(output_dir/'videos/pose_output.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    cap = cv2.VideoCapture(str(video))
    try:
        for frame in range(count):
            ok, image = cap.read()
            if not ok: raise ValueError('Video ended while rendering identity preview.')
            p = display_points[frame]
            color = (0,165,245) if inferred[frame] else (30,200,50)
            status = 'INTERPOLATED - display only' if inferred[frame] else ('target observed' if observed[frame] else 'target unavailable')
            if np.isfinite(p).all():
                for a,b in LINKS: cv2.line(image,tuple(p[a].astype(int)),tuple(p[b].astype(int)),color,3,cv2.LINE_AA)
                for xy in p:cv2.circle(image,tuple(xy.astype(int)),4,color,-1,cv2.LINE_AA)
            cv2.rectangle(image,(0,0),(width,75),(245,245,245),-1)
            cv2.putText(image,f'Frame {frame} | {frame/fps:.2f}s | {status}',(20,32),0,.8,(35,35,35),2,cv2.LINE_AA)
            cv2.putText(image,'Identity lock only; joint visibility and contact still require review',(20,62),0,.65,(45,45,45),1,cv2.LINE_AA)
            writer.write(image)
    finally:
        cap.release();writer.release()
    artifacts = ['tables/pose_coordinates.csv','tables/pose_display_coordinates.csv','tables/identity_tracking.csv',
                 'state/identity_guard.npz','state/identity_tracking_summary.json','videos/pose_output.mp4']
    manifest = dict(inputs=inputs, pose_sha256=file_sha256(pose_path),
                    overlay_sha256=file_sha256(output_dir/'videos/pose_output.mp4'),
                    artifacts={p:file_sha256(output_dir/p) for p in artifacts})
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'},indent=2))
    return poses
