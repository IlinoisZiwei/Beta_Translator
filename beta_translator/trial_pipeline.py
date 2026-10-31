"""Compact, resumable video-to-contact workflow for independent climbing trials."""
import json
from pathlib import Path
import subprocess
import sys
import uuid

import cv2
import numpy as np
import pandas as pd

from .project_io import (file_sha256, find_ffmpeg, read_frame, convert_video,
                         save_reviewed_holds, load_reviewed_holds, load_alignment)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def valid_cache(path, inputs):
    path = Path(path)
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved.get('inputs') != inputs:
        return None
    artifacts = saved.get('artifacts', {})
    if artifacts and all(Path(p).is_file() and file_sha256(p) == h for p, h in artifacts.items()):
        return saved
    return None


def checkpoint(path, inputs, artifacts, **details):
    value = dict(inputs=inputs, artifacts={str(p): file_sha256(p) for p in artifacts}, **details)
    write_json(path, value)
    return value


def nearest_source_indices(source_times, analysis_times):
    """Map output times to actual decoded source frames, without hiding VFR timing."""
    source = np.asarray(source_times, float)
    if len(source) < 2 or not np.isfinite(source).all() or np.any(np.diff(source) <= 0):
        raise ValueError('Source timestamps must be finite and strictly increasing.')
    right = np.searchsorted(source, analysis_times).clip(1, len(source)-1)
    left = right-1
    chosen = np.where(np.abs(source[left]-analysis_times) <= np.abs(source[right]-analysis_times), left, right)
    if np.any(np.diff(chosen) <= 0):
        raise ValueError('Analysis rate would duplicate source frames; choose a lower FPS.')
    return chosen


def prepare_video(source, video, output_dir, fps=30, width=720):
    source, video, root = map(Path, (source, video, output_dir))
    if not source.is_file():
        raise FileNotFoundError(source)
    for folder in ('state', 'tables', 'images', 'videos', 'automatic', 'reports'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    inputs = dict(source_sha256=file_sha256(source), fps=fps, width=width, version=1)
    manifest = root/'state/video_manifest.json'
    cached = valid_cache(manifest, inputs)
    if cached:
        print('Reused verified analysis video.', flush=True)
        return cached
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(source)],
        check=True, capture_output=True, text=True)
    frames = json.loads(probe.stdout)['frames']
    original = np.array([float(f['best_effort_timestamp_time']) for f in frames])
    relative = original-original[0]
    times = np.arange(int(np.floor(relative[-1]*fps))+1)/fps
    chosen = nearest_source_indices(relative, times)
    terms = [f'eq(n\\,{i})' for i in chosen]
    # A balanced expression avoids FFmpeg's recursion limit on long frame lists.
    while len(terms) > 1:
        terms = ['(' + '+'.join(terms[i:i+2]) + ')' for i in range(0, len(terms), 2)]
    expression = terms[0]
    filters = f'select={expression},setpts=N/({fps}*TB),scale={width}:-2'
    temporary = video.with_name(video.stem+'.part.mp4')
    subprocess.run([find_ffmpeg(), '-nostdin', '-y', '-v', 'error', '-i', str(source),
        '-map', '0:v:0', '-an', '-vf', filters, '-r', str(fps), '-c:v', 'libx264',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], check=True)
    cap = cv2.VideoCapture(str(temporary))
    count, w, h = int(cap.get(7)), int(cap.get(3)), int(cap.get(4))
    cap.release()
    if count != len(chosen):
        raise ValueError('Encoded frame count does not match the source-frame map.')
    temporary.replace(video)
    table = root/'tables/source_frame_map.csv'
    pd.DataFrame(dict(frame=np.arange(count), time_sec=times, source_frame=chosen,
        source_time_sec=relative[chosen], source_pts_sec=original[chosen],
        resampling_offset_sec=relative[chosen]-times)).to_csv(table, index=False)
    result = checkpoint(manifest, inputs, [video, table], source=str(source),
        frame_count=count, fps=fps, width=w, height=h, source_frame_count=len(frames),
        rotation='FFmpeg autorotation; upright pixels saved', mapping='Exact selected source frame indices')
    print(f'Prepared {count} upright frames at {fps} FPS ({w} x {h}).', flush=True)
    return result


def extract_raw_poses(video, model, root):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from .identity_tracking import LANDMARKS
    settings = dict(min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
                    min_tracking_confidence=.5)
    inputs = dict(video_sha256=file_sha256(video), model_sha256=file_sha256(model), settings=settings)
    path, manifest = root/'tables/pose_coordinates_raw.csv', root/'state/pose_raw_manifest.json'
    if valid_cache(manifest, inputs):
        return path, settings
    options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1, **settings)
    cap = cv2.VideoCapture(str(video))
    fps, w, h = cap.get(5), int(cap.get(3)), int(cap.get(4))
    rows, i = [], 0
    try:
        with vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), round(i*1000/fps))
                if result.pose_landmarks:
                    row = dict(frame=i, time_sec=i/fps, time_ms=round(i*1000/fps))
                    for name, index in LANDMARKS.items():
                        p = result.pose_landmarks[0][index]
                        row.update({name+'_x_px':p.x*w, name+'_y_px':p.y*h,
                            name+'_x_norm':p.x, name+'_y_norm':p.y, name+'_z':p.z,
                            name+'_visibility':p.visibility})
                    rows.append(row)
                i += 1
    finally:
        cap.release()
    if not rows:
        raise ValueError('No body poses detected.')
    pd.DataFrame(rows).to_csv(path, index=False)
    checkpoint(manifest, inputs, [path], observed_frames=len(rows), total_frames=i)
    return path, settings


def choose_reference(video, wholebody, output_dir, requested=None):
    """Choose a sharp sampled view with broad visible coverage outside the person."""
    from .hold_mapping import person_mask
    with np.load(wholebody, allow_pickle=False) as z:
        points, scores = z['keypoints'], z['scores']
    count = len(points)
    if requested is not None and (type(requested) is not int or not 0 <= requested < count):
        raise ValueError('Reference frame is outside the analysis video.')
    samples = np.unique(np.linspace(0, count-1, 7, dtype=int)).tolist()
    if requested is not None and requested not in samples:
        samples.append(requested)
    rows = []
    for frame in samples:
        image = read_frame(video, frame)
        visible = ~person_mask(points[frame], scores[frame], image.shape).astype(bool)
        # Score the wall-dominated central image rather than the floor or roof.
        region = np.zeros(image.shape[:2], bool)
        region[int(.12*len(image)):int(.73*len(image)), :] = True
        visible &= region
        lap = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
        sharp = float(lap[visible].var()) if visible.any() else 0.
        coverage = float(visible.sum()/max(1, region.sum()))
        rows.append(dict(frame=frame, visible_fraction=coverage, sharpness=sharp,
                         score=coverage*np.log1p(sharp)))
    selected = int(requested if requested is not None else max(rows, key=lambda r:r['score'])['frame'])
    write_json(Path(output_dir)/'state/reference_selection.json', dict(reference_frame=selected,
        method='User override' if requested is not None else 'Sampled visible area times log sharpness',
        candidates=rows, video_sha256=file_sha256(video)))
    return selected


def track_climber(project, video, root, reference_frame=None):
    from .identity_tracking import ensure_bootstrap_alignment, refine_target_poses
    project, root = Path(project), Path(root)
    wholebody = root/'automatic/wholebody.npz'
    if not wholebody.exists():
        subprocess.run([str(project/'.venv-contact/bin/python'),
            str(project/'scripts/extract_contact_poses.py'), '--video', str(video),
            '--output', str(wholebody), '--model-cache', str(project/'training_data/model_cache')],
            cwd=project, check=True)
    with np.load(wholebody, allow_pickle=False) as z:
        if json.loads(str(z['metadata']))['video_sha256'] != file_sha256(video):
            raise ValueError('Choose a new output folder: pose cache belongs to another video.')
    reference = choose_reference(video, wholebody, root, reference_frame)
    raw, settings = extract_raw_poses(video, project/'models/pose_landmarker.task', root)
    anchor = root/f'state/identity_anchor_alignment_{reference:04d}.npz'
    # The alignment helper otherwise reuses any existing path without checking its inputs.
    align_settings = dict(reference_frame_number=reference, nfeatures=4000, min_inliers=25,
        ratio_test=.75, ransac_threshold=4., exclusion_pad=60, wall_roi=None)
    if anchor.exists() and load_alignment(anchor, video, raw, align_settings) is None:
        raise ValueError('Stale bootstrap alignment. Use a fresh output folder.')
    ensure_bootstrap_alignment(video, raw, anchor, reference)
    poses = refine_target_poses(video, raw, wholebody, anchor,
        project/'models/pose_landmarker.task', root, settings)
    return dict(reference_frame=reference, observed_pose_frames=len(poses),
                wholebody=str(wholebody), pose_path=str(root/'tables/pose_coordinates.csv'))


def build_scene(project, video, root, reference, threshold=.7):
    from .identity_tracking import ensure_bootstrap_alignment
    from .hold_mapping import reconstruct_wall, detect_candidates, export_candidates, filter_observed_candidates
    project, root = Path(project), Path(root)
    poses, alignment = root/'tables/pose_coordinates.csv', root/'state/alignment_cache.npz'
    settings = dict(reference_frame_number=reference, nfeatures=4000, min_inliers=25,
        ratio_test=.75, ransac_threshold=4., exclusion_pad=60, wall_roi=None)
    if alignment.exists() and load_alignment(alignment, video, poses, settings) is None:
        raise ValueError('Alignment inputs changed. Use a fresh output folder.')
    ensure_bootstrap_alignment(video, poses, alignment, reference)
    transforms = load_alignment(alignment, video, poses, settings)
    valid = [f for f,v in transforms.items() if v[2]=='ok' and v[1]>=25]
    if reference not in valid:
        raise ValueError('Selected reference has insufficient wall features.')
    frames = sorted(set([reference]+[valid[i] for i in np.linspace(0,len(valid)-1,21,dtype=int)]))
    wholebody = root/'automatic/wholebody.npz'
    inputs = dict(video_sha256=file_sha256(video), pose_sha256=file_sha256(wholebody),
        alignment_sha256=file_sha256(alignment), model_sha256=file_sha256(project/'models/model_final.pth'),
        config_sha256=file_sha256(project/'models/experiment_config.yml'),
        reference_frame=reference, frames=frames, threshold=threshold, tiled=True, person_padding=30,
        min_observed_fraction=.98, version=2)
    manifest, map_path = root/'state/scene_manifest.json', root/'state/reviewed_hold_map.npz'
    saved = valid_cache(manifest, inputs)
    if saved:
        print('Reused verified hold map.', flush=True)
        return saved
    wall, reconstruction = reconstruct_wall(video, wholebody, alignment, frames, reference, root/'hold_mapping', person_padding=30)
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    torch.set_num_threads(2)
    cfg = get_cfg()
    cfg.merge_from_file(str(project/'models/experiment_config.yml'))
    cfg.MODEL.WEIGHTS = str(project/'models/model_final.pth')
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = 400
    candidates, details = detect_candidates(wall, DefaultPredictor(cfg), tiled=True)
    with np.load(root/'hold_mapping/pixel_provenance.npz', allow_pickle=False) as pixels:
        unresolved = pixels['unresolved']
    retained = filter_observed_candidates(candidates, unresolved)
    details['rejected_unobserved_candidates'] = len(candidates)-len(retained)
    candidates = retained
    if not candidates:
        raise ValueError('No observed hold candidates; inspect the reconstructed image.')
    records = export_candidates(candidates, wall, root/'hold_mapping/candidates')
    masks = {}
    with np.load(root/'hold_mapping/candidates.npz', allow_pickle=False) as z:
        for r in records:
            mask = np.zeros(wall.shape[:2], bool)
            x0,y0,x1,y1 = r['bbox']
            mask[y0:y1,x0:x1] = z[r['hold_id']]
            masks[r['hold_id']] = mask
            r.update(source='automatic_multiframe_tiled_detection', review_status='unreviewed',
                class_name={0:'hold',1:'volume'}.get(r['class_id'],'other'))
    run_id = uuid.uuid4().hex
    ids = [r['hold_id'] for r in records]
    save_reviewed_holds(map_path, video, wall, records, masks, run_id, ids, reference_frame_number=reference)
    # Legacy filename is an interface contract; it does not mean human review occurred.
    cv2.imwrite(str(root/'images/wall_reference.jpg'), read_frame(video, reference))
    return checkpoint(manifest, inputs, [map_path, root/'hold_mapping/candidates.jpg',
        root/'hold_mapping/reconstructed_wall.png'], candidate_count=len(ids),
        selection='All automatic candidates; no manual route filtering', human_reviewed=False,
        alignment_valid_frames=len(valid), alignment_total_frames=len(transforms),
        reconstruction=reconstruction, detection=details)


def infer_contacts(project, video, root):
    from .contact_review import load_review_bundle
    project, root = Path(project), Path(root)
    args = (video, root, root/'automatic/wholebody.npz', project/'models/contact_v1', root/'automatic/predictions')
    try:
        prediction, contacts = load_review_bundle(*args)
        print('Reused verified automatic contacts.', flush=True)
    except (FileNotFoundError, ValueError, KeyError):
        subprocess.run([str(project/'.venv-contact/bin/python'), '-m', 'beta_translator.contact_runner',
            '--video', str(video), '--outputs', str(root), '--package', str(project/'models/contact_v1')],
            cwd=project, check=True)
        prediction, contacts = load_review_bundle(*args)
    return prediction, contacts


def geometry_status(video, root, height_cm, ratios, px_per_cm=None, single_panel=False,
                    source_height_cm=None, calibration_note=''):
    from .direct_ik import geometry_blockers
    from .inverse_kinematics import make_body
    make_body(height_cm, ratios)
    if px_per_cm is not None and not calibration_note.strip():
        raise ValueError('Record the source of the wall scale before using centimeters.')
    result = dict(video_sha256=file_sha256(video), target_height_cm=height_cm, ratios=dict(ratios),
        source_height_cm=source_height_cm, px_per_cm=px_per_cm, single_panel_confirmed=bool(single_panel),
        calibration_note=calibration_note, blockers=geometry_blockers(px_per_cm, single_panel),
        inherited_trial2_geometry=False)
    write_json(Path(root)/'state/geometry_status.json', result)
    return result


def render_summary(video, root, prediction, contacts, plan):
    """One compact prediction video, explicitly separate from retargeted IK output."""
    from .contact_features import LANDMARKS, LIMBS
    from .identity_tracking import load_identity_guard
    root = Path(root)
    records, _, _, _ = load_reviewed_holds(root/'state/reviewed_hold_map.npz', video)
    centers = {r['hold_id']: np.array([r['cx_px'],r['cy_px']],np.float32) for r in records}
    body = root/'automatic/wholebody.npz'
    identity = load_identity_guard(root, video, body)
    with np.load(body,allow_pickle=False) as z:
        q, scores = z['keypoints'], z['scores']
    with np.load(root/'state/alignment_cache.npz',allow_pickle=False) as z:
        transforms = {int(f):(H,str(s)) for f,H,s in zip(z['frames'],z['matrices'],z['statuses'])}
    indexed = contacts.set_index(['frame','limb'])
    cap = cv2.VideoCapture(str(video))
    width, height, fps, count = int(cap.get(3)), int(cap.get(4)), cap.get(5), int(cap.get(7))
    output = root/'videos/automatic_contacts_h264.mp4'
    inputs = dict(video_sha256=file_sha256(video), predictions=prediction['artifact_sha256'],
        map_sha256=file_sha256(root/'state/reviewed_hold_map.npz'),
        guard_sha256=identity['sha256'] if identity else None, renderer_version=1)
    preview_manifest = root/'state/preview_manifest.json'
    if not valid_cache(preview_manifest, inputs):
        raw = root/'videos/contact_preview_raw.mp4'
        writer = cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height+190))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError('Cannot open contact preview writer.')
        links = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        try:
            for frame in range(count):
                ok, im = cap.read()
                if not ok:
                    raise RuntimeError('Video ended while rendering predictions.')
                observed = identity is None or bool(identity['observed'][frame])
                H,status = transforms[frame]
                if observed:
                    for a,b in links:
                        xy=q[frame,[a,b]]
                        if np.isfinite(xy).all():
                            color=(70,190,50) if min(scores[frame,[a,b]])>1.5 else (160,160,160)
                            cv2.line(im,tuple(xy[0].astype(int)),tuple(xy[1].astype(int)),color,2,cv2.LINE_AA)
                canvas=np.full((height+190,width,3),245,np.uint8)
                cv2.putText(canvas,f'Frame {frame} | {frame/fps:.2f}s | ORIGINAL climber',(12,26),0,.63,(30,30,30),2)
                cv2.putText(canvas,'Decoded contacts: experimental input to IK',(12,52),0,.55,(40,70,160),1)
                for j,limb in enumerate(LIMBS):
                    row=indexed.loc[(frame,limb)]
                    label=str(row.candidate_hold) if observed and status=='ok' else 'unknown'
                    cv2.putText(canvas,f'{limb}: {label}  | model score {row.confidence:.2f}',
                                (12,80+24*j),0,.55,(40,40,40),1)
                    if label in centers:
                        target=cv2.perspectiveTransform(centers[label][None,None],H)[0,0]
                        if np.isfinite(target).all() and 0<=target[0]<width and 0<=target[1]<height:
                            p=tuple(np.rint(target).astype(int))
                            cv2.circle(im,p,10,(0,180,240),2)
                            cv2.putText(im,label,(p[0]+10,p[1]),0,.45,(0,0,0),2)
                canvas[190:]=im
                writer.write(canvas)
        finally:
            cap.release();writer.release()
        convert_video(raw, output)
        checkpoint(preview_manifest, inputs, [output])
        raw.unlink()  # Encoding scratch file; the verified H.264 preview is retained.
    else:
        cap.release()
    identity_summary=json.loads((root/'state/identity_tracking_summary.json').read_text())
    scene=json.loads((root/'state/scene_manifest.json').read_text())
    summary=dict(frame_count=count,fps=fps,hold_candidates=scene['candidate_count'],
        observed_pose_frames=identity_summary['observed_pose_frames'],
        alignment_valid_frames=scene['alignment_valid_frames'],
        model_threshold_coverage=float((contacts.status=='accepted_model').mean()),
        movement_proposals=plan['proposal_count'],solver_compatible=plan['solver_compatible_count'],
        generated_ik_videos=plan['generated_count'],target_height_cm=plan['body']['height_cm'],
        geometry_blockers=plan['geometry_blockers'],preview_video=str(output),
        interpretation='Coverage is not accuracy. No trial3 ground truth or climbing-success probability.',
        automatic_ik='User-authorized experimental mode; predictions were not manually confirmed.')
    write_json(root/'state/run_summary.json',summary)
    return summary
