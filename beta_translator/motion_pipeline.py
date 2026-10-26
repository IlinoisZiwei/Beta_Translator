"""Load reviewed motion inputs, fit a sequence, and export reproducible previews."""

from pathlib import Path
import json
import os
import tempfile

import cv2
import numpy as np
import pandas as pd

from .inverse_kinematics import LIMBS, POSE_VECTOR_NAMES
from .motion_ik import contact_schedule, solve_motion_sequence, validate_motion_annotation
from .pose_adapter import (
    LIMB_LANDMARKS, to_frame0, image_to_wall_cm, wall_cm_to_image,
    pose_row_to_vector, angle_visibility_weights,
)
from .project_io import (
    file_sha256, load_reviewed_holds, load_alignment, load_annotations, convert_video,
)


SUMMARY_NAME = 'state/motion_ik_summary.json'
INPUT_NAMES = ('state/endpoint_annotations.json', 'tables/pose_coordinates.csv',
               'state/reviewed_hold_map.npz', 'state/alignment_cache.npz')
MODULE_NAMES = ('motion_ik.py', 'motion_pipeline.py', 'inverse_kinematics.py', 'pose_adapter.py')


def provenance(video_file, output_dir):
    output_dir = Path(output_dir)
    return dict(video_sha256=file_sha256(video_file),
        inputs={name:file_sha256(output_dir/name) for name in INPUT_NAMES},
        modules={name:file_sha256(Path(__file__).parent/name) for name in MODULE_NAMES})


def current_motion_outputs(video_file, output_dir):
    """Only return artifacts that match the current saved annotations and source data."""
    output_dir = Path(output_dir)
    path = output_dir/SUMMARY_NAME
    if not path.exists():
        return []
    summary = json.loads(path.read_text())
    if summary['provenance'] != provenance(video_file, output_dir):
        raise ValueError('Motion outputs are outdated. Rerun Section 8B before exporting.')
    paths = [path]
    for name, digest in summary['output_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid motion artifact path.')
        item = output_dir/relative
        if not item.is_file() or file_sha256(item) != digest:
            raise ValueError(f'Motion output is missing or changed: {name}. Rerun Section 8B.')
        paths.append(item)
    return paths


def _render(result, annotation, aligned_points, rows, transforms, masks, records,
            video_file, stage, fps):
    frames, poses = result['frames'], result['poses']
    moving = annotation['moving_limb']
    scale = annotation['body']['px_per_cm']
    height, width = next(iter(masks.values())).shape
    all_joints = wall_cm_to_image(result['joints'].reshape(-1,2), scale, height)
    # A stable crop includes the complete fitted and observed bodies throughout the clip.
    points = np.concatenate([all_joints, aligned_points.reshape(-1,2)])
    lo = np.floor(points.min(axis=0)-55).astype(int)
    hi = np.ceil(points.max(axis=0)+55).astype(int)
    x0,y0 = np.maximum(lo,[0,0]); x1,y1 = np.minimum(hi,[width,height])
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Invalid motion preview crop.')
    panel_width = 500
    panel_height = int(round((y1-y0)/(x1-x0)*panel_width/2))*2
    header, footer = 68, 72
    canvas_size = (panel_width*2, panel_height+header+footer)
    background = np.full((height,width,3),245,np.uint8)
    holds = {r['hold_id']:r for r in records}
    for hold_id in annotation['selected_hold_ids']:
        mask = masks[hold_id]
        background[mask] = (210,215,218)
        contours,_ = cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(background,contours,-1,(130,135,140),1)
    for hold_id in set(annotation['contacts']['before'].values()) | set(annotation['contacts']['after'].values()):
        background[masks[hold_id]] = (170,200,220)
        h = holds[hold_id]
        cv2.putText(background,hold_id,(int(h['cx_px'])+8,int(h['cy_px'])-6),
                    cv2.FONT_HERSHEY_SIMPLEX,.48,(35,35,35),1,cv2.LINE_AA)
    names = list(dict.fromkeys(n for chain in LIMB_LANDMARKS.values() for n in chain))
    links = [(chain[i],chain[i+1]) for chain in LIMB_LANDMARKS.values() for i in (0,1)]
    links += [('left_shoulder','right_shoulder'),('left_hip','right_hip'),
              ('left_shoulder','left_hip'),('right_shoulder','right_hip')]
    raw = stage/'raw_motion.mp4'
    writer = cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,canvas_size)
    if not writer.isOpened():
        raise RuntimeError('Could not create the motion preview.')
    capture = cv2.VideoCapture(str(video_file))
    capture.set(cv2.CAP_PROP_POS_FRAMES,int(frames[0]))
    samples = {}
    keyframes = sorted(set([int(frames[0]),annotation['motion']['release_frame']-1,
        annotation['motion']['release_frame'],
        (annotation['motion']['release_frame']+annotation['motion']['arrival_frame'])//2,
        annotation['motion']['arrival_frame'],int(frames[-1])]))
    try:
        for t,frame_index in enumerate(frames):
            ok,frame = capture.read()
            if not ok:
                raise ValueError(f'Cannot read source frame {frame_index}.')
            observed = cv2.warpPerspective(frame,np.linalg.inv(transforms[int(frame_index)][0]),(width,height))
            observed_joints = {n:tuple(np.rint(p).astype(int)) for n,p in zip(names,aligned_points[t])}
            for a,b in links:
                visible = min(rows[t][a+'_visibility'],rows[t][b+'_visibility']) >= .5
                cv2.line(observed,observed_joints[a],observed_joints[b],
                         (220,130,30) if visible else (150,150,150),2,cv2.LINE_AA)
            fitted = background.copy()
            pixels = wall_cm_to_image(result['joints'][t].reshape(-1,2),scale,height).reshape(4,3,2)
            roots = np.rint(pixels[[0,1,3,2,0],0]).astype(int)
            cv2.polylines(fitted,[roots],False,(60,150,35),3,cv2.LINE_AA)
            for limb_index,limb in enumerate(LIMBS):
                chain = np.rint(pixels[limb_index]).astype(int)
                free = result['contacts'][t][limb] == 'free'
                color = (0,155,235) if free else (60,150,35)
                cv2.polylines(fitted,[chain],False,color,3,cv2.LINE_AA)
                for xy in chain:
                    cv2.circle(fitted,tuple(xy),4,color,-1,cv2.LINE_AA)
                if not free:
                    xy = wall_cm_to_image([result['targets'][t,limb_index]],scale,height)[0]
                    cv2.drawMarker(fitted,tuple(np.rint(xy).astype(int)),(35,45,210),cv2.MARKER_CROSS,12,2)
            canvas = np.full((canvas_size[1],canvas_size[0],3),255,np.uint8)
            for panel,offset in ((observed,0),(fitted,panel_width)):
                canvas[header:header+panel_height,offset:offset+panel_width] = cv2.resize(
                    panel[y0:y1,x0:x1],(panel_width,panel_height),interpolation=cv2.INTER_AREA)
            cv2.putText(canvas,'Observed video (aligned)',(15,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(35,35,35),2,cv2.LINE_AA)
            cv2.putText(canvas,f"Candidate IK | {annotation['body']['height_cm']:g} cm",(panel_width+15,26),
                        cv2.FONT_HERSHEY_SIMPLEX,.65,(35,35,35),2,cv2.LINE_AA)
            status = result['contacts'][t][moving]
            cv2.putText(canvas,f'Frame {frame_index} | {frame_index/fps:.2f}s | {moving}: {status}',
                        (15,53),cv2.FONT_HERSHEY_SIMPLEX,.58,(35,35,35),1,cv2.LINE_AA)
            cv2.putText(canvas,'Gray observed links: low visibility | Orange: released limb',
                        (15,header+panel_height+27),cv2.FONT_HERSHEY_SIMPLEX,.52,(50,50,50),1,cv2.LINE_AA)
            cv2.putText(canvas,'2D geometric candidate; free-hand path is inferred. No force or balance model.',
                        (15,header+panel_height+54),cv2.FONT_HERSHEY_SIMPLEX,.49,(50,50,50),1,cv2.LINE_AA)
            writer.write(canvas)
            if int(frame_index) in keyframes:
                # Storyboard uses the fitted panel with frame and contact phase labels.
                sample = np.full((panel_height+56,panel_width,3),255,np.uint8)
                sample[56:] = canvas[header:header+panel_height,panel_width:]
                cv2.putText(sample,f'Frame {frame_index} | {status}',(12,32),
                            cv2.FONT_HERSHEY_SIMPLEX,.7,(35,35,35),2,cv2.LINE_AA)
                samples[int(frame_index)] = sample
    finally:
        capture.release(); writer.release()
    normal = stage/'videos/motion_ik_h264.mp4'
    slow = stage/'videos/motion_ik_slow_h264.mp4'
    normal.parent.mkdir(parents=True,exist_ok=True)
    convert_video(raw,normal)
    # Render an explicitly labeled review copy at one-third speed.
    slow_raw = stage/'raw_motion_slow.mp4'
    source = cv2.VideoCapture(str(raw))
    slow_writer = cv2.VideoWriter(str(slow_raw),cv2.VideoWriter_fourcc(*'mp4v'),fps/3,canvas_size)
    if not slow_writer.isOpened():
        source.release()
        raise RuntimeError('Could not create the slow review video.')
    try:
        while True:
            ok,frame = source.read()
            if not ok: break
            cv2.rectangle(frame,(canvas_size[0]-195,35),(canvas_size[0]-5,66),(255,255,255),-1)
            cv2.putText(frame,'0.33x review',(canvas_size[0]-185,56),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,180),2,cv2.LINE_AA)
            slow_writer.write(frame)
    finally:
        source.release();slow_writer.release()
    convert_video(slow_raw,slow)
    items = [samples[f] for f in keyframes]
    blank = np.full_like(items[0],255)
    while len(items)<6:items.append(blank.copy())
    storyboard = np.vstack([np.hstack(items[:3]),np.hstack(items[3:6])])
    path = stage/'images/ik_motion_storyboard.png';path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),storyboard):
        raise RuntimeError('Could not write the motion storyboard.')
    return [normal,slow,path]


def run_motion_analysis(video_file, output_dir, settings=None):
    video_file,output_dir = Path(video_file),Path(output_dir)
    before_hashes = provenance(video_file,output_dir)
    records,masks,run_id,hold_ids = load_reviewed_holds(output_dir/'state/reviewed_hold_map.npz',video_file)
    cap = cv2.VideoCapture(str(video_file))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps = cap.get(cv2.CAP_PROP_FPS);cap.release()
    annotation = load_annotations(output_dir/'state/endpoint_annotations.json',video_file,run_id,hold_ids,count)
    validate_motion_annotation(annotation)
    scale = annotation['body']['px_per_cm']
    if scale is None or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Set a measured positive wall scale before motion fitting.')
    with np.load(output_dir/'state/alignment_cache.npz',allow_pickle=False) as saved:
        alignment_settings = json.loads(str(saved['settings'].item()))
    transforms = load_alignment(output_dir/'state/alignment_cache.npz',video_file,
                                output_dir/'tables/pose_coordinates.csv',alignment_settings)
    if transforms is None:
        raise ValueError('Alignment is outdated; rerun Section 5.')
    poses = pd.read_csv(output_dir/'tables/pose_coordinates.csv').set_index('frame')
    if not poses.index.is_unique:raise ValueError('Duplicate pose frames.')
    frames,_ = contact_schedule(annotation)
    height = next(iter(masks.values())).shape[0]
    names = list(dict.fromkeys(n for chain in LIMB_LANDMARKS.values() for n in chain))
    reference,weights,aligned,rows = [],[],[],[]
    for frame in frames:
        if frame not in poses.index or frame not in transforms or transforms[frame][2] != 'ok':
            raise ValueError(f'Frame {frame} lacks a pose or verified alignment; review this gap first.')
        row = poses.loc[frame].copy()
        points = to_frame0([[row[n+'_x_px'],row[n+'_y_px']] for n in names],transforms,int(frame))
        if not np.isfinite(points).all():raise ValueError(f'Non-finite landmarks at frame {frame}.')
        for name,point in zip(names,points):row[name+'_x_px'],row[name+'_y_px'] = point
        reference.append(pose_row_to_vector(row,scale,height))
        weights.append(angle_visibility_weights(row));aligned.append(points);rows.append(row)
    targets = {r['hold_id']:image_to_wall_cm([[r['cx_px'],r['cy_px']]],scale,height)[0] for r in records}
    result = solve_motion_sequence(reference,weights,frames,targets,annotation,fps,settings=settings)
    with tempfile.TemporaryDirectory(prefix='beta-motion-') as directory:
        stage = Path(directory)
        files = _render(result,annotation,np.asarray(aligned),rows,transforms,masks,records,video_file,stage,fps)
        table_rows = []
        for t,frame in enumerate(frames):
            row = dict(frame=int(frame),time_sec=float(frame/fps))
            row.update(dict(zip(POSE_VECTOR_NAMES,map(float,result['poses'][t]))))
            for i,limb in enumerate(LIMBS):
                row[limb+'_contact'] = result['contacts'][t][limb]
                row[limb+'_contact_error_cm'] = (float(result['contact_errors_cm'][t,i])
                                                if result['active'][t,i] else np.nan)
            table_rows.append(row)
        table = stage/'tables/motion_ik_poses.csv';table.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(table_rows).to_csv(table,index=False);files.append(table)
        state = stage/'state/motion_ik_trajectory.npz';state.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(state,frames=frames,poses=result['poses'],reference_poses=result['reference_poses'],
                            visibility_weights=result['weights'],targets_cm=result['targets'],active=result['active'],
                            swing_guide_cm=result['swing_guide_cm'])
        files.append(state)
        # Verify encoded frame counts and speed instead of assuming video encoding succeeded.
        for path,expected_fps in [(files[0],fps),(files[1],fps/3)]:
            cap=cv2.VideoCapture(str(path));encoded_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));encoded_fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
            if encoded_count != len(frames) or not np.isclose(encoded_fps,expected_fps):
                raise RuntimeError('Encoded motion timing does not match the source frame schedule.')
        summary = dict(schema_version=1,annotation=annotation,provenance=before_hashes,
            frame_count=len(frames),source_fps=fps,endpoint_interval_seconds=float((frames[-1]-frames[0])/fps),
            encoded_duration_seconds=float(len(frames)/fps),metrics=result['metrics'],
            frame_convention='release=first free frame; arrival=first new-contact frame; endpoints inclusive',
            swing_path='Minimum-jerk straight-line soft guide, not an observed hand trajectory.',
            interpretation='2D kinematic candidate with checked soft-contact residuals; no force, friction, balance, collision, or success prediction.',
            output_sha256={str(p.relative_to(stage)):file_sha256(p) for p in files})
        summary_file=stage/SUMMARY_NAME;summary_file.write_text(json.dumps(summary,indent=2)+'\n')
        files.append(summary_file)
        if provenance(video_file,output_dir) != before_hashes:
            raise RuntimeError('Inputs changed during motion fitting; no results installed.')
        for source in files:
            destination=output_dir/source.relative_to(stage);destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix(destination.suffix+'.tmp')
            temporary.write_bytes(source.read_bytes());os.replace(temporary,destination)
    current_motion_outputs(video_file,output_dir)
    return summary
