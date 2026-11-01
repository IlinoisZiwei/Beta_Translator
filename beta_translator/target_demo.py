"""Export a labeled slow replay of one verified target-body IK clip."""
import json
from pathlib import Path

import cv2
import numpy as np

from .motion_pipeline import current_motion_outputs
from .project_io import file_sha256, convert_video
from .trial_pipeline import checkpoint, valid_cache, write_json
from .portable_paths import project_path


def build_target_demo(video, root, plan, slowdown=3, pause_seconds=1):
    root = Path(root)
    candidates = []
    for move in plan['moves']:
        if move['status'] != 'generated':
            continue
        destination = project_path(move['output_dir'], root/'state/direct_ik_plan.json')
        current_motion_outputs(video, destination)
        summary = json.loads((destination/'state/motion_ik_summary.json').read_text())
        candidates.append((summary['metrics']['max_angle_step_deg'], move, summary))
    if not candidates:
        return None
    if type(slowdown) is not int or slowdown < 1 or pause_seconds < 0:
        raise ValueError('Replay speed must be a positive integer factor with nonnegative pauses.')
    _, move, summary = min(candidates, key=lambda item:item[0])
    source = project_path(move['target_video_file'], root/'state/direct_ik_plan.json')
    height_cm = plan['body']['height_cm']
    output = root/f'videos/trial3_target_{height_cm:g}cm_ik_demo.mp4'
    poster = root/'images/target_ik_demo.png'
    state = root/'state/target_ik_demo.json'
    inputs = dict(source_sha256=file_sha256(source),height_cm=height_cm,
        ratios=plan['body']['ratios'],slowdown=slowdown,pause_seconds=pause_seconds,
        module_sha256=file_sha256(__file__))
    manifest = root/'state/target_ik_demo_manifest.json'
    if valid_cache(manifest, inputs):
        result = json.loads(state.read_text())
        for key in ('video_file', 'poster_file', 'source_clip', 'comparison_video'):
            result[key] = str(project_path(result[key], state))
        return result
    cap = cv2.VideoCapture(str(source))
    fps, count = cap.get(5), int(cap.get(7))
    width, height = int(cap.get(3)), int(cap.get(4))
    if not cap.isOpened() or count < 2 or fps <= 0:
        cap.release();raise ValueError('Target video is not decodable.')
    raw = root/'videos/target_demo_encoding.mp4'
    writer = cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height+36))
    if not writer.isOpened():
        cap.release();raise RuntimeError('Cannot encode the target demo.')
    pause_frames = round(pause_seconds*fps)
    try:
        for index in range(count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Target video ended early.')
            canvas=np.full((height+36,width,3),245,np.uint8);canvas[:height]=frame
            cv2.putText(canvas,f'{1/slowdown:.2f}x replay | {pause_seconds:g} s start/end pauses',
                        (12,height+25),0,.48,(30,30,30),1,cv2.LINE_AA)
            repeat=slowdown + (pause_frames if index==0 else 0) + (pause_frames if index==count-1 else 0)
            for _ in range(repeat):writer.write(canvas)
            if index==count//2:cv2.imwrite(str(poster),canvas)
    finally:
        cap.release();writer.release()
    convert_video(raw,output)
    cap=cv2.VideoCapture(str(output));actual=int(cap.get(7));actual_fps=cap.get(5);cap.release()
    expected=count*slowdown+2*pause_frames
    if actual!=expected or not np.isclose(actual_fps,fps):
        raise RuntimeError('Demo timing differs from the requested replay speed.')
    result=dict(video_file=str(output),poster_file=str(poster),height_cm=height_cm,
        ratios=plan['body']['ratios'],move_index=move['move_index'],moving_limb=move['moving_limb'],
        from_hold=move['from_hold'],to_hold=move['to_hold'],before_frame=move['before_frame'],
        after_frame=move['after_frame'],source_clip=str(source),
        comparison_video=str(project_path(move['video_file'], state)),
        source_frame_count=count,output_frame_count=actual,fps=fps,duration_seconds=actual/fps,
        playback_speed=1/slowdown,pause_seconds=pause_seconds,metrics=summary['metrics'],
        selection='Generated clip with the lowest maximum frame-to-frame angle change.',
        interpretation='One fitted 2D target-body movement; not a full-route animation or climbing-success prediction.')
    write_json(state,result)
    checkpoint(manifest,inputs,[output,poster,state])
    raw.unlink()
    return result
