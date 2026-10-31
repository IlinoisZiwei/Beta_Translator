"""Build the trial2 passing-person case report from saved tracking diagnostics."""
import argparse
import base64
import json
from pathlib import Path
import re
import sys
import tempfile
import zipfile

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.identity_tracking import NAMES, LINKS, TORSO, transform, load_identity_guard
from beta_translator.project_io import convert_video, file_sha256


def main(out, video):
    summary=json.loads((out/'state/identity_tracking_summary.json').read_text())
    raw=pd.read_csv(out/'tables/pose_coordinates_raw.csv').set_index('frame')
    observed=pd.read_csv(out/'tables/pose_coordinates.csv').set_index('frame')
    display=pd.read_csv(out/'tables/pose_display_coordinates.csv').set_index('frame')
    diagnostics=pd.read_csv(out/'tables/identity_tracking.csv').set_index('frame')
    guard=load_identity_guard(out,video,out/'automatic/wholebody.npz')
    with np.load(out/'state/identity_anchor_alignment.npz',allow_pickle=False) as z:
        matrices=dict(zip(z['frames'],z['matrices']))
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    hold_count=len(json.loads((out/'state/hold_selection_status.json').read_text())['selected_hold_ids'])
    rejected=summary['raw_rejected_ranges']
    start=max(0,min(a for a,b in rejected)-30) if rejected else 0
    end=min(count-1,max(b for a,b in rejected)+30) if rejected else min(count-1,90)
    def points(row):return np.array([[row[n+'_x_px'],row[n+'_y_px']] for n in NAMES])
    def panel(image, frame, data, before):
        image=image.copy();kind='missing'
        if frame in data.index:
            row=data.loc[frame];p=points(row)
            kind='original full-frame pose' if before else row.pose_source
            color=(40,40,230) if before else ((0,165,245) if kind=='interpolated_display_only' else (30,200,50))
            for a,b in LINKS:cv2.line(image,tuple(p[a].astype(int)),tuple(p[b].astype(int)),color,4,cv2.LINE_AA)
        image=cv2.resize(image,(960,540));canvas=np.full((595,960,3),248,np.uint8);canvas[55:]=image
        title=('BEFORE' if before else 'AFTER')+f' | frame {frame} | {frame/fps:.2f}s'
        cv2.putText(canvas,title,(15,23),0,.6,(30,30,30),1,cv2.LINE_AA)
        cv2.putText(canvas,kind,(15,46),0,.55,(35,35,35),1,cv2.LINE_AA)
        return canvas
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    selected=[195,213,225] if start<=195 and end>=225 else np.linspace(start,end,3,dtype=int).tolist()
    storyboard=[]
    with tempfile.TemporaryDirectory(prefix='identity-review-') as temp:
        path=Path(temp)/'comparison.mp4'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,595))
        try:
            for frame in range(start,end+1):
                ok,image=cap.read()
                if not ok:raise ValueError('Video ended before identity case comparison.')
                combined=np.hstack([panel(image,frame,raw,True),panel(image,frame,display,False)])
                writer.write(combined)
                if frame in selected:storyboard.append(combined)
        finally:cap.release();writer.release()
        convert_video(path,out/'videos/identity_comparison_h264.mp4')
    cv2.imwrite(str(out/'images/identity_before_after.jpg'),np.vstack(storyboard))
    def centers(data):
        return {int(f):transform(points(row)[TORSO],np.linalg.inv(matrices[f])).mean(0)
                for f,row in data.iterrows() if start<=f<=end and f in matrices}
    old,new=centers(raw),centers(display)
    plot,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for data,color,label in [(old,'#be4040','Original full-frame pose'),(new,'#25846e','Target-locked display')]:
        frames=np.array(sorted(data));xy=np.array([data[f] for f in frames])
        ax.plot(frames/fps,xy[:,0],color=color,label=label)
    for a,b in summary['display_interpolated_ranges']:
        if b>=start and a<=end:ax.axvspan(a/fps,(b+1)/fps,color='#e6b643',alpha=.3)
    ax.set(xlabel='Video time (s)',ylabel='Torso center x in wall-reference pixels',
           title='The corrected skeleton stays with the climber; amber spans are display estimates')
    ax.legend();plot.savefig(out/'images/identity_wall_displacement.png',dpi=160);plt.close(plot)
    def max_step(data):
        frames=np.array(sorted(data));xy=np.array([data[f] for f in frames])
        return float(np.max(np.linalg.norm(np.diff(xy,axis=0),axis=1)/np.diff(frames)))
    bad_frames=int((diagnostics.raw_pose_present & ~diagnostics.raw_pose_accepted).sum())
    interpolated=int((display.pose_source=='interpolated_display_only').sum())
    accepted=pd.read_csv(out/'automatic/predictions/limb_contacts.csv')
    blocked=accepted[accepted.frame.isin(np.flatnonzero(~guard['observed']))]
    assert not (blocked.status=='accepted_model').any(), 'Occluded identity frames must not be accepted contacts.'
    metrics=dict(original_identity_rejected_frames=bad_frames,
        cropped_pose_recoveries=summary['crop_recovered_frames'],observed_pose_frames=len(observed),
        display_interpolated_frames=interpolated,wholebody_unobserved_frames=int((~guard['observed']).sum()),
        unobserved_limb_frame_predictions=len(blocked),accepted_contacts_on_guarded_frames=0,
        before_max_wall_step_px_per_frame=max_step(old),after_display_max_wall_step_px_per_frame=max_step(new),
        comparison_start_frame=start,comparison_end_frame=end,
        interpretation='Development-case diagnostics, including interpolation; not an independent pose or identity benchmark.')
    text=f'''### What happened

The original full-frame MediaPipe pass switched to a foreground passerby during
frames **184–236 (6.13–7.87 s)**. Its setting `num_poses=1` limits the number of
returned poses; it does not identify which person the project intends to track.
The existing RTMW tracker retained the climber's bounding box in the reviewed
examples, providing a useful target prior. Severe overlap also weakened RTMW
pose evidence, so its coordinates were not treated as reliable throughout.

### What changed in the pipeline

- Section 2 retains the raw pose pass and checks target identity before Section 3.
- Torso position, scale and short-term displacement are checked in wall coordinates,
  using the initial SIFT alignment to compensate camera motion. Rejected poses are
  re-detected with MediaPipe inside the tracked climber crop.
- The raw RTMW torso score threshold is 2.0, with a two-frame uncertainty buffer.
  These are engineering settings for this development clip, not calibrated probabilities.
- Gaps of at most two seconds are interpolated only when both endpoints pass identity
  checks and no joint moves more than one torso length between endpoints. Orange
  skeletons are display-only estimates. Longer or incompatible gaps remain missing.
- Contact matching and IK read observed poses only. Learned contact inference also
  rejects identity-uncertain RTMW frames. No interpolation confirms a hold or support.

### Observed results for this clip

| Diagnostic | Result |
| --- | ---: |
| Original pose frames rejected by the identity check | {bad_frames} |
| Frames recovered by target-crop detection | {summary['crop_recovered_frames']} |
| Original pose rows, including wrong-person rows | {len(raw)}/{count} |
| Corrected target-observation rows | {len(observed)}/{count} |
| Display-only interpolated frames | {interpolated} |
| RTMW frames excluded from contact evidence | {int((~guard['observed']).sum())} |
| Accepted contacts on those excluded frames | 0/{len(blocked)} |
| Maximum wall-space torso step before / after display correction, case interval | {max_step(old):.1f} / {max_step(new):.1f} px per frame |

The main display gap is **208–218**, about 0.37 seconds. Observation coverage is
lower because wrong-person and uncertain estimates no longer count as evidence.
The before/after video shows the skeleton staying with the climber; lower motion
in an interpolated display is not a pose-accuracy score. The {hold_count} route holds and
trained model weights are unchanged. Section 5 alignment and downstream contacts
are refreshed from the corrected inputs.

### Limits and provenance

This is a reviewed development case, not proof of general multi-person robustness.
The method needs a correctly initialized target track, approximate wall registration
and visible endpoints. Fast real movements or long occlusions may be rejected.
The frozen RTMW cache is preserved; the observation guard and corrected MediaPipe
tables are separate. Raw poses, settings and bootstrap alignment are retained for
reproduction. Joint visibility, grip and support continuity still need review.

MediaPipe options: [official documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).
'''
    evaluation=dict(metrics=metrics,tracking_summary=summary,notebook_markdown=text,
                    diagnostic_sha256=file_sha256(out/'tables/identity_tracking.csv'),
                    identity_guard_sha256=guard['sha256'])
    (out/'state/identity_evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n')
    def embedded(name,caption):
        data=base64.b64encode((out/'images'/name).read_bytes()).decode()
        mime='image/jpeg' if name.endswith('.jpg') else 'image/png'
        return f'<figure><img src="data:{mime};base64,{data}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
    report=out/'reports/hold_map_evaluation.html';body=report.read_text()
    panel=f'''<section id="identity-tracking"><h2>Passing-person occlusion: keep the climber identity</h2>
<p>The original full-frame MediaPipe pass switched to the passerby in frames 184–236. Target-ROI re-detection, torso-scale checks and wall-coordinate motion checks reject that switch. Short gaps appear in orange as display-only estimates.</p>
<p>{bad_frames} raw pose frames were rejected; {summary['crop_recovered_frames']} frames were recovered by cropped detection. {len(observed)}/{count} frames retain observed target poses, and {interpolated} frames use display-only interpolation. The main display gap is 208–218. RTMW contact evidence is excluded in {int((~guard['observed']).sum())} uncertain frames; none of their {len(blocked)} limb-frame predictions is accepted.</p>
<p>These are development-case diagnostics, not a pose-accuracy benchmark. The raw-score threshold is 2.0 plus a two-frame buffer. Gaps longer than two seconds or with endpoint motion exceeding one torso length are not interpolated. No estimated frame confirms contact or support. The current {hold_count}-hold route is preserved.</p>
{embedded('identity_before_after.jpg','Original full-frame tracking versus target-locked tracking; orange is display-only interpolation.')}
{embedded('identity_wall_displacement.png','Torso motion in wall-reference coordinates. Display smoothness is not a measurement of pose accuracy.')}
<p>Implementation: beta_translator/identity_tracking.py; data: state/identity_tracking_summary.json and tables/identity_tracking.csv. The full notebook report describes limitations and provenance. <a href="https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python">MediaPipe configuration reference</a>.</p></section>'''
    if '<section id="identity-tracking">' in body:body=re.sub(r'<section id="identity-tracking">.*?</section>',lambda _:panel,body,count=1,flags=re.S)
    else:body=body.replace('</main>',panel+'</main>')
    report.write_text(body)
    summary_path=out/'state/trial2_run_summary.json';current=json.loads(summary_path.read_text())
    current['identity_tracking']=metrics
    current['videos']['identity_comparison_h264.mp4']=dict(frames=end-start+1,fps=fps,
        source_start_frame=start,source_end_frame=end,sha256=file_sha256(out/'videos/identity_comparison_h264.mp4'))
    summary_path.write_text(json.dumps(current,indent=2)+'\n')
    archive_path=out/'exports/reviewed_contact_data.zip'
    with zipfile.ZipFile(archive_path) as archive:names=set(archive.namelist())
    names.update(['tables/pose_coordinates_raw.csv','tables/pose_display_coordinates.csv','tables/identity_tracking.csv',
                  'state/pose_raw_manifest.json','state/identity_anchor_alignment.npz','state/identity_guard.npz',
                  'state/identity_tracking_summary.json','state/identity_evaluation.json',
                  'images/identity_before_after.jpg','images/identity_wall_displacement.png'])
    temporary=archive_path.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):archive.write(out/name,name)
    temporary.replace(archive_path)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--outputs',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    main(args.outputs,repo/'videos/trial2.mp4')
