"""One selected complete trajectory, with per-frame evidence and explanations."""
from pathlib import Path
import json,html,textwrap
import cv2,numpy as np
from .pose_adapter import project_plane
from .project_io import convert_video


def put(img,text,x,y,size=.53,color=(45,45,45),thickness=1):
    cv2.putText(img,str(text),(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,thickness,cv2.LINE_AA)


def render(root,inputs,result,d,loads,explanations,out):
    frames=result['frames'];fps=inputs['fps'];out=Path(out);raw=out/'selected_encoding.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1440,920))
    if not writer.isOpened():raise RuntimeError('Cannot create selected-plan video.')
    cap=cv2.VideoCapture(str(Path(root)/'videos/trial1.mp4'))
    with np.load(Path(root)/'outputs/state/alignment_cache.npz',allow_pickle=False) as z:alignment=dict(zip(z['frames'],z['matrices']))
    inverse=np.linalg.inv(np.asarray(inputs['geometry']['image_to_wall_cm']))
    amber=(20,145,220);red=(50,50,210);cyan=(175,145,10)
    snapshots=[]
    try:
        for t,frame in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,source=cap.read()
            if not ok:raise RuntimeError('Missing source frame.')
            board=np.full((920,1440,3),248,np.uint8)
            put(board,f'Trial1 | {result["body"]["height_cm"]:g} cm | selected complete candidate: {result["technique_profile"]}',22,34,.8,thickness=2)
            put(board,f'Frame {frame:03d} / {frames[-1]} | {frame/fps:.2f} s | native {fps:g} FPS | fixed bone lengths',22,66,.6)
            put(board,'CONDITIONAL 2D MODEL | automatic timing | no measured effort, fatigue or 3D verification',22,98,.57,(30,95,155))
            put(board,'Observed source',22,130,.6);put(board,'Selected target skeleton',402,130,.6)
            board[145:785,20:380]=cv2.resize(source,(360,640))
            panel=np.full_like(source,252)
            transform=alignment[int(frame)]@inverse
            def xy(points):return np.rint(project_plane(np.asarray(points).reshape(-1,2),transform)).astype(int)
            for name,p in inputs['geometry']['hold_positions_cm'].items():
                if name not in inputs['selected']:continue
                point=tuple(xy([p])[0]);cv2.circle(panel,point,9,(195,199,204),-1)
                put(panel,name,point[0]+9,point[1],.45,(125,125,125))
            joints=result['joints'][t]
            for i in range(4):
                color=red if result['schedule']['known'][t,i] and d['contact_errors_cm'][t,i]>.1 else amber
                points=xy(joints[i]);cv2.polylines(panel,[points],False,color,5,cv2.LINE_AA)
                for point in points:cv2.circle(panel,tuple(point),7,color,-1,cv2.LINE_AA)
            for limb_ids in ([0,1],[2,3],[0,2],[1,3]):cv2.polylines(panel,[xy(joints[limb_ids,0])],False,(105,115,125),4,cv2.LINE_AA)
            shoulder=joints[:2,0].mean(0);hip=result['poses'][t,:2];head=shoulder+.25*(shoulder-hip)
            cv2.line(panel,tuple(xy([shoulder])[0]),tuple(xy([head])[0]),amber,5,cv2.LINE_AA)
            cv2.circle(panel,tuple(xy([head])[0]),15,amber,3,cv2.LINE_AA)
            cv2.circle(panel,tuple(xy([result['center_of_mass_cm'][t]])[0]),9,cyan,-1,cv2.LINE_AA)
            board[145:785,400:760]=cv2.resize(panel,(360,640))
            row=explanations[t];audit=d['support_audit']['frames'][t];x=790;y=165
            put(board,'INTENDED PHASE',x,y,.62,thickness=2);y+=34
            for line in textwrap.wrap(row['phase'],55):put(board,line,x,y,.58);y+=27
            y+=20;put(board,'WHY THIS PLAN',x,y,.62,thickness=2);y+=30
            for line in textwrap.wrap(row['reason'],66):put(board,line,x,y,.51);y+=25
            y+=25;put(board,'INDEPENDENT CHECKS',x,y,.62,thickness=2);y+=30
            for label,value in [('Contact gap',f'{d["max_contact_error_cm"][t]:.4f} cm / 0.1 cm'),
                 ('Support',audit['balance_audit_status']),('Unload',audit['unload_audit_status']),
                 ('Optimizer','converged' if result['optimizer']['success'] else 'NOT CONVERGED'),
                 ('Hand-load lower bound','N/A' if loads[t]['hand_l1_bw'] is None else f'{loads[t]["hand_l1_bw"]:.3f} BW (L1)'),
                 ('Foot support','Unknown/free contacts receive zero force')]:
                for line in textwrap.wrap(f'{label}: {value}',67):put(board,line,x,y,.49);y+=25
            y+=15
            for line in ['Amber skeleton: unverified geometry.', 'Red limb: contact tolerance exceeded.',
                'Cyan: modeled center of mass.', 'Zero load bound does not mean zero human effort.',
                'Original labels and history are retained.']:
                put(board,line,x,y,.47,(35,95,150));y+=26
            put(board,'One whole-sequence optimization per candidate; no concatenation of local winners.',22,825,.58)
            put(board,'Phases are proposed from contact timing; no expert technique labels have been assigned.',22,856,.54)
            put(board,'Comparisons and rejected alternatives: candidate_comparison.csv | phase_explanations.csv',22,886,.52)
            writer.write(board)
            if int(frame) in (113,160,195,275,310):
                cv2.imwrite(str(out/f'frame_{int(frame):04d}.jpg'),board);snapshots.append(cv2.resize(board,(720,460)))
                if int(frame)==195:cv2.imwrite(str(out/'poster.jpg'),board)
    finally:writer.release();cap.release()
    convert_video(raw,out/'selected_plan.mp4');raw.unlink()
    if snapshots:cv2.imwrite(str(out/'storyboard.jpg'),np.vstack(snapshots))


def evidence_sheet(root,inputs,out):
    cap=cv2.VideoCapture(str(Path(root)/'videos/trial1.mp4'));panels=[]
    for frame,note in [(107,'Prepare LH reach'),(124,'LH free / reach'),(160,'RF transfer; flag unconfirmed'),
                       (190,'LF transfer'),(215,'RH reach'),(275,'RF transfer / body rise')]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read()
        if not ok:raise RuntimeError('Missing evidence frame.')
        panel=np.full((690,360,3),250,np.uint8);panel[50:]=cv2.resize(im,(360,640));put(panel,f'Frame {frame}: {note}',10,22,.42)
        put(panel,'Observation / proposal, not expert technique label',10,43,.35,(35,95,150));panels.append(panel)
    cap.release();cv2.imwrite(str(Path(out)/'source_evidence.jpg'),np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])]))


def review_page(out,ranked,explanations):
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(m[k]))}</td>' for k in ['rank','profile','converged','contact_residual','support_balance_failed','preunload_failed','hand_integral_l1_bw_s','arm_moment_integral_bw_cm_s'])+'</tr>' for m in ranked)
    payload=json.dumps(explanations,allow_nan=False).replace('</','<\\/')
    page=r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Trial1 technique candidate review</title>
<style>body{font:16px system-ui;background:#f3f5f7;color:#202d3a;margin:24px}main{max-width:1250px;margin:auto}video{width:100%;background:#111}aside{background:#fff0ce;padding:16px;border-radius:10px}td,th{text-align:left;padding:10px;border-bottom:1px solid #ccd3db}table{border-collapse:collapse;font-size:14px}#explanation{white-space:pre-wrap;line-height:1.7}input{width:85%}</style>
<main><h1>Trial1 · one selected complete technique candidate</h1><aside>Conditional 2D model. Teaching principles generate alternatives; feasibility and evidence take priority. Lower-bound contact loads and simplified joint moments are proxies, not measured effort, fatigue or climbing success. No expert technique classifier was trained.</aside>
<p>Original source and the selected target skeleton appear side by side. All 417 native frames are retained.</p>
<video id="video" src="selected_plan.mp4" controls muted playsinline></video><p><input id="frame" type="range" min="0" max="416" value="0"><span id="number"></span></p><div id="explanation"></div>
<h2>Whole-trajectory comparison</h2><p>The hand-load model allows near-zero hand load on many frames. Candidates with similar numeric proxies are ties, not evidence of different human skill.</p><div style="overflow:auto"><table><thead><tr><th>Rank</th><th>Candidate</th><th>Converged</th><th>Contact failures</th><th>Balance failures</th><th>Unload failures</th><th>Hand L1 BW·s</th><th>Arm moment BW·cm·s</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p><a href="candidate_comparison.csv">All candidate metrics</a> · <a href="phase_explanations.csv">Phase explanations</a> · <a href="technique_review_queue.csv">Unlabeled expert-review queue</a> · <a href="source_evidence.jpg">Source evidence</a></p></main>
<script>const rows=__DATA__,v=document.getElementById('video'),f=document.getElementById('frame');function show(t){let i=Math.max(0,Math.min(rows.length-1,Math.round(t))),r=rows[i];f.value=i;document.getElementById('number').textContent='Frame '+r.frame;document.getElementById('explanation').textContent=r.phase+'\n'+r.reason+'\nSupport: '+r.support_status+' | Unload: '+r.unload_status+'\n'+r.interpretation}f.oninput=()=>{v.pause();v.currentTime=Number(f.value)/30;show(Number(f.value))};v.ontimeupdate=()=>show(v.currentTime*30);show(0);</script></html>'''
    Path(out,'review.html').write_text(page.replace('__ROWS__',rows).replace('__DATA__',payload))
