"""Native-rate selected video and offline contact-region trajectory inspection."""
from pathlib import Path
import json
import cv2,numpy as np,pandas as pd
from .scene3d_render import lines,wall_vertices,sketch,put,make_viewer,export_glb
from .scene3d import project,tilt_rotation
from .contact_regions3d import homography
from .project_io import convert_video


def region_sketch(wall,j,head,contacts,gaps,known,tilt,width=720,height=720,side=False):
    board=np.full((height,width,3),248,np.uint8);Q=tilt_rotation(tilt)
    a=np.deg2rad(90 if side else -35)
    V=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    # Fixed view framing based on the wall, including a 100 cm outward band.
    bounds=np.vstack([wall,wall+[0,0,100]])@Q.T@V.T
    lo,hi=bounds[:,:2].min(0),bounds[:,:2].max(0);center=(lo+hi)/2
    scale=min((width-40)/max(hi[0]-lo[0],50),(height-80)/max(hi[1]-lo[1],50))
    def xy(p):
        x=np.asarray(p)@Q.T@V.T
        return np.rint((x[:,:2]-center)*[scale,-scale]+[width/2,height/2+20]).astype(int)
    cv2.fillPoly(board,[xy(wall)],(222,228,231));cv2.polylines(board,[xy(wall)],True,(155,165,170),1)
    for seg in lines(j,head):cv2.polylines(board,[xy(seg)],False,(25,150,225),3,cv2.LINE_AA)
    for k in range(4):
        if not known[k]:continue
        col=(45,45,220) if gaps[k]>.1 else (180,90,150)
        cv2.polylines(board,[xy([contacts[k],j[k,2]])],False,col,3,cv2.LINE_AA)
        cv2.drawMarker(board,tuple(xy(contacts[k:k+1])[0]),col,cv2.MARKER_DIAMOND,9,2)
    put(board,'SIDE: contact to wrist/ankle' if side else '3D: contact connectors',12,25,.48)
    cv2.arrowedLine(board,(width-25,45),(width-25,95),(80,80,80),2,tipLength=.2)
    return board


def panel(source,data,z,metric,index,regions,tilt,width=720):
    board=np.full((1000,width,3),248,np.uint8)
    img=source.copy();j=z['joints_cm'][index];c=z['contacts_cm'][index];off=z['offsets_cm'][index]
    px,_=project(j,z['K'],z['R'][index],z['t_cm'][index]);cp,_=project(c,z['K'],z['R'][index],z['t_cm'][index])
    hp,_=project(z['head_cm'][index][None],z['K'],z['R'][index],z['t_cm'][index])
    for seg in lines(px,hp[0]):cv2.polylines(img,[np.rint(seg).astype(int)],False,(30,170,240),4,cv2.LINE_AA)
    frame=int(z['frames'][index]);gaps=z['contact_gaps_cm'][index]
    for i in range(4):
        if not data['known'][frame,i]:continue
        e=regions['episodes'][regions['mapping'][frame,i]];x0,y0,x1,y1=e['rectangle_px']
        rect=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        xy=homography(rect,regions['H']);rpx,_=project(np.c_[xy,np.zeros(4)],z['K'],z['R'][index],z['t_cm'][index])
        cv2.polylines(img,[np.rint(rpx).astype(int)],True,(230,195,65),2,cv2.LINE_AA)
        color=(30,50,225) if gaps[i]>.1 else (220,100,160)
        cv2.line(img,tuple(np.rint(cp[i]).astype(int)),tuple(np.rint(px[i,2]).astype(int)),color,4,cv2.LINE_AA)
        cv2.drawMarker(img,tuple(np.rint(cp[i]).astype(int)),color,cv2.MARKER_DIAMOND,14,2)
    board[80:720,:360]=cv2.resize(img,(360,640))
    view=region_sketch(wall_vertices(data),j,z['head_cm'][index],c,gaps,data['known'][frame],tilt,width=360,height=640)
    board[80:720,360:]=view
    put(board,f'Native frame {frame} / 502 | {frame/data["fps"]:.2f} s',15,30,.65)
    put(board,'CYAN: evidence patch | PINK: joint-to-contact connector',15,60,.46)
    checks=[f'Contact gap: {metric.max_contact_gap_cm:.3f} cm (limit 0.1)',
        f'Conditional balance: {metric.conditional_balance} | reserve: {metric.conditional_robust_2cm}',
        f'Pre-release remaining support: {metric.preunload_without_departing}',
        f'Joint step: {metric.joint_step_cm:.2f} cm | gradient convergence: {metric.optimizer_converged}',
        'Unmeasured wall angle, camera, surfaces and hand/foot offsets.',
        'Source height unknown. Geometry and technique remain UNVERIFIED.']
    for k,text in enumerate(checks):put(board,text,15,755+30*k,.47,(50,50,200) if k<4 and metric.status=='constraint_warning' else (45,95,130))
    issues=str(metric.issues)
    for k in range(0,min(len(issues),180),90):put(board,issues[k:k+90],15,950+(k//90)*22,.39,(40,40,210))
    return board


def render_region_outputs(project,data,regions,out,cases,selected):
    out=Path(out);selected_case=out/selected;base_case=out/'center' if (out/'center').exists() else cases[0]
    info=json.loads((selected_case/'summary.json').read_text());tilt=info['tilt_deg'];height=info['height_cm']
    sel=dict(np.load(selected_case/'trajectory.npz'));base=dict(np.load(base_case/'trajectory.npz'))
    metrics=pd.read_csv(selected_case/'metrics.csv').fillna('n/a');bm=pd.read_csv(base_case/'metrics.csv').fillna('n/a')
    reasons=pd.read_csv(out/'phase_explanations.csv').fillna('')
    cap=cv2.VideoCapture(str(Path(project)/'videos/trial3.mp4'))
    raw=out/'selected_plan_raw.mp4';raw_compare=out/'comparison_raw.mp4'
    video=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),data['fps'],(1440,1080))
    comparison=cv2.VideoWriter(str(raw_compare),cv2.VideoWriter_fourcc(*'mp4v'),data['fps'],(1440,1080))
    if not cap.isOpened() or not video.isOpened() or not comparison.isOpened():raise RuntimeError('Video IO unavailable.')
    for i,frame in enumerate(data['frames']):
        ok,source=cap.read()
        if not ok:raise RuntimeError('Missing native source frame '+str(frame))
        left=panel(source,data,sel,metrics.iloc[i],i,regions,tilt)
        before=panel(source,data,base,bm.iloc[i],i,regions,tilt)
        b=np.full((1080,1440,3),248,np.uint8);b[:1000,:720]=left
        b[:720,720:]=region_sketch(wall_vertices(data),sel['joints_cm'][i],sel['head_cm'][i],sel['contacts_cm'][i],
            sel['contact_gaps_cm'][i],data['known'][i],tilt,width=720,height=720,side=True)
        put(b,f'Selected complete plan: {selected}',745,760,.65)
        put(b,'Selection compares all 503 frames, not local winners.',745,800,.50)
        phase=str(reasons.iloc[i].phase)
        for k in range(0,len(phase),65):put(b,phase[k:k+65],745,845+25*(k//65),.48)
        put(b,'Least-failing tested candidate; NOT a validated climb.',745,945,.49,(45,85,190))
        put(b,f'Trial3 | {height:g} cm | assumed {tilt:g} deg tilt | native {data["fps"]:g} FPS',20,1035,.62)
        put(b,'Red = failed check. Amber = uncertain assumptions. Surface contacts stay fixed within each episode.',20,1065,.48)
        video.write(b)
        comp=np.full((1080,1440,3),248,np.uint8);comp[:1000,:720]=before;comp[:1000,720:]=left
        put(comp,f'LEFT: {base_case.name}   |   RIGHT: {selected}',20,1035,.65)
        put(comp,'Same body, frames, labels, camera and thresholds. Contact representation changes explicitly.',20,1065,.49)
        comparison.write(comp)
        if frame in (0,314,486,502):cv2.imwrite(str(out/f'frame_{frame:04d}.jpg'),b)
        if frame==486:cv2.imwrite(str(out/'comparison_poster.jpg'),comp)
    cap.release();video.release();comparison.release()
    convert_video(raw,out/'selected_plan.mp4');convert_video(raw_compare,out/'comparison.mp4');raw.unlink();raw_compare.unlink()
    # Reuse the existing orbit UI, extend its embedded data with actual contact
    # points and virtual connectors, and replace all legacy artifact links.
    make_viewer(out,data,cases)
    html=(out/'viewer.html').read_text();tag='<script id="scene-data" type="application/json">';start=html.index(tag)+len(tag);end=html.index('</script>',start)
    payload=json.loads(html[start:end])
    for record,case in zip(payload['cases'],cases):
        z=np.load(case/'trajectory.npz');record['contacts']=z['contacts_cm'].round(5).tolist();record['known']=z['known'].tolist()
    html=html[:start]+json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/')+html[end:]
    marker="ctx.fillStyle='#203241';ctx.font='18px system-ui';"
    connectors="for(let k=0;k<4;k++){if(s.known[i][k]){path([s.contacts[i][k],j[k][2]],'#a75bc4',3);let p=point(s.contacts[i][k]);ctx.fillStyle='#7b36ab';ctx.fillRect(p[0]-4,p[1]-4,8,8)}}\n"
    html=html.replace(marker,connectors+marker)
    html=html.replace('scenario_metrics.csv','candidate_comparison.csv').replace('failure_review.csv',selected+'/metrics.csv')
    html=html.replace('Scenario metrics','Complete candidate comparison').replace('Failure review','Selected-frame checks')
    html=html.replace('Drag to orbit;', 'Purple connectors separate joint centers from surface contacts; their lengths and directions are assumptions. Each contact is episode-constant. Drag to orbit;')
    html=html.replace('<div class="row"><label>Scenario','<p><a href="selected_plan.mp4">Selected full video</a> · <a href="comparison.mp4">Center versus selected video</a> · <a href="phase_explanations.csv">Phase explanations</a> · <a href="angle_sensitivity.csv">Angle sensitivity</a></p><div class="row"><label>Candidate')
    # Open the actual selected candidate first, preserve access to every branch.
    html=html.replace('select();requestAnimationFrame(tick);',f'S.value="{[c.name for c in cases].index(selected)}";select();requestAnimationFrame(tick);')
    (out/'viewer.html').write_text(html)
    for frame in (0,314,486,502):
        export_glb(out/f'selected_{frame:04d}.glb',wall_vertices(data),sel['joints_cm'][frame],sel['head_cm'][frame],tilt,
            {'native_frame':frame,'selected_candidate':selected,'contact_points_wall_cm':sel['contacts_cm'][frame].tolist(),
             'joint_to_contact_offsets_cm':sel['offsets_cm'][frame].tolist(),'offsets_measured':False})
