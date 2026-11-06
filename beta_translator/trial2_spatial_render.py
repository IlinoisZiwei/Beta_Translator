"""True 3D box animations and an offline orbit viewer for joined Trial2 panels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .project_io import convert_video

# Historical rendering API delegates to the shared drawing primitives.
from .body_render import COLORS,plot_xyz,bounds,clip_polygon,setup_axis,set_line,draw_frame


def read_case(out,name):
    with np.load(out/name/'trajectory.npz',allow_pickle=False) as z:fit={k:z[k] for k in z.files}
    scene=json.loads((out/name/'scene.json').read_text());metrics=pd.read_csv(out/name/'metrics.csv')
    return fit,scene,metrics


def render_outputs(project,data,out,ranked,selected):
    out=Path(out);project=Path(project)
    fit,scene,metrics=read_case(out,selected);n=len(data['frames'])
    phases=pd.read_csv(out/'phase_explanations.csv')
    fig=plt.figure(figsize=(16,10),dpi=100,facecolor='#f6f8fb')
    ax=fig.add_axes([.01,.07,.70,.83],projection='3d');artists=setup_axis(ax,scene,fit['joints'])
    source_ax=fig.add_axes([.73,.61,.25,.25]);source_ax.set_axis_off()
    source_artist=source_ax.imshow(np.zeros((270,480,3),np.uint8))
    middle,upper=scene['angles_deg'];height=ranked[0].get('height_cm',None)
    cfg=json.loads((out/selected/'summary.json').read_text())['config'];height=cfg['height_cm']
    fig.text(.04,.955,f'Trial2 | {height:g} cm | selected complete 3D plan',fontsize=22,fontweight='bold',color='#173647')
    fig.text(.04,.921,f'Assumed middle panel: {middle:g}° overhang  →  upper panel: {upper:g}°   |   fixed wall, moving body',fontsize=12)
    fig.text(.735,.865,'Source video · reported climber: 173 cm',fontsize=10)
    status=fig.text(.735,.57,'',fontsize=12,va='top',linespacing=1.7,color='#244351')
    fig.text(.04,.025,'Unmeasured geometry and contact offsets · purple: wrist/ankle connector · red/orange: contact failure · yellow: COM',fontsize=10)
    cap=cv2.VideoCapture(str(project/'videos/trial2.mp4'));raw=out/'selected_encoding.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),30,(1600,1000))
    if not writer.isOpened():raise RuntimeError('Cannot open 3D video encoder')
    keyframes=set([0,130,450,600,800,n-1])
    for t in range(n):
        ok,source=cap.read()
        if not ok:raise RuntimeError(f'Source video ended at {t}')
        draw_frame(artists,fit,t);source_artist.set_data(cv2.cvtColor(cv2.resize(source,(480,270)),cv2.COLOR_BGR2RGB))
        ax.view_init(elev=19,azim=-57+12*np.sin(2*np.pi*t/max(n-1,1)))
        row=metrics.iloc[t];unload='n/a' if pd.isna(row.preunload) else ('pass' if row.preunload else 'FAIL')
        phase=str(phases.iloc[t].phase).replace('; ','\n')
        status.set_text(f'Frame {t:03d} / {n-1}   ·   {t/30:.2f} s\n\n{phase}\n\nContact gap: {row.max_contact_gap_cm:.2f} cm\nConditional balance: {"pass" if row.conditional_balance else "FAIL"}\n±2 cm reserve: {"pass" if row.robust_2cm else "FAIL"}\nPre-release: {unload}\nMotion step: {row.step_cm:.2f} cm\n\nSolver: {"converged" if ranked[0]["converged"] else "NOT CONVERGED"}\nGeometry: UNCERTAIN\n\nSelected: {selected}')
        fig.canvas.draw();im=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
        writer.write(cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        if t in keyframes:cv2.imwrite(str(out/f'frame_{t:04d}.jpg'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
    cap.release();writer.release();plt.close(fig)
    convert_video(raw,out/'selected_plan_3d.mp4');raw.unlink()
    angle_names=[r['candidate'] for r in ranked if r['candidate'].startswith('region_m')]
    angle_names=sorted(angle_names,key=lambda name:tuple(json.loads((out/name/'scene.json').read_text())['angles_deg']))[:4]
    angle_cases=[(name,*read_case(out,name)) for name in angle_names]
    common_joints=np.concatenate([z['joints'] for _,z,_,_ in angle_cases],axis=0)
    fig=plt.figure(figsize=(16,10),dpi=100,facecolor='#f6f8fb');panels=[]
    for i,(name,z,sc,mt) in enumerate(angle_cases):
        ax=fig.add_subplot(2,2,i+1,projection='3d');a=setup_axis(ax,sc,common_joints)
        panels.append((name,z,sc,mt,ax,a));ax.tick_params(labelsize=7)
    fig.subplots_adjust(top=.90,bottom=.04,wspace=.01,hspace=.18)
    title=fig.suptitle('Trial2 | Complete 3D angle hypotheses',fontsize=18)
    raw=out/'angle_encoding.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),30,(1600,1000))
    for t in range(n):
        title.set_text(f'Trial2 | frame {t}/{n-1} | same body and contacts; angles are hypotheses')
        for name,z,sc,mt,ax,a in panels:
            draw_frame(a,z,t);m,u=sc['angles_deg'];row=mt.iloc[t]
            ax.set_title(f'Middle {m:g}° → upper {u:g}° | gap {row.max_contact_gap_cm:.2f} cm\nBalance: {"pass" if row.conditional_balance else "FAIL"} | geometry uncertain',fontsize=10)
        fig.canvas.draw();im=np.asarray(fig.canvas.buffer_rgba())[:,:,:3];writer.write(cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        if t==600:cv2.imwrite(str(out/'angle_comparison_poster.jpg'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
    writer.release();plt.close(fig);convert_video(raw,out/'angle_comparison_3d.mp4');raw.unlink()
    write_viewer(out,ranked,selected)


def write_viewer(out,ranked,selected):
    payload={}
    for row in ranked:
        name=row['candidate'];z,sc,mt=read_case(out,name);lo,hi=bounds(z['joints'])
        payload[name]=dict(scene=sc,joints=np.round(z['joints'],3).tolist(),head=np.round(z['head'],3).tolist(),
            contacts=np.round(z['contacts'],3).tolist(),known=z['known'].tolist(),
            gaps=np.round(z['contact_gaps_cm'],3).tolist(),com=np.round(z['com'],3).tolist(),
            bounds=[lo.tolist(),hi.tolist()],metrics=mt.fillna('n/a').to_dict('records'),rank=row)
    html='''<!doctype html><meta charset="utf-8"><title>Trial2 — complete 3D plans</title>
<style>body{margin:24px;font:16px system-ui;background:#f5f8fb;color:#153646}canvas{width:100%;height:72vh;background:white;border-radius:16px}select,input,button{font:inherit;padding:8px}#frame{width:45%}#status{white-space:pre-line}a{color:#126984}h1{font-size:25px}</style>
<h1>Trial2 · joined wall panels and complete 3D body plans</h1><p>Drag to orbit. Scroll to zoom. Angles, ground and joint/contact offsets are hypotheses; failures remain visible.</p>
<select id="case"></select> <button id="play">Play</button> <input id="frame" type="range" min="0" max="975" value="0"> <span id="time"></span>
<canvas id="view"></canvas><div id="status"></div><p><a href="selected_plan_3d.mp4">Selected 3D video</a> · <a href="angle_comparison_3d.mp4">Angle comparison</a> · <a href="candidate_comparison.csv">All candidates and reasons</a></p>
<script>const DATA=__PAYLOAD__,SELECTED=__SELECTED__;const el=id=>document.getElementById(id),canvas=el('view'),ctx=canvas.getContext('2d');let yaw=-.55,pitch=.2,zoom=1,drag=null,playing=false,previous=0;
for(const name of Object.keys(DATA)){const o=document.createElement('option');o.value=name;o.textContent=name;el('case').appendChild(o);}el('case').value=SELECTED;
const add=(a,b)=>a.map((v,i)=>v+b[i]),mean=a=>a[0].map((_,i)=>a.reduce((s,p)=>s+p[i],0)/a.length);
function draw(){const d=DATA[el('case').value],t=+el('frame').value,b=d.bounds,c=mean(b),span=Math.max(...b[1].map((v,i)=>v-b[0][i]));canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;const w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);
function project(p){let x=(p[0]-c[0])/span,y=(p[1]-c[1])/span,z=(p[2]-c[2])/span;let u=x*Math.cos(yaw)+z*Math.sin(yaw),v=-x*Math.sin(yaw)+z*Math.cos(yaw);let a=y*Math.cos(pitch)-v*Math.sin(pitch),depth=y*Math.sin(pitch)+v*Math.cos(pitch),s=Math.min(w,h)*1.4*zoom/(1.7-depth);return[w/2+u*s,h/2-a*s,depth];}
function line(points,color,width=2){if(!points.length)return;ctx.strokeStyle=color;ctx.lineWidth=width*devicePixelRatio;ctx.beginPath();points.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});ctx.stroke();}
function dot(p,color,r=4){let q=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],r*devicePixelRatio,0,2*Math.PI);ctx.fill();}
let polys=d.scene.meshes_cm.map((p,i)=>({p,i,z:project(mean(p))[2]})).sort((a,b)=>a.z-b.z);for(const {p,i} of polys){ctx.beginPath();p.forEach((v,k)=>{let q=project(v);k?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});ctx.closePath();ctx.fillStyle=['#60b6c133','#e6a36140','#a5acb633'][i];ctx.fill();ctx.strokeStyle=['#60b6c1','#e6a361','#a5acb6'][i];ctx.stroke();}
for(const p of d.scene.hinges_cm)line(p,'#96632f',3);
const corners=[];for(let i=0;i<8;i++)corners.push([b[(i>>0)&1][0],b[(i>>1)&1][1],b[(i>>2)&1][2]]);for(let i=0;i<8;i++)for(const bit of [1,2,4])if(!(i&bit))line([corners[i],corners[i|bit]],'#cad6df',1);
ctx.fillStyle='#426274';ctx.font=(13*devicePixelRatio)+'px system-ui';for(const [p,label] of [[corners[1],'X · width'],[corners[2],'Y · height'],[corners[4],'Z · depth']]){let q=project(p);ctx.fillText(label,q[0],q[1]);}
const j=d.joints[t];j.forEach((l,i)=>{line(l,d.known[t][i]&&d.gaps[t][i]>.1?'#d66936':'#167da1',4);l.forEach(p=>dot(p,'#167da1',3));if(d.known[t][i]){dot(d.contacts[t][i],'#148465',5);line([l[2],d.contacts[t][i]],d.gaps[t][i]>.1?'#d24b43':'#9952a6',2);}});
let hip=mean([j[2][0],j[3][0]]),shoulder=mean([j[0][0],j[1][0]]);line([hip,shoulder,d.head[t]],'#173e56',5);line([j[0][0],j[1][0]],'#173e56',4);line([j[2][0],j[3][0]],'#173e56',4);dot(d.head[t],'#173e56',9);dot(d.com[t],'#e0b623',5);
const m=d.metrics[t];el('time').textContent=`Frame ${t}/975 · ${(t/30).toFixed(2)} s`;el('status').textContent=`Assumed middle ${d.scene.angles_deg[0]}° → upper ${d.scene.angles_deg[1]}° | contact gap ${m.max_contact_gap_cm.toFixed(3)} cm | conditional balance: ${m.conditional_balance?'pass':'FAIL'}\n${m.issues}\nRank ${d.rank.rank}: ${d.rank.selection_reason}`;}
el('case').onchange=draw;el('frame').oninput=draw;el('play').onclick=()=>{playing=!playing;el('play').textContent=playing?'Pause':'Play';previous=performance.now();};
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=null;canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.3,Math.min(1.3,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw();}};canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.4,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));draw();};
function tick(now){if(playing&&now-previous>=1000/30){let step=Math.floor((now-previous)*30/1000);el('frame').value=(+el('frame').value+step)%976;previous+=step*1000/30;draw();}requestAnimationFrame(tick);}window.onresize=draw;draw();requestAnimationFrame(tick);</script>'''
    html=html.replace('__PAYLOAD__',json.dumps(payload,separators=(',',':'),allow_nan=False)).replace('__SELECTED__',json.dumps(selected))
    (out/'viewer.html').write_text(html)
