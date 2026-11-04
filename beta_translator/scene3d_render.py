"""Offline XYZ viewer, actual glTF meshes and diagnostic video exports."""
from pathlib import Path
import json,struct
import cv2,numpy as np,pandas as pd
from .scene3d import project,tilt_rotation
from .project_io import convert_video

AMBER=(25,150,225);RED=(55,50,210);INK=(45,45,45)


def lines(joints,head):
    j=np.asarray(joints)
    return [*(j[i,k:k+2] for i in range(4) for k in range(2)),
            j[[0,1],0],j[[2,3],0],j[[0,2],0],j[[1,3],0],np.stack([j[:2,0].mean(0),head])]


def wall_vertices(data):
    from .pose_adapter import project_plane
    obs=data['old']['calibration']
    xy=project_plane(obs['plane_polygon_xy'],obs['image_to_wall_cm'])
    return np.c_[xy,np.zeros(len(xy))]


def cylinder(a,b,radius=.018,sides=10):
    delta=b-a;length=np.linalg.norm(delta)
    if length<1e-8:return np.empty((0,3)),np.empty((0,3),int)
    direction=delta/length;helper=np.array([0.,0.,1.]) if abs(direction[2])<.9 else np.array([1.,0.,0.])
    u=np.cross(direction,helper);u/=np.linalg.norm(u);v=np.cross(direction,u)
    ring=np.array([np.cos(t)*u+np.sin(t)*v for t in np.linspace(0,2*np.pi,sides,endpoint=False)])*radius
    vertices=np.vstack([a+ring,b+ring,a[None],b[None]])
    faces=[]
    for i in range(sides):
        k=(i+1)%sides;faces.extend([[i,k,sides+i],[k,sides+k,sides+i],[2*sides,k,i],[2*sides+1,sides+i,sides+k]])
    return vertices,np.array(faces,int)


def export_glb(path,wall,joints,head,tilt,metadata):
    """glTF meters, world Y-up, cylinders depict a skeleton rather than a body scan."""
    Q=tilt_rotation(tilt);wall=np.asarray(wall)@Q.T/100.
    vertices=[wall];faces=[np.array([[0,i,i+1] for i in range(1,len(wall)-1)],int)]
    colors=[np.tile([.75,.78,.79,1.],(len(wall),1))]
    count=len(wall)
    for xy in metadata.get('hold_markers_wall_xy_cm',{}).values():
        center=np.r_[xy,0.]/100.;a=np.linspace(0,2*np.pi,12,endpoint=False)
        marker=np.vstack([center,center+np.c_[.035*np.cos(a),.035*np.sin(a),np.zeros(len(a))]])@Q.T
        triangles=np.array([[0,i+1,(i+1)%len(a)+1] for i in range(len(a))],int)
        vertices.append(marker);faces.append(triangles+count);colors.append(np.tile([.23,.52,.65,1.],(len(marker),1)));count+=len(marker)
    for segment in lines(joints,head):
        a,b=segment@Q.T/100.;v,f=cylinder(a,b)
        vertices.append(v);faces.append(f+count);colors.append(np.tile([.92,.53,.08,1.],(len(v),1)));count+=len(v)
    v=np.concatenate(vertices).astype('<f4');f=np.concatenate(faces).astype('<u4').ravel();c=np.concatenate(colors).astype('<f4')
    normal=np.zeros_like(v)
    for tri in f.reshape(-1,3):
        n=np.cross(v[tri[1]]-v[tri[0]],v[tri[2]]-v[tri[0]])
        normal[tri]+=n
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    blobs=[v.tobytes(),normal.tobytes(),c.tobytes(),f.tobytes()];offset=0;views=[]
    for b in blobs:views.append({'buffer':0,'byteOffset':offset,'byteLength':len(b)});offset+=len(b)
    doc={'asset':{'version':'2.0','generator':'Beta Translator conditional 3D experiment'},
         'extras':{**metadata,'units':'meters','wall_tilt_measured':False,'surface':'Skeleton cylinders and planar wall; no measured skin or hold shape.'},
         'buffers':[{'byteLength':offset}],'bufferViews':views,
         'accessors':[{'bufferView':0,'componentType':5126,'count':len(v),'type':'VEC3','min':v.min(0).tolist(),'max':v.max(0).tolist()},
                      {'bufferView':1,'componentType':5126,'count':len(v),'type':'VEC3'},
                      {'bufferView':2,'componentType':5126,'count':len(v),'type':'VEC4'},
                      {'bufferView':3,'componentType':5125,'count':len(f),'type':'SCALAR'}],
         'materials':[{'doubleSided':True,'pbrMetallicRoughness':{'metallicFactor':0,'roughnessFactor':.8}}],
         'meshes':[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1,'COLOR_0':2},'indices':3,'material':0}]}],
         'nodes':[{'mesh':0}],'scenes':[{'nodes':[0]}],'scene':0}
    js=json.dumps(doc,separators=(',',':')).encode();js+=b' '*((-len(js))%4)
    binary=b''.join(blobs);binary+=b'\0'*((-len(binary))%4)
    total=12+8+len(js)+8+len(binary)
    Path(path).write_bytes(struct.pack('<III',0x46546C67,2,total)+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binary),0x004E4942)+binary)


def put(img,text,x,y,scale=.48,color=INK):cv2.putText(img,str(text),(x,y),0,scale,color,1,cv2.LINE_AA)


def sketch(wall,joints,head,targets,gaps,known,tilt,width=300,height=535,side=False):
    canvas=np.full((height,width,3),248,np.uint8);Q=tilt_rotation(tilt)
    # An orthographic overview is an inspection view, not the source camera.
    a=np.deg2rad(-35 if not side else 90)
    view=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    allpoints=np.vstack([wall,joints.reshape(-1,3),head[None]])@Q.T@view.T
    lo=allpoints[:,:2].min(0);hi=allpoints[:,:2].max(0)
    scale=min((width-30)/max(hi[0]-lo[0],30),(height-65)/max(hi[1]-lo[1],30))
    center=(lo+hi)/2
    def xy(p):
        z=np.asarray(p)@Q.T@view.T
        return np.rint((z[:,:2]-center)*[scale,-scale]+[width/2,height/2+10]).astype(int)
    cv2.fillPoly(canvas,[xy(wall)],(222,228,231));cv2.polylines(canvas,[xy(wall)],True,(155,165,170),1,cv2.LINE_AA)
    for segment in lines(joints,head):cv2.polylines(canvas,[xy(segment)],False,AMBER,3,cv2.LINE_AA)
    for i in range(4):
        color=RED if known[i] and gaps[i]>.1 else AMBER
        cv2.polylines(canvas,[xy(joints[i])],False,color,3,cv2.LINE_AA)
        for p in xy(joints[i]):cv2.circle(canvas,tuple(p),3,color,-1,cv2.LINE_AA)
        if known[i]:cv2.drawMarker(canvas,tuple(xy(targets[i:i+1])[0]),(80,80,80),cv2.MARKER_CROSS,9,1)
    put(canvas,'SIDE (gravity-aligned)' if side else '3D inspection view',10,23,.46)
    cv2.arrowedLine(canvas,(width-22,40),(width-22,90),(90,90,90),2,tipLength=.2);put(canvas,'g',width-20,107)
    return canvas


def case_panel(source,data,z,row,info,index,width=600):
    frame=int(z['frames'][index]);j=z['joints_cm'][index];head=z['head_cm'][index];gap=z['contact_gaps_cm'][index]
    px,_=project(j,z['K'],z['R'][index],z['t_cm'][index]);img=source.copy()
    for seg in lines(px,project(head[None],z['K'],z['R'][index],z['t_cm'][index])[0][0]):
        cv2.polylines(img,[np.rint(seg).astype(int)],False,AMBER,5,cv2.LINE_AA)
    for i in range(4):
        color=RED if data['known'][frame,i] and gap[i]>.1 else AMBER
        cv2.polylines(img,[np.rint(px[i]).astype(int)],False,color,5,cv2.LINE_AA)
    board=np.full((700,600,3),248,np.uint8)
    put(board,f"Tilt {info['tilt_deg']:g} deg | depth prior {info['depth_prior_cm']:g} cm",12,24,.59)
    put(board,f"{info['height_cm']:g} cm target | frame {frame} | NOT measured 3D",12,47,.48)
    board[65:600,:300]=cv2.resize(img,(300,535))
    board[65:600,300:]=sketch(wall_vertices(data),j,head,z['targets_cm'][index],gap,data['known'][frame],info['tilt_deg'],side=True)
    put(board,f"Contact gap {row['max_contact_gap_cm']:.3f} cm | actual hip depth {row['hip_depth_cm']:.1f} cm",12,624,.49)
    put(board,f"Conditional balance: {row['conditional_balance']} | solver: {row['optimizer_converged']}",12,647,.49)
    put(board,'Amber: unverified | Red: missed contact | normals assumed',12,674,.46,(30,90,175))
    return board


def render_comparison(project,data,cases,destination,playback_fps=10):
    """One comparison frame per native source frame. No montage or stitched poses."""
    loaded=[]
    for case in cases:
        z=np.load(Path(case)/'trajectory.npz',allow_pickle=False)
        loaded.append((z,pd.read_csv(Path(case)/'metrics.csv'),json.loads((Path(case)/'summary.json').read_text())))
    frames=loaded[0][0]['frames'];assert all(np.array_equal(z['frames'],frames) for z,_,_ in loaded)
    destination=Path(destination);raw=destination.with_name(destination.stem+'_encoding.mp4')
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),playback_fps,(600*len(cases),700))
    if not writer.isOpened():raise RuntimeError('Cannot initialize comparison video writer.')
    cap=cv2.VideoCapture(str(Path(project)/'videos/trial3.mp4'))
    for t,f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,img=cap.read()
        if not ok:raise RuntimeError('Source video frame unavailable.')
        board=np.hstack([case_panel(img,data,z,table.iloc[t],info,t) for z,table,info in loaded])
        writer.write(board)
        if t==len(frames)//2:cv2.imwrite(str(destination.with_suffix('.jpg')),board)
    writer.release();cap.release();convert_video(raw,destination);raw.unlink()


def make_viewer(out,data,cases):
    """Self-contained Canvas XYZ viewer: no CDN, server, telemetry or network fetch."""
    out=Path(out);payload={'wall':wall_vertices(data).tolist(),'holds':data['old']['geometry']['hold_positions_cm'],'cases':[]}
    for case in cases:
        info=json.loads((case/'summary.json').read_text());metrics=pd.read_csv(case/'metrics.csv').fillna('not applicable')
        with np.load(case/'trajectory.npz') as z:
            payload['cases'].append({'name':case.name,'tilt':info['tilt_deg'],'depth':info['depth_prior_cm'],
                'frames':z['frames'].tolist(),'joints':z['joints_cm'].round(5).tolist(),'head':z['head_cm'].round(5).tolist(),
                'metrics':metrics.to_dict('records')})
    encoded=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    target_height=json.loads((cases[0]/'summary.json').read_text())['height_cm']
    html='''<!doctype html><meta charset="utf-8"><title>Trial3 — conditional 3D laboratory</title>
<style>body{font:16px system-ui;background:#f2f4f6;color:#202c38;margin:24px}main{max-width:1250px;margin:auto}h1{font-size:27px}select,button,input{font:inherit;margin:8px;padding:6px}canvas{width:100%;background:white;border:1px solid #ccd5df;border-radius:12px;touch-action:none}aside{background:#fff3d8;padding:14px;border-radius:8px}#stats{white-space:pre-wrap;font:14px ui-monospace;line-height:1.7}.row{display:flex;flex-wrap:wrap;align-items:center}a{color:#175b91}</style>
<main><h1>Trial3 · __HEIGHT__ cm · conditional 3D experiment</h1>
<aside>All wall tilts, body depths, camera intrinsics and contact normals are unmeasured hypotheses. This is an articulated 3D skeleton and planar scene, not a recovered human scan. Amber is uncertain; red is a missed contact. Drag to orbit; use the slider to inspect native frames.</aside>
<div class="row"><label>Scenario <select id="scenario"></select></label><button id="play">Play</button><button id="reset">Reset view</button><label>Zoom <input id="zoom" type="range" min="0.5" max="2.5" step="0.05" value="1"></label></div>
<div class="row"><input id="frame" type="range" min="0" value="0" style="flex:1"><span id="label"></span></div>
<canvas id="view" width="1200" height="760"></canvas><div id="stats"></div>
<p>World Y points up. Depth is distance along the wall normal. The origin is a grid seed, not the floor. Current model omits true hold shape, grip strength, measured normals, inertia and self-collision. Numerical convergence is not physical validation.</p>
<p><a href="scenario_metrics.csv">Scenario metrics</a> · <a href="failure_review.csv">Failure review</a> · <a href="camera.json">Camera assumptions</a></p></main>
<script id="scene-data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('scene-data').textContent),S=document.getElementById('scenario'),F=document.getElementById('frame'),C=document.getElementById('view'),ctx=C.getContext('2d');
let yaw=-0.55,pitch=0.15,playing=false,drag=null,last=0;
D.cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c.name;S.add(o)});
function scenario(){return D.cases[Number(S.value)||0]}
function select(){F.max=scenario().frames.length-1;F.value=0;draw()}S.onchange=select;F.oninput=draw;document.getElementById('zoom').oninput=draw;
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play'};
document.getElementById('reset').onclick=()=>{yaw=-.55;pitch=.15;draw()};
C.onpointerdown=e=>{drag=[e.clientX,e.clientY];C.setPointerCapture(e.pointerId)};
C.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.006;pitch=Math.max(-1.2,Math.min(1.2,pitch+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];draw()}};C.onpointerup=()=>drag=null;
function world(p,t){let a=t*Math.PI/180;return[p[0],p[1]*Math.cos(a)-p[2]*Math.sin(a),p[1]*Math.sin(a)+p[2]*Math.cos(a)]}
function draw(){let s=scenario(),i=Number(F.value),j=s.joints[i],m=s.metrics[i],tilt=s.tilt,center=world([0,-180,0],tilt),scale=1.45*Number(document.getElementById('zoom').value);
function point(p){let w=world(p,tilt).map((v,k)=>v-center[k]);let x=w[0]*Math.cos(yaw)+w[2]*Math.sin(yaw),z=-w[0]*Math.sin(yaw)+w[2]*Math.cos(yaw),y=w[1]*Math.cos(pitch)-z*Math.sin(pitch);return[600+x*scale,380-y*scale]}
function path(ps,color,width=3,fill=false){ctx.beginPath();ps.map(point).forEach((p,k)=>k?ctx.lineTo(...p):ctx.moveTo(...p));ctx.strokeStyle=color;ctx.lineWidth=width;if(fill){ctx.closePath();ctx.fillStyle=color;ctx.fill()}else ctx.stroke()}
ctx.clearRect(0,0,1200,760);path(D.wall,'#dce5e9',1,true);path([...D.wall,D.wall[0]],'#9aacb5',1);
ctx.font='13px system-ui';for(let [name,p]of Object.entries(D.holds)){let a=point([...p,0]);ctx.fillStyle='#5b6976';ctx.fillRect(a[0]-3,a[1]-3,6,6);ctx.fillText(name,a[0]+7,a[1])}
for(let chain of j){path(chain,m.max_contact_gap_cm>.1?'#c83e47':'#d78c13',5);for(let p of chain){let a=point(p);ctx.beginPath();ctx.arc(...a,5,0,2*Math.PI);ctx.fillStyle='#f5b546';ctx.fill()}}
path([j[0][0],j[1][0],j[3][0],j[2][0],j[0][0]],'#6c7d89',4);path([j[0][0].map((x,k)=>(x+j[1][0][k])/2),s.head[i]],'#d78c13',5);
ctx.fillStyle='#203241';ctx.font='18px system-ui';ctx.fillText(`Tilt ${tilt}° | hip depth prior ${s.depth} cm | native frame ${s.frames[i]}`,25,32);
ctx.fillText('UNVERIFIED 3D HYPOTHESIS',25,65);document.getElementById('label').textContent=`Frame ${s.frames[i]}`;
document.getElementById('stats').textContent=`Contact gap: ${m.max_contact_gap_cm.toFixed(3)} cm   Bone-length error: ${m.max_segment_error_cm.toExponential(2)} cm\nActual hip depth: ${m.hip_depth_cm.toFixed(2)} cm   Conditional balance: ${m.conditional_balance}   ±2 cm reserve: ${m.conditional_robust_2cm}\nOptimizer converged: ${m.optimizer_converged}   Source-guide discrepancy: ${m.source_guide_rmse_px.toFixed(1)} px (not target-user accuracy)\nMinimum joint wall clearance: ${m.min_wall_clearance_cm.toFixed(2)} cm   Low-confidence contacts: ${m.low_confidence_contacts}`}
function tick(now){if(playing&&now-last>1000/30){F.value=(Number(F.value)+1)%(Number(F.max)+1);last=now;draw()}requestAnimationFrame(tick)}select();requestAnimationFrame(tick);
</script>'''.replace('__DATA__',encoded).replace('__HEIGHT__',f'{target_height:g}')
    (out/'viewer.html').write_text(html)
