"""Shared fixed-camera 3D rendering of saved poses; no smoothing or interpolation."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .project_io import file_sha256,convert_video

COLORS=['#60b6c1','#e6a361','#a5acb6']


def plot_xyz(p):
    a=np.asarray(p)/100.
    return a[...,0],a[...,2],a[...,1]


def bounds(joints):
    p=np.asarray(joints).reshape(-1,3)
    lo=np.min(p,axis=0)-[45,25,45];hi=np.max(p,axis=0)+[45,55,45]
    lo[1]=0
    return lo,hi


def clip_polygon(poly,lo,hi):
    points=list(np.asarray(poly))
    for axis in range(3):
        for limit,sign in [(lo[axis],1),(hi[axis],-1)]:
            result=[]
            if not points:break
            for a,b in zip(points,points[1:]+points[:1]):
                va=sign*(a[axis]-limit);vb=sign*(b[axis]-limit)
                if va>=0:result.append(a)
                if (va>=0)!=(vb>=0):result.append(a+(b-a)*va/(va-vb))
            points=result
    return np.array(points)


def setup_axis(ax,scene,joints):
    lo,hi=bounds(joints)
    for i,poly in enumerate(scene['meshes_cm']):
        poly=clip_polygon(poly,lo,hi)
        if len(poly)<3:continue
        xyz=np.stack(plot_xyz(poly),axis=-1)
        ax.add_collection3d(Poly3DCollection([xyz],facecolor=COLORS[i],alpha=.22,edgecolor=COLORS[i],linewidth=.9))
    for h in scene['hinges_cm']:
        h=np.asarray(h);a,b=h
        # Clip the long hinge to the displayed route corridor.
        t=np.clip((np.array([lo[0],hi[0]])-a[0])/(b[0]-a[0]),0,1)
        a,b=a+t[:,None]*(b-a)
        ax.plot(*plot_xyz(np.array([a,b])),color='#865a26',linewidth=3)
    ax.set(xlim=(lo[0]/100,hi[0]/100),ylim=(lo[2]/100,hi[2]/100),zlim=(0,hi[1]/100),
        xlabel='Wall width X (m)',ylabel='Depth Z (m)',zlabel='Height Y (m)')
    size=(hi-lo)/100;ax.set_box_aspect([size[0],size[2],size[1]])
    ax.view_init(elev=19,azim=-57);ax.grid(True,alpha=.25)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.set_facecolor((.96,.97,.99,.12))
    lines=[ax.plot([],[],[],color='#167da1',linewidth=3,marker='o',markersize=3)[0] for _ in range(4)]
    trunk=ax.plot([],[],[],color='#143f59',linewidth=4)[0]
    shoulder=ax.plot([],[],[],color='#143f59',linewidth=3)[0]
    hips=ax.plot([],[],[],color='#143f59',linewidth=3)[0]
    head=ax.plot([],[],[],color='#143f59',marker='o',markersize=8)[0]
    contacts=[ax.plot([],[],[],marker='s',color='#0c8b65',markersize=5,linewidth=0)[0] for _ in range(4)]
    connectors=[ax.plot([],[],[],color='#9859a1',linewidth=1.5)[0] for _ in range(4)]
    com=ax.plot([],[],[],marker='o',color='#e2b622',markersize=6)[0]
    shoes=[ax.plot([],[],[],color='#119b9b',linewidth=2)[0] for _ in range(2)]
    return dict(shoes=shoes,lines=lines,trunk=trunk,shoulder=shoulder,hips=hips,head=head,contacts=contacts,connectors=connectors,com=com)


def set_line(line,points):
    x,y,z=plot_xyz(points);line.set_data(x,y);line.set_3d_properties(z)


def draw_frame(artists,fit,t):
    j=fit['joints'][t];gaps=fit['contact_gaps_cm'][t];known=fit['known'][t]
    for l,line in enumerate(artists['lines']):
        set_line(line,j[l]);line.set_color('#d66936' if known[l] and gaps[l]>.1 else '#167da1')
        set_line(artists['contacts'][l],[fit['contacts'][t,l]] if known[l] else np.empty((0,3)))
        set_line(artists['connectors'][l],[j[l,2],fit['contacts'][t,l]] if known[l] else np.empty((0,3)))
        artists['connectors'][l].set_color('#cc443e' if gaps[l]>.1 else '#9859a1')
    if 'femur_distal' in fit:
        for side in range(2):
            set_line(artists['lines'][side+2],[j[side+2,0],fit['femur_distal'][t,side],j[side+2,1],j[side+2,2]])
    if 'foot_toe' in fit:
        for side,line in enumerate(artists['shoes']):
            a=fit['foot_heel'][t,side];b=fit['foot_toe'][t,side];w=float(fit.get('shoe_width_cm',4.))/2*fit['foot_frames'][t,side,:,1]
            set_line(line,[a+w,b+w,b-w,a-w,a+w])
    hip=j[2:,0].mean(0);shoulder=j[:2,0].mean(0)
    set_line(artists['trunk'],[hip,shoulder,fit['head'][t]])
    set_line(artists['shoulder'],j[:2,0]);set_line(artists['hips'],j[2:,0]);set_line(artists['head'],[fit['head'][t]])
    set_line(artists['com'],[fit['com'][t]])


def render_run(output):
    from .body_pipeline import verified_run,write_json
    out=Path(output);summary=verified_run(out)
    identity={'summary':file_sha256(out/'summary.json'),'renderer':file_sha256(Path(__file__))}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:10]
    dest=out/f'presentation_{key}'
    if (dest/'manifest.json').exists():
        manifest=json.loads((dest/'manifest.json').read_text())
        for f,h in manifest['output_sha256'].items():
            if file_sha256(dest/f)!=h:raise ValueError('Changed presentation: '+f)
        return dest
    dest.mkdir(exist_ok=True)
    names=[r['candidate'] for r in [*summary['ranking'],*summary.get('controls',[])]];scene=json.loads((out/'scene.json').read_text())
    cases=[]
    for name in names:
        with np.load(out/name/'trajectory.npz',allow_pickle=False) as z:fit={k:z[k] for k in z.files}
        profile=summary['provenance']['config'].get('user_profile')
        if profile:fit['shoe_width_cm']=profile['dimensions_cm']['foot_width']
        cases.append((name,fit,pd.read_csv(out/name/'metrics.csv')))
    common=np.concatenate([z['joints'] for _,z,_ in cases])
    for comparison in (False,True):
        chosen=([cases[-1],cases[0]] if summary.get('controls') else cases) if comparison else cases[:1];count=len(chosen)
        fig=plt.figure(figsize=(7*count,7),dpi=100,facecolor='#f6f8fb');panels=[]
        for i,(name,z,m) in enumerate(chosen):
            ax=fig.add_axes([i/count,.30,.97/count,.55],projection='3d')
            art=setup_axis(ax,scene,common)
            low=float(common[...,1].min()-30)/100;high=float(common[...,1].max()+40)/100
            ax.set_zlim(low,high);ax.set_box_aspect([np.ptp(ax.get_xlim()),np.ptp(ax.get_ylim()),high-low])
            ax.set_title(name+(' (selected)' if name==summary['selected_candidate'] else ''),fontsize=12)
            hands=[ax.plot([],[],[],color='#873bab',linewidth=2)[0] for _ in range(4)]
            status=fig.text((i+.06)/count,.23,'',fontsize=9,va='top')
            panels.append((art,hands,status))
        title=fig.text(.035,.965,'',fontsize=15,fontweight='bold')
        fig.text(.035,.92,f"{summary['height_cm']:g} cm | {scene['kind'] if 'kind' in scene else 'joined wall panels'} | fixed camera, native timestamps",fontsize=10)
        profile=summary['provenance']['config'].get('user_profile')
        note=(f"Profile {profile['user_id']}: arm span {profile['arm_span_cm']:g} cm; uncalibrated mechanics / fatigue proxies." if profile else 'Diagnostic only: geometry and contact orientation inferred. Full physiology remains unknown.')
        fig.text(.035,.04,note,fontsize=9)
        fig.text(.035,.015,'Purple: joint to patch. Red: patch mismatch. Green: contact. Orange: contact failure.',fontsize=9)
        raw=dest/'encoding.mp4';video=dest/('candidate_comparison_3d.mp4' if comparison else 'selected_plan_3d.mp4')
        fig.canvas.draw();size=fig.canvas.get_width_height()
        writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),summary['fps'],size)
        if not writer.isOpened():raise RuntimeError('Cannot create video')
        try:
            for t in range(summary['frame_count']):
                frame=int(cases[0][1]['frames'][t]);title.set_text(f"{summary['trial'].title()} | frame {frame} | {frame/summary['fps']:.2f} s")
                for (name,z,m),(art,patch,status) in zip(chosen,panels):
                    draw_frame(art,z,t)
                    for limb in range(4):
                        active=z['known'][t,limb];point=z['contact_points'][t,limb]
                        set_line(patch[limb],[z['joints'][t,limb,2],point] if active else np.empty((0,3)))
                        set_line(art['connectors'][limb],[point,z['contacts'][t,limb]] if active else np.empty((0,3)))
                    row=m.iloc[t];release='n/a' if pd.isna(row.preunload) else ('pass' if row.preunload and not row.departing_force_failed else 'FAIL')
                    status.set_text(f"Patch gap: {row.max_contact_gap_cm:.3f} cm (limit 0.1)\nSupport: {'pass' if row.conditional_balance else 'FAIL'} | release: {release}\nROM/rates: {'FAIL' if row.anatomy_bounds_failed or row.rate_failed or row.angular_acceleration_failed else 'pass'}\nAnkle ROM: {'checked' if row.ankle_rom_checked else 'unchecked'} | foot pose: inferred")
                    if 'advanced_failure' in m:
                        state=lambda col:'FAIL' if row[col] else 'pass'
                        status.set_text(status.get_text()+f"\nShoulder: {state('shoulder_envelope_failed')} | collision: {state('self_collision_failed')} | torque: {state('torque_capacity_failed')}\nFatigue/capacity: {state('fatigue_failed')} | inertia balance: {state('dynamic_balance_failed')}")
                fig.canvas.draw();writer.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:,:,:3],cv2.COLOR_RGB2BGR))
                if t==summary['frame_count']//2:fig.savefig(video.with_suffix('.png'),dpi=100)
        finally:writer.release();plt.close(fig)
        convert_video(raw,video);raw.unlink()
        cap=cv2.VideoCapture(str(video));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        if n!=summary['frame_count'] or abs(fps-summary['fps'])>1e-6:raise RuntimeError('Video time axis mismatch')
    fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True,layout='constrained')
    for name,z,m in cases:
        t=z['frames']/summary['fps'];axes[0].plot(t,m.max_contact_gap_cm,label=name)
        axes[1].step(t,(~m.conditional_balance).astype(int),where='mid')
        axes[2].plot(t,m.max_rate_ratio)
    axes[0].axhline(.1,color='gray',linestyle='--');axes[0].legend()
    for ax,label in zip(axes,['Patch gap (cm)','Support failure','Rate / cap']):ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[2].set_xlabel('Original video time (s)');fig.savefig(dest/'metrics.png',dpi=140);plt.close(fig)
    write_json(dest/'manifest.json',dict(provenance=identity,frames=summary['frame_count'],fps=summary['fps'],
        output_sha256={f.name:file_sha256(f) for f in dest.iterdir() if f.is_file()}))
    return dest
