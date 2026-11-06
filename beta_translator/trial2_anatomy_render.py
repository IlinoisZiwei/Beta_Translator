"""Fixed-camera before/after videos; no trajectory resampling or pose smoothing."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .trial2_spatial_render import setup_axis,draw_frame,set_line,read_case
from .trial2_spatial_pipeline import load_fit,write_json
from .trial2_anatomy_pipeline import LEGACY,CLIPS,slice_arrays
from .anatomical_arms import elbow_swivel_rates,ARM_NAMES,ARM_LIMITS_DEG,ARM_SPEED_DEG_S,ARM_ACCEL_DEG_S2
from .project_io import convert_video,file_sha256
from .trial2_reach_diagnostics import target_reach_diagnostics


def hands(ax):
    return [ax.plot([],[],[],color='#9238b0',linewidth=3,marker='.',markersize=5)[0] for _ in range(2)]


def draw_hands(lines,fit,t):
    for k,line in enumerate(lines):
        set_line(line,[fit['joints'][t,k,2],fit['hand_tips'][t,k],fit['palm_contacts'][t,k]])


def status(row,anatomical=False):
    release='n/a' if pd.isna(row.preunload) else ('pass' if row.preunload else 'FAIL')
    text=f'Contact gap: {row.max_contact_gap_cm:.2f} cm\nSupport: {"pass" if row.conditional_balance else "FAIL"} | release: {release}'
    if anatomical:
        text+=f'\nROM: {"FAIL" if row.anatomy_bounds_failed else "pass"} | rate: {"FAIL" if row.rate_failed else "pass"}'
        text+=f' | acceleration: {"FAIL" if row.angular_acceleration_failed else "pass"}'
        text+=f'\nPalm proxy: {row.grip_proxy_gap_cm:.2f} cm ({"FAIL" if row.grip_proxy_failed else "pass"})'
    if getattr(row,'target_reach_conflict',False):text+='\nChosen targets exceed a necessary reach bound'
    return text


def encode_video(fig,raw,dest,frames,update,progress):
    fig.canvas.draw();width,height=fig.canvas.get_width_height()
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),30,(width,height))
    if not writer.isOpened():raise RuntimeError('Cannot open video writer')
    try:
        for t in range(frames):
            update(t);fig.canvas.draw();im=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
            writer.write(cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
            if t==min(15,frames-1):fig.savefig(dest.with_suffix('.jpg'),dpi=100)
            if t%200==0:progress(f'Rendering {dest.name}: {t}/{frames}',flush=True)
    finally:writer.release();plt.close(fig)
    convert_video(raw,dest);raw.unlink()
    cap=cv2.VideoCapture(str(dest));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if count!=frames or abs(fps-30)>1e-6:raise RuntimeError(f'Video frame count/rate mismatch: {count}, {fps}')


def comparison_video(old,new,old_m,new_m,scene,dest,title,progress):
    common=np.concatenate([old['joints'],new['joints']]);n=len(new['joints'])
    fig=plt.figure(figsize=(16,10),dpi=100,facecolor='#f6f8fb');panels=[]
    for i,label in enumerate(['Previous model','MoBL-inspired constrained arms']):
        ax=fig.add_axes([.01+.50*i,.20,.48,.65],projection='3d')
        artists=setup_axis(ax,scene,common);ax.set_title(label,fontsize=15,fontweight='bold')
        panels.append((ax,artists));ax.tick_params(labelsize=8)
    handlines=hands(panels[1][0]);texts=[fig.text(.06+.5*i,.17,'',fontsize=11,va='top',linespacing=1.5) for i in range(2)]
    heading=fig.text(.04,.95,title,fontsize=20,fontweight='bold',color='#173647')
    fig.text(.04,.91,'180 cm | assumed middle 15° → upper 0° | identical fixed cameras, axis limits and timestamps',fontsize=11)
    fig.text(.04,.025,'Diagnostic comparison · contact tolerance 0.1 cm unchanged · wrist orientation inferred · anatomy/geometry approximate',fontsize=10)
    def update(t):
        heading.set_text(f'{title} | frame {int(new["frames"][t])} | {new["frames"][t]/30:.2f} s')
        for i,(fit,m) in enumerate([(old,old_m),(new,new_m)]):
            draw_frame(panels[i][1],fit,t);texts[i].set_text(status(m.iloc[t],i==1))
        draw_hands(handlines,new,t)
    encode_video(fig,dest.with_name(dest.stem+'_encoding.mp4'),dest,n,update,progress)


def final_video(project,fit,m,scene,summary,dest,progress):
    n=len(fit['frames']);fig=plt.figure(figsize=(16,10),dpi=100,facecolor='#f6f8fb')
    ax=fig.add_axes([.01,.09,.69,.80],projection='3d');artists=setup_axis(ax,scene,fit['joints']);hl=hands(ax)
    source_ax=fig.add_axes([.73,.65,.25,.22]);source_ax.set_axis_off();source=source_ax.imshow(np.zeros((270,480,3),np.uint8))
    fig.text(.04,.955,'Trial2 | 180 cm | constrained anatomical trajectory',fontsize=22,fontweight='bold',color='#173647')
    fig.text(.04,.918,'MoBL-inspired reduced arms · assumed middle 15° → upper 0° · fixed camera',fontsize=12)
    fig.text(.735,.875,'Source video · reported climber 173 cm',fontsize=10)
    txt=fig.text(.72,.60,'',fontsize=11,va='top',linespacing=1.65)
    fig.text(.04,.03,'Purple: articulated wrist/palm proxy · red/orange: contact mismatch · yellow: COM · unmeasured wall geometry',fontsize=10)
    cap=cv2.VideoCapture(str(Path(project)/'videos/trial2.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,int(fit['frames'][0]))
    def update(t):
        ok,im=cap.read()
        if not ok:raise RuntimeError('Source video ended early')
        source.set_data(cv2.cvtColor(cv2.resize(im,(480,270)),cv2.COLOR_BGR2RGB));draw_frame(artists,fit,t);draw_hands(hl,fit,t)
        row=m.iloc[t];phase='Mixed wall panels' if row.upper_contacts and row.middle_contacts else ('Upper panel' if row.upper_contacts else 'Lower / middle panels')
        if row.ground_proposals:phase+='; ground inferred'
        txt.set_text(f'Frame {int(fit["frames"][t])} / 975 · {fit["frames"][t]/30:.2f} s\n{phase}\n\n'+status(row,True)+
            f'\nGeometry gate: {"FAIL" if row.geometric_failed else "pass"} | reserve: {"pass" if row.robust_2cm else "FAIL"}'
            f'\nWall/floor: {row.wall_penetration_cm:.2f}/{row.below_floor_cm:.2f} cm'
            f'\nMotion: {"FAIL" if row.motion_warning else "pass"} | source: {"missing" if "source_pose_missing" in str(row.issues) else "available"}'
            f'\n\nSolver: {"converged" if summary["solver"]["success"] else "NOT CONVERGED"}\nSelection: {"numerically feasible" if summary["metrics"]["numerically_feasible"] else "DIAGNOSTIC ONLY"}\n\nWrist/contact orientation: inferred\nGeometry: UNCERTAIN\nFull self-collision: unchecked\nMuscles / dynamic validity: unchecked')
        if int(fit['frames'][t])==630:
            fig.savefig(dest.parent/'frame_0630.jpg',dpi=100)
    try:encode_video(fig,dest.with_name('selected_encoding.mp4'),dest,n,update,progress)
    finally:cap.release()


def metric_plot(old,new,m,dest):
    fps=30.;fig,axes=plt.subplots(4,1,figsize=(14,11),sharex=True,layout='constrained')
    for fit,label,color in [(old,'Previous','#a96736'),(new,'Constrained','#197da2')]:
        j=fit['joints'];sw=elbow_swivel_rates(j,fps);valid=np.isfinite(sw)
        y=np.max(np.where(valid,sw,-np.inf),axis=1);y[~valid.any(axis=1)]=np.nan
        axes[0].plot(np.arange(1,len(j))/fps,y,label=label,color=color,lw=1)
        hip=j[:,2:,0].mean(axis=1);a=np.linalg.norm(np.diff(hip,n=2,axis=0),axis=-1)*fps**2
        axes[1].plot(np.arange(2,len(j))/fps,a,color=color,lw=1)
        gap=np.max(np.where(fit['known'],fit['contact_gaps_cm'],0),axis=1)
        axes[2].plot(np.arange(len(j))/fps,gap,color=color,lw=1)
    axes[0].set_ylabel('Elbow swivel (deg/s)');axes[0].legend();axes[1].set_ylabel('Hip acceleration (cm/s²)')
    axes[2].set_ylabel('Max contact gap (cm)');axes[2].axhline(.1,color='gray',ls='--',lw=1)
    flags=[m.anatomy_bounds_failed,m.rate_failed,m.angular_acceleration_failed,m.contact_failed,~m.conditional_balance,m.preunload.eq(False),m.grip_proxy_failed]
    axes[3].imshow(np.asarray(flags,dtype=int),aspect='auto',interpolation='nearest',extent=[0,(len(m)-1)/fps,6.5,-.5],cmap='Reds',vmin=0,vmax=1)
    axes[3].set_yticks(range(7),['ROM','Rate','Acceleration','Contact','Support','Release','Palm proxy']);axes[3].set_xlabel('Time (s); red = failed, white = no failure flag')
    for ax in axes[:3]:ax.grid(alpha=.2)
    fig.suptitle('Trial2 · 180 cm · same 976 frames / 30 fps · no display smoothing',fontsize=16)
    fig.savefig(dest,dpi=150);plt.close(fig)


def common_swivel_comparison(old,new):
    """Compare the same limb/time pairs, reporting observability explicitly."""
    a=elbow_swivel_rates(old['joints'],30);b=elbow_swivel_rates(new['joints'],30)
    common=np.isfinite(a)&np.isfinite(b);rows=[]
    for label,sw in [('legacy',a),('anatomical',b)]:
        values=sw[common]
        rows.append(dict(model=label,common_observable_limb_intervals=int(common.sum()),
            total_limb_intervals=int(common.size),
            swivel_max_common_deg_s=float(values.max()) if len(values) else None,
            swivel_p95_common_deg_s=float(np.quantile(values,.95)) if len(values) else None))
    return rows


def render_anatomical_trial2(project,output,progress=print):
    p=Path(project);out=Path(output);s=json.loads((out/'summary.json').read_text())
    identity=dict(numerical_summary=file_sha256(out/'summary.json'),renderer=file_sha256(Path(__file__)),
        shared_renderer=file_sha256(Path(__file__).with_name('trial2_spatial_render.py')),
        reach_diagnostic=file_sha256(Path(__file__).with_name('trial2_reach_diagnostics.py')))
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:10]
    dest=out/f'presentation_{key}';manifest=dest/'manifest.json'
    if manifest.exists():
        record=json.loads(manifest.read_text())
        for f,h in record['output_sha256'].items():
            if file_sha256(dest/f)!=h:raise ValueError('Rendered artifact changed: '+f)
        return dest
    dest.mkdir(exist_ok=True)
    old=load_fit(p/LEGACY);old_m=pd.read_csv(p/LEGACY/'metrics.csv');new,scene,m=read_case(out,s['selected_candidate'])
    old_reach=target_reach_diagnostics(old);new_reach=target_reach_diagnostics(new)
    for col in ['target_reach_conflict','target_span_excess_cm']:
        old_m[col]=old_reach[col];m[col]=new_reach[col]
    old_reach.to_csv(dest/'legacy_target_reach.csv',index=False);new_reach.to_csv(dest/'selected_target_reach.csv',index=False)
    pd.DataFrame(common_swivel_comparison(old,new)).to_csv(dest/'common_observable_swivel.csv',index=False)
    angle_rows=[]
    for side,start in [('left',6),('right',13)]:
        angle=np.rad2deg(new['anatomical_q'][:,start:start+7])
        speed=abs(np.diff(angle,axis=0))*30;acc=abs(np.diff(angle,n=2,axis=0))*900
        for k,name in enumerate(ARM_NAMES):
            angle_rows.append(dict(arm=side,coordinate=name,min_deg=float(angle[:,k].min()),max_deg=float(angle[:,k].max()),
                allowed_min_deg=ARM_LIMITS_DEG[k,0],allowed_max_deg=ARM_LIMITS_DEG[k,1],
                peak_speed_deg_s=float(speed[:,k].max()),speed_cap_deg_s=ARM_SPEED_DEG_S[k],
                peak_acceleration_deg_s2=float(acc[:,k].max()),acceleration_cap_deg_s2=ARM_ACCEL_DEG_S2[k]))
    pd.DataFrame(angle_rows).to_csv(dest/'joint_range_rate_summary.csv',index=False)
    short_common=[]
    for label,start,end in CLIPS:
        cs=json.loads((out/'clips'/f'{label}_comparison.json').read_text());name=cs['ranking'][0]['candidate']
        clip,sc,mt=read_case(out/'clips',name);ix=np.arange(start,end+1)
        old_clip=slice_arrays(old,ix,len(old['joints']))
        short_common.extend(dict(clip=label,**row) for row in common_swivel_comparison(old_clip,clip))
        comparison_video(old_clip,clip,old_m.iloc[ix],mt,sc,dest/f'{label}_comparison.mp4',f'Trial2 · {label.replace("_"," ")}',progress)
    pd.DataFrame(short_common).to_csv(dest/'clip_common_observable_swivel.csv',index=False)
    summary=json.loads((out/s['selected_candidate']/'summary.json').read_text())
    report=['# Trial2 anatomical experiment: how to read this result',
        f"Selected candidate: **{s['selected_candidate']}**. Status: **{s['selection_status']}**.",
        'The selected video is a complete independent 976-frame fit, not a concatenation of the short clips.',
        '## Full-run comparison',
        '| Check | Previous model | Selected constrained model |', '|---|---:|---:|']
    for label,col in [('Contact-failure frames','contact_failure_frames'),('Support-failure frames','support_failure_frames'),
                      ('Pre-release failure frames','preunload_failure_frames'),('Hip acceleration RMS (cm/s²)','hip_acceleration_rms_cm_s2')]:
        a,b=[x[col] for x in s['before_after']]
        report.append(f'| {label} | {a:.3f} | {b:.3f} |')
    report+=['', 'A lower motion metric does not compensate for a contact or support failure. Both are reported; contact tolerance remains 0.1 cm.',
        '## Why this candidate was selected',s['comparison'][0]['selection_reason']]
    for other in s['comparison'][1:]:report.append(f"- Compared with {other['candidate']}: {other['selection_reason']}")
    report+=['', f"Solver converged: **{summary['solver']['success']}**. {summary['solver']['message']}",
        f"ROM / coordinate speed / coordinate acceleration failure frames: **{summary['metrics']['anatomy_temporal_failure_frames']}**.",
        '## Remaining uncertainty',
        f"Necessary target-span conflicts occur in **{int(new_reach.target_reach_conflict.sum())}** frames. These concern the chosen model targets, not the filmed climber or every point in the hold regions.",
        'Wall angles, ground geometry and wrist orientation are inferred. The reduced arm chain does not include scapular rhythm, full self-collision, muscles, or dynamic validation.',
        'Elbow swivel is omitted near straight arms. Use the common-observability CSVs and their sample counts for like-for-like comparisons; undefined rotation is not zero rotation.',
        'The full constrained model can lose contact accuracy even where a short isolated fit succeeds. Short-clip success does not certify the full sequence.']
    (dest/'result_notes.md').write_text('\n'.join(report)+'\n')
    final_video(p,new,m,scene,summary,dest/'selected_plan_3d.mp4',progress)
    comparison_video(old,new,old_m,m,scene,dest/'full_comparison_3d.mp4','Trial2 · full comparison',progress)
    metric_plot(old,new,m,dest/'motion_and_failures.png')
    write_json(manifest,dict(provenance=identity,frames=976,fps=30,output_sha256={f.name:file_sha256(f) for f in dest.iterdir() if f.is_file()}))
    return dest
