"""Show physical contact patches in a shared 3D box, with unchanged timestamps."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .project_io import file_sha256
from .trial2_spatial_pipeline import load_fit,write_json
from .trial2_spatial_render import setup_axis,draw_frame,set_line
from .trial2_anatomy_render import encode_video,common_swivel_comparison
from .trial2_reach_diagnostics import target_reach_diagnostics


def render_stage1(output,progress=print):
    out=Path(output);summary=json.loads((out/'summary.json').read_text())
    identity={'summary':file_sha256(out/'summary.json'),'renderer':file_sha256(Path(__file__)),
        'shared_renderer':file_sha256(Path(__file__).with_name('trial2_spatial_render.py')),
        'encoder':file_sha256(Path(__file__).with_name('trial2_anatomy_render.py'))}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:10]
    dest=out/f'presentation_{key}'
    if (dest/'manifest.json').exists():
        saved=json.loads((dest/'manifest.json').read_text())
        for f,h in saved['output_sha256'].items():
            if file_sha256(dest/f)!=h:raise ValueError('Changed presentation: '+f)
        return dest
    dest.mkdir(exist_ok=True)
    selected=summary['selected_candidate'];cases=[]
    for name in ['legacy_control',selected]:
        z=load_fit(out/name);z['contact_gaps_cm']=z['common_patch_gaps_cm']
        m=pd.read_csv(out/name/'common_contact_metrics.csv')
        cases.append((z,m))
    scene=json.loads((out/selected/'scene.json').read_text())
    common=np.concatenate([z['joints'] for z,_ in cases]);fig=plt.figure(figsize=(16,10),dpi=100,facecolor='#f6f8fb')
    panels=[];labels=['Control: separate joint and palm targets','Stage 1: one articulated contact definition']
    for i,label in enumerate(labels):
        ax=fig.add_axes([.01+.5*i,.28,.48,.56],projection='3d')
        artists=setup_axis(ax,scene,common);ax.set_title(label,fontsize=13,fontweight='bold')
        # The short clip is high on the wall: crop empty floor space while
        # preserving absolute coordinates and identical limits in both views.
        y0=max(0.,float(common[...,1].min()-30)/100);y1=float(common[...,1].max()+45)/100
        ax.set_zlim(y0,y1)
        ax.set_box_aspect([np.ptp(ax.get_xlim()),np.ptp(ax.get_ylim()),y1-y0])
        # Purple = actual joint-to-patch connector; red = patch-to-surface error.
        patch=[ax.plot([],[],[],color='#8138ac',linewidth=3,marker='.',markersize=5)[0] for _ in range(4)]
        shoes=[ax.plot([],[],[],color='#119b9b',linewidth=2)[0] for _ in range(2)]
        panels.append((artists,patch,shoes))
    texts=[fig.text(.045+.5*i,.17,'',fontsize=11,va='top',linespacing=1.5) for i in range(2)]
    title=fig.text(.04,.96,'',fontsize=20,fontweight='bold',color='#173647')
    fig.text(.04,.91,'180 cm | assumed wall: middle 15° → upper 0° | same initialization, frames and iteration budget',fontsize=11)
    fig.text(.04,.045,'Purple: joint → patch. Red: patch mismatch. Green: target. Teal: assumed foot frame. No pose smoothing.',fontsize=11)
    fig.text(.04,.02,'Diagnostic only. Hand/foot orientation and wall geometry inferred; ankle ROM, full collision and dynamics unchecked.',fontsize=10)
    def update(t):
        frame=int(cases[0][0]['frames'][t]);title.set_text(f'Trial2 Stage 1 | frame {frame} | {frame/30:.2f} s')
        for i,(fit,m) in enumerate(cases):
            artists,patch,shoes=panels[i];draw_frame(artists,fit,t)
            for limb in range(4):
                if fit['known'][t,limb]:
                    point=fit['common_patch_points'][t,limb]
                    set_line(patch[limb],[fit['joints'][t,limb,2],point])
                    set_line(artists['connectors'][limb],[point,fit['contacts'][t,limb]])
                else:set_line(patch[limb],np.empty((0,3)))
            for side,line in enumerate(shoes):
                if fit['known'][t,side+2]:
                    point=fit['common_patch_points'][t,side+2]
                    frame=cases[1][0]['foot_frames'][t,side]
                    toe,side_axis=frame[:,0],frame[:,1]
                    heel=point-.07*180*toe;half=.012*180*side_axis
                    set_line(line,[point+half,point-half,heel-half,heel+half,point+half])
                else:set_line(line,np.empty((0,3)))
            row=m.iloc[t];release='n/a' if pd.isna(row.preunload) else ('pass' if row.preunload else 'FAIL')
            texts[i].set_text(f'Actual patch gap: {row.max_contact_gap_cm:.3f} cm | limit 0.1 cm\n'
                f'Support on reached patches: {"pass" if row.conditional_balance else "FAIL"} | release: {release}\n'
                'Foot frame: inferred | ankle ROM: unchecked')
    encode_video(fig,dest/'encoding.mp4',dest/'contact_comparison_3d.mp4',summary['frame_count'],update,progress)
    pd.DataFrame(common_swivel_comparison(cases[0][0],cases[1][0])).to_csv(dest/'common_swivel.csv',index=False)
    reach=target_reach_diagnostics(cases[1][0])
    reach.to_csv(dest/'selected_target_reach.csv',index=False)
    contact_rows=[]
    for name,(z,m) in zip(['legacy_control',selected],cases):
        for t,frame in enumerate(z['frames']):
            for limb,label in enumerate(['left_hand','right_hand','left_foot','right_foot']):
                if z['known'][t,limb]:
                    contact_rows.append(dict(candidate=name,frame=int(frame),limb=label,
                        hold=str(z['labels'][t,limb]),patch_gap_cm=float(z['common_patch_gaps_cm'][t,limb]),
                        reached=bool(z['common_patch_gaps_cm'][t,limb]<=.1),
                        orientation_status='inferred',scene_status='assumed',
                        ground_proposed=bool(z['ground'][t,limb])))
    pd.DataFrame(contact_rows).to_csv(dest/'contact_review.csv',index=False)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
    for (z,m),label in zip(cases,['Control','Stage 1']):
        t=z['frames']/30;gaps=z['common_patch_gaps_cm'];known=z['known']
        axes[0].plot(t,np.max(np.where(known[:,:2],gaps[:,:2],0),axis=1),label=label)
        axes[1].plot(t,np.max(np.where(known[:,2:],gaps[:,2:],0),axis=1),label=label)
        axes[2].step(t,(~m.conditional_balance).astype(int),where='mid',label=label)
    for ax in axes[:2]:ax.axhline(.1,color='gray',linestyle='--');ax.grid(alpha=.2)
    axes[0].set_ylabel('Hand patch gap (cm)');axes[0].legend()
    axes[1].set_ylabel('Foot patch gap (cm)');axes[2].set_ylabel('Support failure');axes[2].set_xlabel('Original video time (s)')
    fig.suptitle('Same physical-patch audit for both models · 180 cm · no hidden failed frames')
    fig.savefig(dest/'contact_metrics.png',dpi=140);plt.close(fig)
    notes=['# Trial2 Stage 1 comparison','',f"Frames {summary['first_frame']}–{summary['last_frame']}, 180 cm, 30 fps.",
        f"Selected: **{selected}**. Status: **{summary['selection_status']}**.",
        summary['ranking'][0]['selection_reason'],'',
        'The control and Stage 1 candidates use the same initial body coordinates, contact labels, wall hypothesis and iteration budget. '
        'The common audit checks actual hand/foot patch gaps at 0.1 cm in BOTH models. The old joint-target score is retained separately.',
        '', 'The source wrist rotation coupling is restored. Shoulder-girdle translations and bone offsets remain omitted. '
        'Foot frames are surface/image hypotheses, not validated ankle anatomy. No contact label was deleted or changed.',
        '', 'Short-segment convergence and accuracy do not establish full-sequence feasibility. Solver failures remain visible in candidate_comparison.csv.']
    (dest/'result_notes.md').write_text('\n'.join(notes)+'\n')
    write_json(dest/'manifest.json',dict(provenance=identity,frames=summary['frame_count'],fps=30,
        output_sha256={f.name:file_sha256(f) for f in dest.iterdir() if f.is_file()}))
    return dest
