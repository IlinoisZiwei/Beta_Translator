"""Versioned presentation-only exports, independent of expensive Trial2 fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .project_io import file_sha256,convert_video
from .trial2_spatial_complete import run_complete_trial2
from .trial2_spatial_pipeline import write_json
from .trial2_spatial_render import read_case,setup_axis,draw_frame


def present_trial2(project,result):
    p=Path(project).resolve();source=Path(result['output_dir'])
    identity={'input_sha256':{str((source/'summary.json').relative_to(p)):file_sha256(source/'summary.json'),
        'beta_translator/trial2_spatial_presentation.py':file_sha256(Path(__file__))},
        'purpose':'Presentation only. Numerical trajectories, ranking and checks are unchanged.'}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    out=source/f'presentation_{key}'
    if (out/'summary.json').exists():
        meta=json.loads((out/'summary.json').read_text())
        if meta['provenance']!=identity:raise ValueError('Presentation provenance mismatch')
        for name,digest in meta['output_sha256'].items():
            if not (out/name).is_file() or file_sha256(out/name)!=digest:raise ValueError('Changed presentation artifact: '+name)
        return {**result,'presentation_dir':str(out)}
    if out.exists():raise FileExistsError('Incomplete presentation preserved: '+str(out))
    out.mkdir()
    names=sorted([r['candidate'] for r in result['comparison'] if r['candidate'].startswith('region_m')],
        key=lambda name:tuple(json.loads((source/name/'scene.json').read_text())['angles_deg']))
    cases=[(name,*read_case(source,name)) for name in names]
    joints=np.concatenate([fit['joints'] for _,fit,_,_ in cases]);n=len(cases[0][1]['frames'])
    fig=plt.figure(figsize=(16,10),dpi=100,facecolor='#f6f8fb');panels=[]
    for i,(name,fit,scene,metrics) in enumerate(cases):
        ax=fig.add_subplot(2,2,i+1,projection='3d');artists=setup_axis(ax,scene,joints)
        # Shared axis meanings are printed once below: four sets of rotated
        # axis titles otherwise collide with the next row or the image edge.
        ax.set(xlabel='',ylabel='',zlabel='');ax.tick_params(labelsize=7,pad=0)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(4))
        solved=json.loads((source/name/'summary.json').read_text())['solver']['success']
        panels.append((fit,scene,metrics,ax,artists,solved))
    fig.subplots_adjust(top=.88,bottom=.10,left=.03,right=.97,wspace=.08,hspace=.40)
    title=fig.suptitle('',fontsize=18,y=.97)
    height=result['provenance']['config']['height_cm']
    fig.text(.5,.935,f'Same {height:g} cm body and contact schedule · each panel is a complete 3D trajectory',ha='center',fontsize=11)
    fig.text(.5,.047,'Axes (metres): X = wall width · Z = depth · Y = height. Axis limits are shared across all four boxes.',ha='center',fontsize=10)
    fig.text(.5,.021,'Angles, camera, contact offsets and ground remain unmeasured hypotheses. Orange/red = contact miss; yellow = COM.',ha='center',fontsize=10)
    raw=out/'angle_encoding.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),30,(1600,1000))
    if not writer.isOpened():raise RuntimeError('Cannot open comparison video encoder')
    for t in range(n):
        title.set_text(f'Trial2 | frame {t}/{n-1} | wall-angle hypotheses')
        for fit,scene,metrics,ax,artists,solved in panels:
            draw_frame(artists,fit,t);m,u=scene['angles_deg'];row=metrics.iloc[t]
            ax.set_title(f'Middle {m:g}° → upper {u:g}° | gap {row.max_contact_gap_cm:.2f} cm\nBalance: {"pass" if row.conditional_balance else "FAIL"} | solver: {"converged" if solved else "NOT CONVERGED"}',fontsize=10,pad=4)
        fig.canvas.draw();frame=cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:,:,:3],cv2.COLOR_RGB2BGR)
        writer.write(frame)
        if t==600:cv2.imwrite(str(out/'angle_comparison_poster.jpg'),frame)
    writer.release();plt.close(fig);convert_video(raw,out/'angle_comparison_3d.mp4');raw.unlink()
    html=(source/'viewer.html').read_text().replace('href="selected_plan_3d.mp4"','href="../selected_plan_3d.mp4"').replace('href="candidate_comparison.csv"','href="../candidate_comparison.csv"')
    (out/'viewer.html').write_text(html)
    for name,digest in identity['input_sha256'].items():
        if file_sha256(p/name)!=digest:raise RuntimeError('Presentation input changed: '+name)
    write_json(out/'summary.json',dict(provenance=identity,frame_count=n,fps=30.,
        output_sha256={f.name:file_sha256(f) for f in sorted(out.iterdir()) if f.is_file()}))
    return {**result,'presentation_dir':str(out)}


def run_trial2_spatial(project,**kwargs):
    return present_trial2(project,run_complete_trial2(project,**kwargs))
