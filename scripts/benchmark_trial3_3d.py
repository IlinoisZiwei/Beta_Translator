#!/usr/bin/env python3
"""Measure short spatial solves and optionally continue one complete trajectory."""
from pathlib import Path
import argparse,json,sys,resource
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.trial3_3d_pipeline import load_inputs,run_case,case_summary,write_json,identity,failure_review
from beta_translator.project_io import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--lengths',nargs='*',type=int,default=[12,36,72]);p.add_argument('--max-iterations',type=int,default=1200)
    p.add_argument('--refine-from');a=p.parse_args();root=Path(__file__).resolve().parents[1]
    out=Path(a.output)
    if not out.is_absolute():out=root/out
    if out.exists():raise FileExistsError('Use a new benchmark output; prior results are preserved.')
    out.mkdir(parents=True);data=load_inputs(root);rows=[]
    write_json(out/'camera.json',data['camera']);failure_review(root,data,out,180)
    config=dict(lengths=a.lengths,max_iterations=a.max_iterations,refine_from=a.refine_from,
                height_cm=180,tilt_deg=15,depth_prior_cm=25,mode='geometry')
    provenance=identity(root,config)
    for length in a.lengths:
        if not 1<=length<=207:raise ValueError('Benchmark must stay within native Trial3 frames.')
        info,result,table=run_case(data,np.arange(296,296+length),15,25,180,out,f'frames_{length}',
             max_iterations=a.max_iterations,force_weight=0,support_weight=0)
        row=case_summary(info,table,'benchmark');row['seconds_per_frame']=info['solver']['seconds']/length
        row['process_peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
        rows.append(row)
    if a.refine_from:
        source=Path(a.refine_from)
        if not source.is_absolute():source=root/source
        old=json.loads((source/'summary.json').read_text())
        if (old['height_cm'],old['tilt_deg'],old['depth_prior_cm'],old['force_weight'],old['support_weight'])!=(180.,15.,25.,0.,0.):
            raise ValueError('Continuation requires the same 180 cm / 15 degree / 25 cm geometry configuration.')
        for name,sha in old['code_sha256'].items():
            if file_sha256(root/'beta_translator'/name)!=sha:raise ValueError('Continuation code changed: '+name)
        with np.load(source/'trajectory.npz',allow_pickle=False) as z:frames=z['frames'];initial=z['values']
        info,result,table=run_case(data,frames,15,25,180,out,'continued_trajectory',max_iterations=a.max_iterations,
                                  force_weight=0,support_weight=0,initial=initial)
        rows.append(case_summary(info,table,'continuation'))
        from beta_translator.scene3d_render import render_comparison,make_viewer
        render_comparison(root,data,[out/'continued_trajectory'],out/'continued_trajectory.mp4')
        make_viewer(out,data,[out/'continued_trajectory'])
        config['continuation_note']='Whole original clip jointly reoptimized, not assembled from per-frame alternatives.'
    pd.DataFrame(rows).to_csv(out/'benchmark.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'scenario_metrics.csv',index=False)
    if identity(root,provenance['config'])!=provenance:raise RuntimeError('Inputs changed during benchmark.')
    write_json(out/'summary.json',dict(provenance=provenance,rows=rows,
        memory_note='Peak RSS is cumulative for this process, not isolated per-window memory.',
        output_sha256={str(f.relative_to(out)):file_sha256(f) for f in sorted(out.rglob('*')) if f.is_file()}))
    print('Benchmark:',out)


if __name__=='__main__':main()
