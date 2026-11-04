#!/usr/bin/env python3
"""Re-audit one verified converged geometric trajectory under different gravity tilts."""
from pathlib import Path
import argparse,sys,json
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.trial3_3d_pipeline import load_inputs,run_case,write_json,case_summary,identity,verified_experiment,failure_review
from beta_translator.scene3d_render import render_comparison,make_viewer
from beta_translator.project_io import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-case',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];source=root/a.source_case;out=root/a.output
    if out.exists():raise FileExistsError('Choose a new output directory.')
    verified_experiment(source.parent);old=json.loads((source/'summary.json').read_text())
    if not old['solver']['success'] or old['force_weight'] or old['support_weight']:raise ValueError('Source must be a converged geometry-only solve.')
    for name,digest in old['code_sha256'].items():
        if file_sha256(root/'beta_translator'/name)!=digest:raise ValueError('Source code has changed.')
    data=load_inputs(root)
    with np.load(source/'trajectory.npz',allow_pickle=False) as z:
        frames=z['frames'];result={k:z[v].copy() for k,v in dict(values='values',joints='joints_cm',com='center_of_mass_cm',head='head_cm',basis_helper='basis_helper').items()}
        if not np.allclose(z['targets_cm'],data['targets'][frames],equal_nan=True):raise ValueError('Source used a different contact-target representation.')
    result['solver']=old['solver'];result['candidate_forces_fitted']=False
    provenance=identity(root,dict(source_case=str(source.relative_to(root)),source_trajectory_sha256=file_sha256(source/'trajectory.npz'),
        source_summary_sha256=file_sha256(source/'summary.json'),runner_sha256=file_sha256(__file__),angles=[0,15,30]))
    out.mkdir(parents=True);write_json(out/'camera.json',data['camera']);failure_review(root,data,out,old['height_cm'])
    rows=[];cases=[]
    for angle in [0,15,30]:
        name=f'refined_tilt{angle}'
        info,_,table=run_case(data,frames,angle,old['depth_prior_cm'],old['height_cm'],out,name,
            force_weight=0,support_weight=0,reuse=result,reused_from=str(source.relative_to(root)))
        rows.append(case_summary(info,table,'shared_converged_geometry'));cases.append(out/name)
    pd.DataFrame(rows).to_csv(out/'scenario_metrics.csv',index=False)
    render_comparison(root,data,cases,out/'refined_angles.mp4');make_viewer(out,data,cases)
    if identity(root,provenance['config'])!=provenance:raise RuntimeError('Inputs changed.')
    write_json(out/'summary.json',dict(provenance=provenance,rows=rows,
        note='Same newly refined joint trajectory, rotated with its camera and independently re-audited; does not identify tilt.',
        output_sha256={str(p.relative_to(out)):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(out)


if __name__=='__main__':main()
