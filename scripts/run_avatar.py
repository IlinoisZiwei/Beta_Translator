#!/usr/bin/env python3
"""Render Anny from saved shared-body runs; one environment, all three trials."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.avatar_pipeline import run_avatar


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',choices=['trial1','trial2','trial3','all'],default='trial1')
    p.add_argument('--source-run',type=Path,help='Explicit shared_body run; otherwise use personalized_results.json')
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output-root',type=Path)
    p.add_argument('--no-render',action='store_true')
    a=p.parse_args()
    if a.source_run and a.trial=='all':p.error('--source-run identifies one trial')
    trials=['trial1','trial2','trial3'] if a.trial=='all' else [a.trial]
    index=json.loads((a.project/'configs/personalized_results.json').read_text())
    for trial in trials:
        source=a.source_run or a.project/index[trial]
        out=a.output_root/trial if a.output_root and len(trials)>1 else a.output_root
        r=run_avatar(a.project,source,output_root=out,render=not a.no_render)
        print(json.dumps({k:r[k] for k in ['trial','output_dir','mapping_pass','max_patch_drift_cm','source_failed_frames']},indent=2))


if __name__=='__main__':main()
