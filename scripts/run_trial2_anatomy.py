#!/usr/bin/env python3
"""Compare MoBL-inspired bounded arms on abnormal clips, then all Trial2 frames."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.trial2_anatomy_pipeline import run_anatomical_trial2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-render',action='store_true',help='Fit and audit only; skip video encoding')
    parser.add_argument('--height-cm',type=float,default=180)
    parser.add_argument('--stage',choices=['clips','full','all'],default='all')
    parser.add_argument('--clip-iterations',type=int,default=150)
    parser.add_argument('--full-iterations',type=int,default=160)
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    if args.stage=='clips':
        r=run_anatomical_trial2(project,height_cm=args.height_cm,stage='clips',clip_iterations=args.clip_iterations)
    else:
        from beta_translator.trial2_anatomy_controls import run_controlled_trial2
        r=run_controlled_trial2(project,height_cm=args.height_cm,clip_iterations=args.clip_iterations,full_iterations=args.full_iterations)
    if args.stage!='clips' and not args.no_render:
        from beta_translator.trial2_anatomy_render import render_anatomical_trial2
        r['presentation_dir']=str(render_anatomical_trial2(Path(__file__).resolve().parents[1],r['output_dir']))
    print(json.dumps({'output_dir':r['output_dir'],'selected_candidate':r.get('selected_candidate'),'stage':args.stage,'presentation_dir':r.get('presentation_dir')},indent=2))


if __name__=='__main__':main()
