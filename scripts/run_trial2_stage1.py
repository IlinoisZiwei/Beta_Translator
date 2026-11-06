#!/usr/bin/env python3
"""Compare source-checked rotations and consistent hand/foot patches at 180 cm."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.trial2_stage1_pipeline import run_stage1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height-cm',type=float,default=180.)
    parser.add_argument('--clip',choices=['right_elbow','left_elbow','hinge_transition'],default='right_elbow')
    parser.add_argument('--iterations',type=int,default=250)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--no-render',action='store_true')
    args=parser.parse_args()
    result=run_stage1(Path(__file__).resolve().parents[1],height_cm=args.height_cm,
        clip=args.clip,iterations=args.iterations,output_root=args.output_root)
    if not args.no_render:
        from beta_translator.trial2_stage1_render import render_stage1
        render_stage1(result['output_dir'])
    print(json.dumps({'output_dir':result['output_dir'],'selected':result['selected_candidate'],
                      'status':result['selection_status']},indent=2))

if __name__=='__main__':main()
