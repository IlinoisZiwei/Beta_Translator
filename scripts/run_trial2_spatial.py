#!/usr/bin/env python3
"""Fit complete Trial2 3D plans on explicitly hypothetical joined wall panels."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.trial2_spatial_presentation import run_trial2_spatial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height-cm',type=float,default=180.)
    parser.add_argument('--hole-spacing-cm',type=float,default=15.)
    parser.add_argument('--max-iterations',type=int,default=350)
    parser.add_argument('--knot-stride',type=int,default=4)
    parser.add_argument('--native-iterations',type=int,default=600)
    args=parser.parse_args()
    result=run_trial2_spatial(Path(__file__).resolve().parents[1],height_cm=args.height_cm,
        spacing_cm=args.hole_spacing_cm,max_iterations=args.max_iterations,stride=args.knot_stride,
        native_iterations=args.native_iterations)
    print(json.dumps({'output_dir':result['output_dir'],'selected_candidate':result['selected_candidate'],'presentation_dir':result['presentation_dir']},indent=2))


if __name__=='__main__':main()
