#!/usr/bin/env python3
"""Compare complete native-frame Trial3 contact-region/offset trajectories."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.trial3_region_pipeline import run_region_experiment
from beta_translator.trial3_native_region_pipeline import run_complete_region_plan
from beta_translator.region_ik3d import PROFILES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--height-cm',type=float,default=180.)
    p.add_argument('--max-iterations',type=int,default=800)
    p.add_argument('--knot-stride',type=int,default=4)
    p.add_argument('--profiles',nargs='+',choices=PROFILES,default=PROFILES)
    a=p.parse_args();s=run_complete_region_plan(Path(__file__).resolve().parents[1],height_cm=a.height_cm,
        max_iterations=a.max_iterations,stride=a.knot_stride,profiles=a.profiles)
    print(json.dumps({'output_dir':s['output_dir'],'selected_candidate':s['selected_candidate']},indent=2))

if __name__=='__main__':main()
