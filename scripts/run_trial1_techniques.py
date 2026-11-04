#!/usr/bin/env python3
"""Compare complete Trial1 technique candidates and render one selected plan."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.trial1_technique_pipeline import run_trial1_techniques

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--height-cm',type=float,default=180.);p.add_argument('--baseline-dir')
    p.add_argument('--profiles',nargs='+');p.add_argument('--max-iterations',type=int,default=1600);p.add_argument('--max-outer',type=int,default=4)
    a=p.parse_args();r=run_trial1_techniques(Path(__file__).resolve().parents[1],a.height_cm,a.baseline_dir,a.profiles,a.max_iterations,a.max_outer)
    print('Selected:',r['selected_profile']);print('Output:',r['output_dir'])
