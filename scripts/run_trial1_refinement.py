"""Refine all 417 Trial1 poses, preserving the preceding ground-support run."""
from pathlib import Path
import argparse,json,sys
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from beta_translator.trial1_refinement_pipeline import run_trial1_refinement

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height-cm',type=float,default=180.)
    parser.add_argument('--mass-kg',type=float,default=70.)
    parser.add_argument('--foot-friction',type=float,default=.8)
    args=parser.parse_args()
    result=run_trial1_refinement(PROJECT,**vars(args),progress=lambda m:print(m,flush=True))
    print(json.dumps({k:result[k] for k in ('video_file','optimizer','comparison','balance_audit_counts')},indent=2))
