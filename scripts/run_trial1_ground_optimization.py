"""Run the Trial1 whole-video ground-support experiment (417 native frames)."""
from pathlib import Path
import argparse
import json
import sys
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from beta_translator.trial1_ground_pipeline import run_trial1_ground_optimization

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height-cm',type=float,default=180.)
    parser.add_argument('--mass-kg',type=float,default=70.)
    parser.add_argument('--foot-friction',type=float,default=.8)
    args=parser.parse_args()
    result=run_trial1_ground_optimization(PROJECT,**vars(args),progress=lambda x:print(x,flush=True))
    print(json.dumps({k:result[k] for k in ('video_file','optimizer','published_model_comparison')},indent=2))
