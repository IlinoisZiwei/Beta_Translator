"""Optimize Trial1 poses for contacts, force reserve and pre-release unloading."""
from pathlib import Path
import argparse,json,sys
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from beta_translator.trial1_support_pipeline import run_trial1_support_optimization

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--height-cm',type=float,default=180.)
    p.add_argument('--mass-kg',type=float,default=70.,help='Assumed target mass, not video-estimated.')
    p.add_argument('--foot-friction',type=float,default=.8)
    p.add_argument('--max-force-n',type=float,default=None)
    a=p.parse_args()
    result=run_trial1_support_optimization(PROJECT,height_cm=a.height_cm,mass_kg=a.mass_kg,
        foot_friction=a.foot_friction,max_force_n=a.max_force_n,progress=lambda m:print(m,flush=True))
    print(json.dumps({k:result[k] for k in ('video_file','optimizer','comparison','balance_audit_counts',
        'last_pre_release_check_counts')},indent=2))
