"""Render Trial3 with a configurable body and joint quasi-static pose/force fitting."""
from pathlib import Path
import argparse
import json
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from beta_translator.gravity_sequence_pipeline import run_trial3_gravity_sequence


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height-cm',type=float,default=180.)
    parser.add_argument('--mass-kg',type=float,default=70.,help='Assumed target mass; not estimated from video.')
    parser.add_argument('--foot-friction',type=float,default=.8)
    parser.add_argument('--max-force-n',type=float,default=None,help='Per-contact force component limit; default one body weight.')
    parser.add_argument('--com-perturbation-cm',type=float,default=2.,help='Horizontal COM sensitivity assumption.')
    parser.add_argument('--unload-lookahead-frames',type=int,default=6,help='Fixed-pose checks before predicted departures.')
    parser.add_argument('--margin-warning',type=float,default=.1,help='Force-component reserve warning threshold.')
    args=parser.parse_args()
    result=run_trial3_gravity_sequence(PROJECT,height_cm=args.height_cm,mass_kg=args.mass_kg,
        foot_friction=args.foot_friction,max_force_n=args.max_force_n,
        audit_config=dict(com_perturbation_cm=args.com_perturbation_cm,
            unload_lookahead_frames=args.unload_lookahead_frames,margin_warning=args.margin_warning),
        progress=lambda s:print(s,flush=True))
    print(json.dumps({k:result[k] for k in ('video_file','frame_count','optimizer','flag_counts',
        'balanced_frame_count','numerical_pass_frame_count','max_contact_error_cm',
        'max_force_residual_n','max_moment_residual_nm','max_segment_length_error_cm',
        'balance_audit_counts','preunload_frame_counts','last_pre_release_check_counts','release_event_count')},indent=2))
