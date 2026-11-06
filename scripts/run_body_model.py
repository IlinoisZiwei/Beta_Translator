#!/usr/bin/env python3
"""One command for shared-model verification or full-sequence fitting."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.body_model import BodyModel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',default='trial2',help='trial1, trial2, trial3, all, or a config JSON path')
    parser.add_argument('--height-cm',type=float,default=None,help='Optional scaled demo height; must agree with an explicit user profile')
    parser.add_argument('--user-profile',type=Path,help='User JSON; defaults to configs/users/trial_user_175.json')
    parser.add_argument('--candidates',nargs='+',default=['anatomical_control','balanced','steady'])
    parser.add_argument('--frames',choices=['verification','all'],default='verification')
    parser.add_argument('--iterations',type=int,default=150)
    parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args();model=BodyModel(height_cm=args.height_cm,user_profile=args.user_profile)
    trials=['trial1','trial2','trial3'] if args.trial=='all' else [args.trial]
    for trial in trials:
        root=args.output_root/trial if args.output_root and len(trials)>1 else args.output_root
        result=model.fit(trial,project=Path(__file__).resolve().parents[1],frames=args.frames,
            iterations=args.iterations,candidates=tuple(args.candidates),output_root=root,render=not args.no_render)
        print(json.dumps({k:result[k] for k in ['trial','output_dir','selected_candidate','selection_status','specification_status']},indent=2))


if __name__=='__main__':main()
