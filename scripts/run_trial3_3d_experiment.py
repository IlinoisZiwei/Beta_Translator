#!/usr/bin/env python3
"""Trial3 conditional spatial experiment, retaining every existing run."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.trial3_3d_pipeline import run_experiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--height-cm',type=float,default=180)
    p.add_argument('--angles',type=float,nargs='+',default=[0,15,30])
    p.add_argument('--depths',type=float,nargs='+',default=[10,25,40])
    p.add_argument('--keyframes',type=int,nargs='*',default=None)
    p.add_argument('--clip',action='append',metavar='NAME:START:END')
    p.add_argument('--no-clips',action='store_true')
    p.add_argument('--mode',choices=['geometry','support'],default='geometry')
    p.add_argument('--max-iterations',type=int,default=1200)
    a=p.parse_args();clips=None
    if a.clip:
        clips={}
        for entry in a.clip:
            name,start,end=entry.split(':');clips[name]=(int(start),int(end))
    if a.no_clips:clips={}
    s=run_experiment(Path(__file__).resolve().parents[1],height_cm=a.height_cm,angles=a.angles,
        depths=a.depths,keyframes=a.keyframes,clips=clips,max_iterations=a.max_iterations,solver_mode=a.mode)
    print('Verified output:',s['output_dir'])


if __name__=='__main__':main()
