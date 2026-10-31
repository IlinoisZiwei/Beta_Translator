"""Build a trial hold map from observed donor frames and overlapping image tiles.

Requires an existing pose/alignment pass. Writes candidate artifacts separately;
review selections before promoting them to state/reviewed_hold_map.npz.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.hold_mapping import reconstruct_wall,detect_candidates,export_candidates
from beta_translator.project_io import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--outputs',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--reference',type=int,required=True)
    parser.add_argument('--frames',type=int,nargs='+',required=True)
    parser.add_argument('--threshold',type=float,default=.7)
    parser.add_argument('--max-detections',type=int,default=400)
    args=parser.parse_args()
    if not 0<args.threshold<1 or args.max_detections<1 or min(args.frames+[args.reference])<0:
        parser.error('Invalid detector settings or frame index.')
    repo=Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    image,reconstruction=reconstruct_wall(args.video,args.outputs/'automatic/wholebody.npz',
        args.outputs/'state/alignment_cache.npz',args.frames,args.reference,args.destination)
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    config=get_cfg();config.merge_from_file(str(repo/'models/experiment_config.yml'))
    config.MODEL.WEIGHTS=str(repo/'models/model_final.pth');config.MODEL.DEVICE='cpu'
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST=args.threshold
    config.TEST.DETECTIONS_PER_IMAGE=args.max_detections
    candidates,details=detect_candidates(image,DefaultPredictor(config),tiled=True)
    records=export_candidates(candidates,image,args.destination/'candidates')
    manifest={'video_sha256':file_sha256(args.video),'reference_frame_number':args.reference,
        'source_frames':args.frames,'model_sha256':file_sha256(repo/'models/model_final.pth'),
        'alignment_sha256':file_sha256(args.outputs/'state/alignment_cache.npz'),
        'pose_sha256':file_sha256(args.outputs/'automatic/wholebody.npz'),
        'threshold':args.threshold,'max_detections':args.max_detections,
        'retained_candidates':len(records),'details':details,'reconstruction':reconstruction,
        'candidate_sha256':file_sha256(args.destination/'candidates.npz'),
        'status':'Unreviewed candidates; no route or physical-contact confirmation'}
    (args.destination/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
