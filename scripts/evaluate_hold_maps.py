"""Reproduce the saved trial2 point-coverage audit, with one-to-one matches."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(root, audit=None, include_matches=False):
    audit=audit if audit is not None else json.loads((root/'route_audit.json').read_text())
    result={}
    files={'original100':'single_frame_450_original.npz','cap400':'cap400.npz',
           'fusion':'fusion.npz','fusion_tiled':'fusion_tiled.npz'}
    for stage,name in files.items():
        with np.load(root/name,allow_pickle=False) as data:
            full='metadata' in data.files
            records=json.loads(str(data['metadata'].item()))['records'] if full else json.loads(str(data['records'].item()))
            distance=np.full((len(audit['holds']),len(records)),1e6)
            for j,r in enumerate(records):
                ys,xs=np.nonzero(data[r['hold_id']])
                if not full:xs=xs+r['bbox'][0];ys=ys+r['bbox'][1]
                for i,h in enumerate(audit['holds']):
                    x,y=h['point'];distance[i,j]=np.hypot(xs-x,ys-y).min()
        row,col=linear_sum_assignment(np.where(distance<=audit['tolerance_px'],distance,1e6))
        matched=int((distance[row,col]<=audit['tolerance_px']).sum())
        result[stage]={'candidate_count':len(records),'matched_points':matched,
            'audited_points':len(audit['holds']),'priority_point_recall':matched/len(audit['holds'])}
        if include_matches:
            assignments={int(i):int(j) for i,j in zip(row,col)
                         if distance[i,j]<=audit['tolerance_px']}
            result[stage]['matches']=[{
                'audit_id':h['audit_id'], 'point':h['point'],
                'hold_id':records[assignments[i]]['hold_id'] if i in assignments else None,
                'distance_px':float(distance[i,assignments[i]]) if i in assignments else None,
            } for i,h in enumerate(audit['holds'])]

    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(evaluate(args.artifacts),indent=2))
