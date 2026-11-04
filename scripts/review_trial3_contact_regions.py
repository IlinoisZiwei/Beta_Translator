#!/usr/bin/env python3
"""Audit point-center assumptions and run an unreviewed source-evidence ablation."""
from pathlib import Path
import argparse,json,sys,copy
import cv2,numpy as np,pandas as pd
from scipy.spatial.distance import cdist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.trial3_3d_pipeline import load_inputs,write_json,run_case,case_summary,identity
from beta_translator.scene3d import contact_span_certificate
from beta_translator.pose_adapter import project_plane
from beta_translator.inverse_kinematics import LIMBS,make_body
from beta_translator.contact_features import LANDMARKS
from beta_translator.project_io import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);a=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(a.output)
    if not out.is_absolute():out=root/out
    if out.exists():raise FileExistsError('Use a new output directory.')
    out.mkdir(parents=True);data=load_inputs(root);old=data['old'];H=np.asarray(old['geometry']['image_to_wall_cm'])
    provenance=identity(root,dict(height_cm=180,tilt_deg=15,depth_prior_cm=25,contact_model='unreviewed source fingertip/shoe proxy',runner_sha256=file_sha256(__file__)))
    regions={};boundaries={}
    for hold in old['selected']:
        mask=old['masks'][hold].astype(np.uint8);yy,xx=np.where(mask);regions[hold]=np.c_[xx,yy]
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        boundaries[hold]=np.concatenate([c[:,0] for c in contours])
    pair_rows=[];cache={};body=make_body(180)
    for f in data['frames']:
        for p in contact_span_certificate(data['targets'][f],data['known'][f],body):
            i,j=p['limb_i'],p['limb_j'];left,right=data['labels'][f,[i,j]]
            key=(str(left),str(right))
            if key not in cache:
                cache[key]=(float(cdist(project_plane(boundaries[left],H),project_plane(boundaries[right],H)).min()) if left!=right else 0.)
            minimum=cache[key]
            pair_rows.append(dict(frame=int(f),limb_a=LIMBS[i],limb_b=LIMBS[j],hold_a=left,hold_b=right,
                center_distance_cm=p['distance_cm'],maximum_span_cm=p['maximum_span_cm'],center_excess_cm=p['excess_cm'],
                minimum_region_distance_cm=minimum,region_excess_cm=max(0,minimum-p['maximum_span_cm'])))
    pd.DataFrame(pair_rows).to_csv(out/'region_span_audit.csv',index=False)
    z=np.load(root/'outputs/trial3/automatic/wholebody.npz',allow_pickle=False);points=z['keypoints'];scores=z['scores']
    align=np.load(root/'outputs/trial3/state/alignment_cache.npz',allow_pickle=False)['matrices']
    proposed=data['targets'].copy();evidence=[];episodes=[]
    for limb in range(4):
        labels=data['labels'][:,limb];starts=np.flatnonzero(np.r_[True,labels[1:]!=labels[:-1]])
        for start,stop in zip(starts,np.r_[starts[1:],len(labels)]):
            hold=str(labels[start])
            if hold not in regions:continue
            samples=[];episode=len(episodes)
            for f in range(start,stop):
                ids=np.unique(LANDMARKS[limb][1:]);valid=(scores[f,ids]>.5)&np.isfinite(points[f,ids]).all(1)
                source=points[f,ids[valid]]
                if not len(source):continue
                reference=project_plane(source,np.linalg.inv(align[f]));distance=cdist(reference,regions[hold]);near=distance.min(1)
                inside=reference[near<=1.5]
                point=np.median(inside,axis=0) if len(inside) else reference[np.argmin(near)]
                nearest=regions[hold][np.argmin(np.linalg.norm(regions[hold]-point,axis=1))]
                gap=float(near.min());accepted=gap<=12.
                if accepted:samples.append(nearest)
                evidence.append(dict(frame=f,limb=LIMBS[limb],hold=hold,episode=episode,
                    minimum_landmark_to_region_px=gap,evidence_accepted=accepted,
                    classifier_confidence=float(data['confidence'][f,limb]),
                    raw_simcc_score_is_probability=False,source='RTMW fingertips or toes/heel; not anatomical wrist/ankle'))
            accepted=len(samples)>=3
            if accepted:
                median=np.median(samples,axis=0);pixel=regions[hold][np.argmin(np.linalg.norm(regions[hold]-median,axis=1))]
                point=np.r_[project_plane(pixel[None],H)[0],0.];proposed[start:stop,limb]=point
            else:point=data['targets'][start,limb]
            episodes.append(dict(episode=episode,limb=LIMBS[limb],hold=hold,start_frame=int(start),end_frame=int(stop-1),
                accepted_frames=len(samples),point_cm=point.tolist(),used_proxy=accepted,
                center_to_proxy_cm=float(np.linalg.norm(point-data['targets'][start,limb])),
                review_status='assistant-derived unreviewed contact proxy; not ground truth',
                rule='Median of near-region source fingertips/toes, projected to the observed mask; no target-body input.'))
    write_json(out/'contact_point_proposals.json',episodes);pd.DataFrame(evidence).to_csv(out/'contact_evidence.csv',index=False)
    candidate={**data,'targets':proposed};rows=[];cases=[]
    for name,frames in [('key_314',[314]),('key_486',[486]),('late_clip',np.arange(474,498))]:
        info,result,table=run_case(candidate,frames,15,25,180,out,name,max_iterations=1200,force_weight=0,support_weight=0)
        rows.append(case_summary(info,table,'source_contact_proxy'));cases.append(out/name)
    pd.DataFrame(rows).to_csv(out/'scenario_metrics.csv',index=False)
    from beta_translator.scene3d_render import make_viewer,render_comparison
    make_viewer(out,candidate,cases);render_comparison(root,candidate,[out/'late_clip'],out/'late_clip_proxy.mp4')
    write_json(out/'camera.json',data['camera'])
    # This is a change of target representation, not an improvement under identical constraints.
    pd.DataFrame(pair_rows).to_csv(out/'failure_review.csv',index=False)
    cap=cv2.VideoCapture(str(root/'videos/trial3.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,486);ok,img=cap.read();cap.release()
    if ok:
        for limb in range(4):
            hold=str(data['labels'][486,limb]);poly=project_plane(boundaries[hold],align[486]);cv2.polylines(img,[poly.astype(int)],True,(230,180,40),2)
            center=project_plane(data['targets'][486,limb:limb+1,:2],np.linalg.inv(H));center=project_plane(center,align[486])[0]
            proxy=project_plane(proposed[486,limb:limb+1,:2],np.linalg.inv(H));proxy=project_plane(proxy,align[486])[0]
            cv2.drawMarker(img,tuple(center.astype(int)),(220,40,220),cv2.MARKER_CROSS,15,2)
            cv2.circle(img,tuple(proxy.astype(int)),7,(40,180,250),2)
            for pt in points[486,np.unique(LANDMARKS[limb][1:])]:cv2.circle(img,tuple(pt.astype(int)),3,(250,250,250),-1)
        board=np.full((1360,720,3),245,np.uint8);board[80:]=img
        cv2.putText(board,'Frame 486 | cyan: hold regions | purple: old centers',(15,27),0,.58,(30,30,30),1,cv2.LINE_AA)
        cv2.putText(board,'Amber: UNREVIEWED source contact proxy | white: RTMW',(15,57),0,.55,(20,100,170),1,cv2.LINE_AA)
        cv2.imwrite(str(out/'contact_evidence_0486.jpg'),board)
    if identity(root,provenance['config'])!=provenance:raise RuntimeError('Inputs changed during evidence ablation.')
    table=pd.DataFrame(pair_rows)
    write_json(out/'summary.json',dict(provenance=provenance,rows=rows,
        center_span_conflict_frames=int(table[table.center_excess_cm>.2].frame.nunique()),
        region_span_conflict_frames=int(table[table.region_excess_cm>.2].frame.nunique()),
        interpretation='No region pairwise obstruction is NOT proof of a simultaneous whole-body solution. Proxy comparisons change contact targets and remain unreviewed.',
        output_sha256={str(p.relative_to(out)):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print('Contact-region review:',out)


if __name__=='__main__':main()
