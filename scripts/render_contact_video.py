"""Render learned contacts with source poses and explicit review/coverage warnings."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.contact_features import LIMBS, LANDMARKS
from beta_translator.identity_tracking import load_identity_guard
from beta_translator.project_io import load_reviewed_holds, file_sha256, convert_video


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--outputs',type=Path,required=True)
    a=p.parse_args()
    out=a.outputs
    prediction=json.loads((out/'automatic/predictions/automatic_contacts.json').read_text())
    reference_index = prediction.get('reference_frame_number', 0)
    if prediction['video_sha256']!=file_sha256(a.video):raise ValueError('Predictions belong to another video.')
    contacts=pd.read_csv(out/'automatic/predictions/limb_contacts.csv').set_index(['frame','limb'],verify_integrity=True)
    records,_,_,_=load_reviewed_holds(out/'state/reviewed_hold_map.npz',a.video)
    centers={r['hold_id']:np.array([r['cx_px'],r['cy_px']],np.float32) for r in records}
    with np.load(out/'automatic/wholebody.npz',allow_pickle=False) as z:
        points=z['keypoints'];scores=z['scores']
    identity = load_identity_guard(out, a.video, out/'automatic/wholebody.npz')
    if prediction.get('identity_guard_sha256') != (identity['sha256'] if identity is not None else None):
        raise ValueError('Refresh contact inference after changing the identity guard.')
    display_points = points if identity is None else identity['display']
    with np.load(out/'state/alignment_cache.npz',allow_pickle=False) as z:
        transforms={int(f):(H,str(s)) for f,H,s in zip(z['frames'],z['matrices'],z['statuses'])}
    cap=cv2.VideoCapture(str(a.video));fps=cap.get(cv2.CAP_PROP_FPS)
    width,height=int(cap.get(3)),int(cap.get(4));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale=1280/width;panel_h=int(height*scale);sidebar=420;header=70
    size=(1280+sidebar,panel_h+header)
    links=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    diagnostics=[];samples=[]
    sample_frames=set(np.linspace(0,count-1,6,dtype=int).tolist())
    destination=out/'videos/automatic_contacts_h264.mp4'
    with tempfile.TemporaryDirectory(prefix='trial-contact-preview-') as tmp:
        raw=Path(tmp)/'contacts.mp4'
        writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
        if not writer.isOpened():raise RuntimeError('Cannot create preview writer.')
        try:
            for frame in range(count):
                ok,im=cap.read()
                if not ok:raise RuntimeError(f'Video ended at frame {frame}.')
                canvas=np.full((size[1],size[0],3),248,np.uint8)
                im=cv2.resize(im,(1280,panel_h))
                identity_observed = identity is None or identity['observed'][frame]
                display_inferred = identity is not None and identity['display_inferred'][frame]
                for i,j in links:
                    xy=display_points[frame,[i,j]]*scale
                    if np.isfinite(xy).all():
                        color=(0,165,245) if display_inferred else (80,180,40) if np.min(scores[frame,[i,j]])>1.5 else (150,150,150)
                        cv2.line(im,tuple(np.rint(xy[0]).astype(int)),tuple(np.rint(xy[1]).astype(int)),color,2,cv2.LINE_AA)
                entry=transforms.get(frame);valid=entry is not None and entry[1]=='ok'
                cv2.putText(canvas,f'Frame {frame} | {frame/fps:.2f}s | analysis copy: {fps:g} FPS',(16,30),0,.72,(30,30,30),2,cv2.LINE_AA)
                warning = 'INTERPOLATED pose: display only; contacts unobserved' if display_inferred else ('Target occluded; contacts unobserved' if not identity_observed else 'Automatic predictions: contact identities and timing still need review')
                cv2.putText(canvas,warning,(16,57),0,.60,(45,70,170),1,cv2.LINE_AA)
                cv2.putText(canvas,'Hand / foot candidates',(1300,100),0,.65,(35,35,35),2,cv2.LINE_AA)
                for limb,name in enumerate(LIMBS):
                    row=contacts.loc[(frame,name)];accepted=row.status=='accepted_model'
                    color=(55,150,40) if accepted else (0,140,220)
                    xy=np.mean(points[frame,LANDMARKS[limb][1:]],axis=0)
                    within=False
                    if identity_observed and valid and np.isfinite(xy).all():
                        ref=cv2.perspectiveTransform(xy.astype(np.float32)[None,None],np.linalg.inv(entry[0]))[0,0]
                        within=bool(0<=ref[0]<width and 0<=ref[1]<height)
                    if identity_observed and np.isfinite(xy).all():cv2.circle(im,tuple(np.rint(xy*scale).astype(int)),6,color,-1,cv2.LINE_AA)
                    candidate=str(row.candidate_hold)
                    if identity_observed and valid and candidate in centers:
                        target=cv2.perspectiveTransform(centers[candidate][None,None],entry[0])[0,0]*scale
                        if np.isfinite(target).all() and 0<=target[0]<1280 and 0<=target[1]<panel_h:
                            loc=tuple(np.rint(target).astype(int))
                            cv2.circle(im,loc,12,color,2,cv2.LINE_AA)
                            cv2.putText(im,candidate,(loc[0]+12,loc[1]),0,.5,(0,0,0),3,cv2.LINE_AA)
                            cv2.putText(im,candidate,(loc[0]+12,loc[1]),0,.5,color,1,cv2.LINE_AA)
                    y=145+limb*110
                    cv2.putText(canvas,name.replace('_',' '),(1300,y),0,.62,(35,35,35),2,cv2.LINE_AA)
                    contact_text = f'{candidate} | confidence {row.confidence:.2f}' if identity_observed else 'unobserved | contact unknown'
                    cv2.putText(canvas,contact_text,(1300,y+29),0,.56,color,1,cv2.LINE_AA)
                    status='identity uncertain; review required' if not identity_observed else 'alignment uncertain' if not valid else ('outside reference map' if not within else ('high-confidence draft' if accepted else 'low-confidence draft'))
                    cv2.putText(canvas,status,(1300,y+56),0,.48,(35,70,160),1,cv2.LINE_AA)
                    diagnostics.append(dict(frame=frame,limb=name,alignment_valid=valid,inside_reference_image=within,accepted_model=accepted))
                for j,line in enumerate(['Green: high model confidence','Orange: review required',f'Holds: reference frame {reference_index}','Timing and supports unconfirmed','No climbing-success prediction']):
                    cv2.putText(canvas,line,(1300,620+j*25),0,.46,(65,65,65),1,cv2.LINE_AA)
                canvas[header:header+panel_h,:1280]=im
                writer.write(canvas)
                if frame in sample_frames:samples.append(cv2.resize(canvas,(850,size[1]//2)))
                if (frame+1)%250==0:print(f'Rendered {frame+1}/{count}',flush=True)
        finally:cap.release();writer.release()
        convert_video(raw,destination)
    pd.DataFrame(diagnostics).to_csv(out/'tables/reference_coverage.csv',index=False)
    cv2.imwrite(str(out/'images/automatic_contact_storyboard.jpg'),np.vstack([np.hstack(samples[:2]),np.hstack(samples[2:4]),np.hstack(samples[4:])]))
    print(destination,flush=True)


if __name__=='__main__':main()
