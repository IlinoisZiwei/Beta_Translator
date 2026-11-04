#!/usr/bin/env python3
"""Optional official Depth Anything V2 Small comparison; no environment changes.

Source/weights stay in the explicitly supplied assets directory, outside Git.
Relative depth is aligned to a hypothetical wall camera, never called measured.
"""
from pathlib import Path
import argparse,sys,json,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,pandas as pd,torch
from beta_translator.trial3_3d_pipeline import load_inputs,write_json
from beta_translator.scene3d import project


def run(project_root,assets,out,frames=(296,314,334,486),input_size=280):
    assets=Path(assets);out=Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new output directory; previous probes are preserved.')
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=assets/'depth_anything_v2_vits.pth'
    expected='715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=expected:raise ValueError('Unexpected official Small checkpoint hash.')
    sys.path.insert(0,str(assets/'Depth-Anything-V2'))
    from depth_anything_v2.dpt import DepthAnythingV2
    torch.set_num_threads(4)
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
    data=load_inputs(project_root);K=data['K'];inverse=np.linalg.inv(K)
    root=Path(project_root)/'outputs/trial3/state'
    grid=json.loads((root/'white_wall_grid_points.json').read_text())
    xy=np.asarray(grid['grid_column_row'])*[15,-15];xyz=np.c_[xy,np.zeros(len(xy))]
    train=~np.asarray(grid['reserved_from_final_fit']);rows=[];depths=[]
    cap=cv2.VideoCapture(str(Path(project_root)/'videos/trial3.mp4'))
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,img=cap.read()
        if not ok:raise RuntimeError('Missing requested source frame.')
        start=time.monotonic()
        with torch.inference_mode():
            tensor,(ih,iw)=model.image2tensor(img,input_size)
            predicted=model(tensor.cpu())
            prediction=torch.nn.functional.interpolate(predicted[:,None],(ih,iw),mode='bilinear',align_corners=True)[0,0].numpy()
        seconds=time.monotonic()-start;depths.append(prediction)
        R,t=data['R'][frame],data['t'][frame]
        pixels,z=project(xyz,K,R,t)
        ij=np.rint(pixels).astype(int);h,w=prediction.shape
        keep=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
        # Exclude the entire observed person box from wall scale/shift fitting.
        human=data['observed_px'][frame].reshape(-1,2);lo=human.min(0)-35;hi=human.max(0)+35
        keep&=~np.all((pixels>=lo)&(pixels<=hi),axis=1)
        ij=np.clip(ij,[0,0],[w-1,h-1]);raw=prediction[ij[:,1],ij[:,0]]
        fit=keep&train;test=keep&~train
        coeff=np.linalg.lstsq(np.c_[raw[fit],np.ones(fit.sum())],1/z[fit],rcond=None)[0]
        inv=coeff[0]*prediction+coeff[1]
        pred_z=1/np.maximum(1e-8,coeff[0]*raw+coeff[1])
        relative_error=np.abs(pred_z[test]-z[test])/z[test]
        points=np.rint(human).astype(int);points=np.clip(points,[0,0],[w-1,h-1]);inv_joint=inv[points[:,1],points[:,0]]
        rays=np.c_[points,np.ones(len(points))]@inverse.T
        joints=((rays/np.maximum(inv_joint[:,None],1e-8))-t)@R
        row=dict(frame=int(frame),seconds=seconds,input_size=input_size,wall_fit_points=int(fit.sum()),
             wall_reserved_points=int(test.sum()),inverse_depth_scale=float(coeff[0]),inverse_depth_shift=float(coeff[1]),
             reserved_wall_relative_depth_error=float(relative_error.mean()),
             source_keypoint_surface_depth_median_cm=float(np.median(joints[:,2])),
             source_keypoint_surface_depth_min_cm=float(joints[:,2].min()),
             source_keypoint_surface_depth_max_cm=float(joints[:,2].max()),
             negative_inverse_depth_fraction=float((inv<=0).mean()),metric_depth_measured=False,
             used_in_ik=False,reason='Surface-depth probe only; joint centers, scale/shift and wall camera are unverified.')
        rows.append(row);print(row,flush=True)
        loq,hiq=np.percentile(prediction,[2,98]);color=cv2.applyColorMap(np.uint8(np.clip((prediction-loq)/(hiq-loq),0,1)*255),cv2.COLORMAP_TURBO)
        board=np.hstack([cv2.resize(img,(360,640)),cv2.resize(color,(360,640))])
        canvas=np.full((710,720,3),248,np.uint8);canvas[70:]=board
        cv2.putText(canvas,f'Frame {frame} | Original / DAV2 Small relative depth',(15,25),0,.55,(30,30,30),1,cv2.LINE_AA)
        cv2.putText(canvas,'NOT measured metric depth; not used as IK ground truth',(15,52),0,.52,(30,80,170),1,cv2.LINE_AA)
        cv2.imwrite(str(out/f'frame_{frame:04d}.jpg'),canvas)
        # Export a genuine XYZ point cloud, conditional on fitted inverse-depth affine map.
        if frame==334:
            yy,xx=np.mgrid[0:h:6,0:w:6];uv=np.c_[xx.ravel(),yy.ravel()];dd=inv[yy,xx].ravel()
            valid=dd>1e-5;rays=np.c_[uv[valid],np.ones(valid.sum())]@inverse.T
            points=((rays/dd[valid,None])-t)@R;rgb=img[uv[valid,1],uv[valid,0],::-1]
            good=np.isfinite(points).all(1)&(np.linalg.norm(points,axis=1)<1500);points=points[good];rgb=rgb[good]
            with (out/'conditional_neural_surface.ply').open('w') as f:
                f.write('ply\nformat ascii 1.0\ncomment Unverified DAV2 surface aligned to assumed wall camera; units cm\n')
                f.write(f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
                for p,c in zip(points,rgb):f.write(' '.join(map(str,[*p,*map(int,c)]))+'\n')
    cap.release();pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    np.savez_compressed(out/'relative_depth.npz',frames=frames,relative_depth=np.stack(depths))
    write_json(out/'summary.json',dict(model='Depth Anything V2 Small',license='Apache-2.0 (Small)',
        source_revision='a561b849ebae10a6f5ef49e26c83cbbcd36c71bf',weights_sha256=expected,device='cpu',
        frames=list(frames),input_size=input_size,rows=rows,used_in_ik=False,
        links=['https://github.com/DepthAnything/Depth-Anything-V2','https://huggingface.co/depth-anything/Depth-Anything-V2-Small'],
        caution='Wall consistency and visually plausible surfaces are not independent 3D accuracy.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets-dir',default=str(Path.home()/'.cache/beta-translator/depth-anything-v2-small'));p.add_argument('--output',required=True)
    p.add_argument('--download',action='store_true')
    p.add_argument('--input-size',type=int,default=280);a=p.parse_args()
    if a.download:
        from download_3d_probe_assets import prepare
        prepare(a.assets_dir)
    run(Path(__file__).resolve().parents[1],a.assets_dir,a.output,input_size=a.input_size)
