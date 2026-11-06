"""CPU-only, fixed-camera surface/stick comparison and registered video overlay."""
from pathlib import Path
import itertools
import cv2
import numpy as np
from .body_render import clip_polygon
from .contact_regions3d import homography
from .project_io import convert_video


def text(image, message, xy, size=.55, color=(65, 56, 45)):
    cv2.putText(image, str(message), tuple(map(int, xy)), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 1, cv2.LINE_AA)


def mesh_image(image, vertices, faces, xy, depth, colors):
    """Flat-shaded triangles, back-to-front; visualization, not a depth sensor."""
    tri = vertices[faces]; normals = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([-.45,.8,.65]);light /= np.linalg.norm(light)
    shade = .42+.58*np.maximum(normals@light, 0)
    palette = np.clip(colors*shade[:,None], 0, 255).astype(np.uint8)
    pixels = xy[faces]
    valid = np.isfinite(pixels).all((1,2)) & (abs(pixels).max((1,2))<1e6)
    h,w = image.shape[:2]
    valid &= (pixels[:,:,0].max(1)>=0)&(pixels[:,:,0].min(1)<w)&(pixels[:,:,1].max(1)>=0)&(pixels[:,:,1].min(1)<h)
    order = np.flatnonzero(valid)
    order = order[np.argsort(depth[faces[order]].mean(1), kind='stable')]
    for f in order:
        cv2.fillConvexPoly(image, np.rint(pixels[f]).astype(np.int32), tuple(int(v) for v in palette[f]), lineType=cv2.LINE_AA)


class Camera:
    def __init__(self, lo, hi, size=(640,650)):
        self.lo,self.hi,self.size=np.array(lo),np.array(hi),size
        self.view=np.array([.48,.22,.85]);self.view/=np.linalg.norm(self.view)
        self.right=np.cross([0,1,0],self.view);self.right/=np.linalg.norm(self.right)
        self.up=np.cross(self.view,self.right)
        self.corners=np.array(list(itertools.product(*zip(lo,hi))))
        raw=np.c_[self.corners@self.right,-self.corners@self.up]
        self.center=(raw.min(0)+raw.max(0))/2
        self.scale=min((np.array(size)-[100,100])/np.ptp(raw,axis=0))

    def project(self,p):
        p=np.asarray(p)
        return (np.stack([p@self.right,-p@self.up],-1)-self.center)*self.scale+np.array(self.size)/2

    def background(self,scene):
        w,h=self.size;image=np.full((h,w,3),(250,248,245),np.uint8)
        for i,poly in enumerate(scene['meshes_cm']):
            poly=clip_polygon(poly,self.lo,self.hi)
            if len(poly)<3:continue
            layer=image.copy();pts=np.int32(self.project(poly))
            cv2.fillPoly(layer,[pts],[(210,219,211),(220,216,200),(218,213,216)][i%3])
            image=cv2.addWeighted(layer,.6,image,.4,0)
            cv2.polylines(image,[pts],True,(180,187,182),1,cv2.LINE_AA)
        for i,a in enumerate(self.corners):
            for b in self.corners[i+1:]:
                if np.count_nonzero(a!=b)==1:
                    x,y=np.int32(self.project([a,b]));cv2.line(image,x,y,(196,191,185),1,cv2.LINE_AA)
        for axis,label in enumerate(('X: wall width','Y: height','Z: depth')):
            a=self.lo.copy();b=a.copy();b[axis]=self.hi[axis]
            for fraction in np.linspace(0,1,4):
                p=a+fraction*(b-a);x,y=self.project(p)
                cv2.circle(image,(int(x),int(y)),2,(135,129,120),-1)
                text(image,f'{p[axis]/100:.1f}',(x+4,y+13),.32)
            x,y=self.project((a+b)/2);text(image,label+' (m)',(x+7,y-7),.35)
        return image


def palette(avatar):
    # A matte mannequin, with hands/feet distinct; material is not evidence.
    colors=np.array([[158,112,41],[196,183,166]]+
                    [[158,112,41],[158,112,41],[196,183,166]]*2+
                    [[94,78,63],[94,78,63],[135,126,111]]*2,float)
    return (avatar.weights@colors)[avatar.faces].mean(1)


def draw_stick(image,fit,t,project):
    j=fit['joints'][t]
    paths=[*list(j), j[:2,0], j[2:,0], np.array([j[2:,0].mean(0),j[:2,0].mean(0),fit['head'][t]])]
    for path in paths:
        pts=np.int32(project(path));cv2.polylines(image,[pts],False,(151,112,37),3,cv2.LINE_AA)
        for p in pts:cv2.circle(image,tuple(p),4,(112,79,26),-1,cv2.LINE_AA)
    for s in range(2):
        cv2.polylines(image,[np.int32(project([fit['foot_heel'][t,s],fit['foot_toe'][t,s]]))],False,(133,125,70),4,cv2.LINE_AA)


def draw_contacts(image,fit,t,surface,project):
    for limb in range(4):
        if not fit['known'][t,limb]:continue
        a,b=np.int32(project([surface[limb],fit['contacts'][t,limb]]))
        failed=np.linalg.norm(surface[limb]-fit['contacts'][t,limb])>.1
        color=(56,64,200) if failed else (94,136,25)
        cv2.line(image,a,b,color,2,cv2.LINE_AA)
        cv2.circle(image,tuple(a),4,color,-1,cv2.LINE_AA)
        cv2.drawMarker(image,tuple(b),color,cv2.MARKER_CROSS,10,1,cv2.LINE_AA)


def _finish(writer,raw,path,frames,fps):
    writer.release();convert_video(raw,path);raw.unlink()
    cap=cv2.VideoCapture(str(path));actual=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));rate=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if actual!=frames or abs(rate-fps)>1e-5:raise RuntimeError('Rendered time axis differs from input')


def render(out, avatar, fit, metrics, scene, scene_report, config, project, fps, progress=print):
    out=Path(out);n=len(fit['frames']);p=Path(project)
    # Include the actual surface extent, with one fixed camera for the whole clip.
    lo=np.full(3,np.inf);hi=-lo
    for start in range(0,n,32):
        V=avatar.pose({k:v[start:start+32] for k,v in fit.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==n})
        lo=np.minimum(lo,V.min((0,1)));hi=np.maximum(hi,V.max((0,1)))
    lo-=18;hi+=18;cam=Camera(lo,hi);back=cam.background(scene_report);colors=palette(avatar)
    raw=out/'comparison_encoding.mp4';path=out/'avatar_comparison_3d.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,800))
    if not writer.isOpened():raise RuntimeError('Cannot open comparison video writer')
    video=cv2.VideoCapture(str(p/config['video']))
    if not video.isOpened():raise ValueError('Source video unavailable')
    video.set(cv2.CAP_PROP_POS_FRAMES,int(fit['frames'][0]))
    sw,sh=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    factor=min(1.,760/sh);ow,oh=int(sw*factor)//2*2,int(sh*factor)//2*2
    overlay_raw=out/'overlay_encoding.mp4';overlay_path=out/'avatar_source_overlay.mp4'
    overlay_writer=cv2.VideoWriter(str(overlay_raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(max(ow,640),oh+120))
    if not overlay_writer.isOpened():raise RuntimeError('Cannot open overlay video writer')
    with np.load(p/config['input_root']/'state/alignment_cache.npz',allow_pickle=False) as z:alignment=z['matrices']
    # Reference-image geometry and source-video pixels use the same reviewed size.
    def projected(V,frame):
        return homography(scene.pixels(V),alignment[frame])*np.array([ow/sw,oh/sh])
    try:
        for start in range(0,n,16):
            batch={k:v[start:start+16] for k,v in fit.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==n}
            vertices=avatar.pose(batch);points=avatar.contact_points(vertices,batch['ground'])
            for local,V in enumerate(vertices):
                t=start+local;frame=int(fit['frames'][t]);row=metrics.iloc[t]
                left,right=back.copy(),back.copy();draw_stick(left,fit,t,cam.project)
                mesh_image(right,V,avatar.faces,cam.project(V),V@cam.view,colors)
                draw_contacts(left,fit,t,fit['contact_points'][t],cam.project)
                draw_contacts(right,fit,t,points[local],cam.project)
                canvas=np.full((800,1280,3),(250,248,245),np.uint8)
                canvas[75:725,:640]=left;canvas[75:725,640:]=right
                text(canvas,f"{config['trial'].title()} | {avatar.profile['height_cm']:g} cm | frame {frame} | {frame/fps:.2f}s",(24,30),.75)
                text(canvas,'Existing anatomical skeleton',(24,61),.6)
                text(canvas,'Anny surface - same motion and contacts',(664,61),.6)
                state='FAIL' if row.source_failure else 'no recorded failure'
                text(canvas,f'Source constraints: {state} | surface/solver patch drift: {row.mesh_patch_drift_cm:.4f} cm',(24,748),.55,(55,65,185) if row.source_failure else (80,90,55))
                text(canvas,f'Hold gap: {row.mesh_hold_gap_cm:.2f} cm | wall penetration: {row.mesh_wall_penetration_cm:.2f} cm | red = contact gap > 0.1 cm',(24,772),.48)
                convergence='converged' if row.source_solver_converged else 'NOT converged'
                text(canvas,f'Source solver: {convergence}. Wall/camera/foot pose inferred; fixed fingers, grasp unvalidated.',(24,793),.43)
                writer.write(canvas)
                ok,source=video.read()
                if not ok:raise ValueError(f'Cannot read source video frame {frame}')
                source=cv2.resize(source,(ow,oh));layer=source.copy()
                xy=projected(V,frame)
                if hasattr(scene,'R') and hasattr(scene,'K'):depth=-(V@scene.R.T+scene.t)[:,2]
                else:depth=V[:,2]
                mesh_image(layer,V,avatar.faces,xy,depth,colors)
                blend=cv2.addWeighted(layer,.78,source,.22,0)
                draw_contacts(blend,fit,t,points[local],lambda a:projected(a,frame))
                overlay=np.full((oh+120,max(ow,640),3),(250,248,245),np.uint8)
                overlay[70:70+oh,:ow]=blend
                if overlay.shape[1]-ow>180:
                    x=ow+12
                    for k,line in enumerate(['DISPLAY PROFILE',f"Height: {avatar.profile['height_cm']:g} cm",f"Arm span: {avatar.profile['arm_span_cm']:g} cm",'',
                                            'CONTACTS','Dot: mesh surface','Cross: target hold','Red: gap > 0.1 cm','',
                                            'ASSUMPTIONS','Fixed finger pose','Inferred foot pose','Inferred wall/camera','',
                                            'SOURCE MOTION','Same native frames','No smoothing','No hold snapping']):
                        text(overlay,line,(x,100+24*k),.39)
                text(overlay,f"{config['trial'].title()} | frame {frame} | Anny overlay",(12,26),.57)
                text(overlay,f'Source: {state} | hold gap {row.mesh_hold_gap_cm:.2f} cm',(12,52),.45,(55,65,185))
                text(overlay,'Original climber retained; wall/person occlusion unmodeled.',(12,oh+92),.43)
                text(overlay,'Registered using the existing camera + frame homography.',(12,oh+112),.43)
                overlay_writer.write(overlay)
                if t==n//2:
                    cv2.imwrite(str(out/'avatar_comparison_3d.png'),canvas)
                    cv2.imwrite(str(out/'avatar_source_overlay.png'),overlay)
                if t%100==0:progress(f"Avatar render {config['trial']}: {t}/{n}",flush=True)
    finally:
        video.release();writer.release();overlay_writer.release()
    _finish(writer,raw,path,n,fps);_finish(overlay_writer,overlay_raw,overlay_path,n,fps)

