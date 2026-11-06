"""Conditional, joined Trial2 panels. Geometry assumptions never become measurements."""
from pathlib import Path
import json
import cv2
import numpy as np
from .scene3d import fit_camera, tilt_rotation, project
from .contact_regions3d import homography


def line_height(points, line):
    p=np.asarray(points);a,b=np.asarray(line,float)
    return a[1]+(p[...,0]-a[0])*(b[1]-a[1])/(b[0]-a[0])


class Trial2Scene:
    def __init__(self, project, middle_deg=15., upper_deg=0., spacing_cm=15.):
        self.spec=json.loads((Path(project)/'data/trial2_spatial_hypotheses.json').read_text())
        if not (0<=middle_deg<=55 and -10<=upper_deg<=30 and middle_deg>=upper_deg):
            raise ValueError('Expected upper [-10,30], middle [0,55], middle >= upper.')
        if not np.isfinite(spacing_cm) or spacing_cm<=0:raise ValueError('Positive assumed hole spacing required.')
        self.angles=(float(middle_deg),float(upper_deg));self.spacing=float(spacing_cm)
        xy=np.asarray(self.spec['upper_grid_px'],float)
        uv=np.asarray(self.spec['upper_grid_indices'],float)*spacing_cm
        H,_=cv2.findHomography(xy,uv,0)
        hinge=homography(self.spec['upper_hinge_px'],H)
        axis=hinge[1]-hinge[0];axis/=np.linalg.norm(axis);up=np.array([-axis[1],axis[0]])
        T=np.array([[*axis,-axis@hinge.mean(0)],[*up,-up@hinge.mean(0)],[0,0,1.]])
        self.H_upper=T@H
        xyz=np.c_[homography(xy,self.H_upper),np.zeros(len(xy))]
        train=np.ones(len(xy),bool);train[-1]=False
        camera=fit_camera(xyz,xy,train,self.spec['image_size'])
        Q=tilt_rotation(upper_deg)
        self.K=np.asarray(camera['K']);self.R=np.asarray(camera['R'])@Q.T
        self.t=np.asarray(camera['t_cm']);self.C=-self.R.T@self.t
        self.normals=np.array([tilt_rotation(upper_deg)[:,2],tilt_rotation(middle_deg)[:,2],[0.,0.,1.],[0.,1.,0.]])
        self.origins=np.zeros((4,3))
        # The local grid camera is a least-squares fit: its reconstructed hinge
        # need not lie on the ideal UV x-axis. Rotate around the actual shared
        # 3D edge, so both panels meet exactly along every observed hinge ray.
        upper_edge=self.intersect(self.spec['upper_hinge_px'],0)
        edge_axis=upper_edge[1]-upper_edge[0];edge_axis/=np.linalg.norm(edge_axis)
        fold=cv2.Rodrigues(edge_axis*np.deg2rad(middle_deg-upper_deg))[0]
        self.normals[1]=fold@self.normals[0];self.origins[1]=upper_edge.mean(0)
        lower_edge=self.intersect(self.spec['lower_hinge_px'],1)
        edge=lower_edge[1]-lower_edge[0];normal=np.cross(edge,[0.,1.,0.]);normal/=np.linalg.norm(normal)
        if normal[2]<0:normal=-normal
        self.normals[2]=normal;self.origins[2]=lower_edge.mean(0)
        floor_edge=self.intersect(self.spec['floor_edge_px'],2)
        floor=float(floor_edge[:,1].mean());self.floor_edge_disagreement=float(np.ptp(floor_edge[:,1]))
        # Translate the entire calibrated scene, including camera, so the floor is Y=0.
        shift=np.array([0.,floor,0.]);self.origins-=shift;self.origins[3]=0
        self.t+=self.R@shift;self.C=-self.R.T@self.t
        self.meshes=[]
        for panel,name in enumerate(('upper','middle','lower')):
            self.meshes.append(self.intersect(self.spec['wall_polygons_px'][name],panel))
        self.hinges=[self.intersect(self.spec['upper_hinge_px'],0),self.intersect(self.spec['lower_hinge_px'],1)]
        self.camera_report={**camera,'middle_deg':middle_deg,'upper_deg':upper_deg,
            'K':self.K.tolist(),'R':self.R.tolist(),'t_cm':self.t.tolist(),
            'floor_edge_height_disagreement_cm':self.floor_edge_disagreement,
            'actual_panel_tilts_deg':np.rad2deg(np.arcsin(-self.normals[:3,1])).tolist(),
            'intrinsics_status':'Unmeasured square-pixel, centered-principal-point, zero-distortion model.',
            'grid_validation_status':'One local held-out point; not independent global calibration.',
            'tracking_status':'Existing global image homographies are approximate for moving cameras and multiple planes.'}

    def panel(self, xy):
        xy=np.asarray(xy)
        upper=homography(xy,self.H_upper)[...,1]>=0
        lower=xy[...,1]>line_height(xy,self.spec['lower_hinge_px'])
        return np.where(upper,0,np.where(lower,2,1)).astype(int)

    def intersect(self, xy, panel, normal_distance=0.):
        xy=np.asarray(xy);panel=np.asarray(panel,int)
        rays=np.concatenate([xy,np.ones((*xy.shape[:-1],1),dtype=xy.dtype)],axis=-1)@np.linalg.inv(self.K).T@self.R
        normal=self.normals[panel];origin=self.origins[panel]+np.asarray(normal_distance)[...,None]*normal
        denom=(rays*normal).sum(-1)
        if np.any(abs(denom.real)<1e-5):raise ValueError('Image ray approaches panel horizon; geometry is indeterminate.')
        distance=((origin-self.C)*normal).sum(-1)/denom
        return self.C+distance[...,None]*rays

    def pixels(self, xyz):return project(xyz,self.K,self.R,self.t)[0]

    def wall_clearance(self, xyz):
        """Signed clearance to the finite panels' Y ranges, not all infinite halfspaces."""
        p=np.asarray(xyz);signed=[];penalties=[]
        for i in range(3):
            origin=self.origins[i];normal=self.normals[i]
            d=((p-origin)*normal).sum(-1)
            surface=p-d[...,None]*normal
            image=self.pixels(surface)
            applies=self.panel(image.real)==i
            signed.append(np.where(applies,d,np.inf))
        return np.min(np.stack(signed),axis=0)

    def report(self):
        return dict(camera=self.camera_report,angles_deg=list(self.angles),normals=self.normals.tolist(),
            origins_cm=self.origins.tolist(),meshes_cm=[m.tolist() for m in self.meshes],
            hinges_cm=[h.tolist() for h in self.hinges],assumptions=self.spec,physical_validation=False)
