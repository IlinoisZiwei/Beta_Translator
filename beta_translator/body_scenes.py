"""Scene adapters keep camera/wall assumptions outside the body model."""
import numpy as np
from .contact_regions3d import homography
from .scene3d import tilt_rotation, project


class ProjectedScene:
    """Trial1 image-plane metric proxy, not a recovered 3D camera or floor."""
    floor_y_cm = None

    def __init__(self, image_to_wall, image_size, depth_cm=25.):
        self.H=np.asarray(image_to_wall);self.image_size=image_size;self.depth=depth_cm
        self.normals=np.array([[0.,0.,1.]]*3+[[0.,1.,0.]])
        self.origins=np.zeros((4,3));self.R=np.eye(3);self.t=np.array([0.,0.,1000.])
        w,h=image_size;self.meshes=[self.intersect([[0,0],[w,0],[w,h],[0,h]],0)]

    def panel(self, xy):return np.zeros(np.shape(xy)[:-1],int)

    def intersect(self, xy, panel, normal_distance=0.):
        xy=homography(xy,self.H);panel=np.broadcast_to(panel,xy.shape[:-1])
        z=np.where(panel==3,self.depth,0.)+np.broadcast_to(normal_distance,xy.shape[:-1])
        return np.concatenate([xy,z[...,None]],axis=-1)

    def pixels(self, xyz):return homography(np.asarray(xyz)[...,:2],np.linalg.inv(self.H))

    def wall_clearance(self, xyz):return np.asarray(xyz)[...,2]

    def report(self):
        return dict(kind='projected_image_plane',angles_deg=[0.,0.],normals=self.normals.tolist(),
            origins_cm=self.origins.tolist(),meshes_cm=[m.tolist() for m in self.meshes],hinges_cm=[],
            physical_validation=False,floor_model='projected contact evidence only; no metric floor plane',
            camera_model='uniform scale; depth is unobserved',depth_prior_cm=self.depth)


class PlanarScene:
    """Calibrated plane/camera hypothesis, with an explicit gravity tilt."""
    floor_y_cm = None

    def __init__(self, camera, image_size, tilt_deg=15., origin_y_cm=400.):
        Q=tilt_rotation(tilt_deg);shift=np.array([0.,origin_y_cm,0.])
        self.K=np.asarray(camera['K']);self.R=np.asarray(camera['R'])@Q.T
        self.t=np.asarray(camera['t_cm'])-self.R@shift;self.C=-self.R.T@self.t
        self.normals=np.array([Q[:,2]]*3+[[0.,1.,0.]])
        self.origins=np.tile(shift,(4,1));self.angle=float(tilt_deg);self.camera=camera
        w,h=image_size
        self.meshes=[self.intersect([[0,0],[w,0],[w,h],[0,h]],0)]

    def panel(self, xy):return np.zeros(np.shape(xy)[:-1],int)

    def intersect(self, xy, panel, normal_distance=0.):
        xy=np.asarray(xy);panel=np.asarray(panel,int)
        rays=np.concatenate([xy,np.ones((*xy.shape[:-1],1),dtype=xy.dtype)],axis=-1)@np.linalg.inv(self.K).T@self.R
        normal=self.normals[panel];origin=self.origins[panel]+np.asarray(normal_distance)[...,None]*normal
        denom=(rays*normal).sum(-1)
        if np.any(abs(denom.real)<1e-5):raise ValueError('Surface intersection is indeterminate')
        return self.C+(((origin-self.C)*normal).sum(-1)/denom)[...,None]*rays

    def pixels(self, xyz):return project(xyz,self.K,self.R,self.t)[0]

    def wall_clearance(self, xyz):return ((np.asarray(xyz)-self.origins[0])*self.normals[0]).sum(-1)

    def report(self):
        return dict(kind='single_plane_hypothesis',angles_deg=[self.angle,self.angle],
            normals=self.normals.tolist(),origins_cm=self.origins.tolist(),
            meshes_cm=[m.tolist() for m in self.meshes],hinges_cm=[],physical_validation=False,
            camera=self.camera,floor_model='unmodeled',world_origin_note='Arbitrary vertical translation; not a measured floor')
