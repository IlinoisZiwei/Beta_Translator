"""Smooth full-DOF native-frame refinement of one complete region candidate."""
import numpy as np
from .region_ik3d import RegionProblem


class NativeRegionProblem(RegionProblem):
    def __init__(self,data,regions,initial,height_cm=180.,tilt_deg=15.,depth_cm=25.,ratios=None):
        super().__init__(data,regions,'region_offset',height_cm,tilt_deg,depth_cm,stride=1,ratios=ratios)
        self.analytic[:]=False
        q=np.asarray(initial['values'])[:,:22].copy();j=np.asarray(initial['joints'])
        vectors=np.diff(j,axis=2)
        angle=np.arctan2(vectors[...,0],-vectors[...,1])
        elevation=np.arctan2(vectors[...,2],np.linalg.norm(vectors[...,:2],axis=-1))
        for limb in range(4):
            for bone in range(2):
                q[:,6+4*limb+2*bone]=np.unwrap(angle[:,limb,bone])
                q[:,7+4*limb+2*bone]=elevation[:,limb,bone]
        self.initial=np.r_[q.ravel(),np.asarray(initial['parameters'])[-self.e*2:]]
        # Angles now always mean physical segment directions, with no switched
        # swivel/free coordinates or reach-circle clipping. This weak prior is
        # explicitly anchored to the complete preceding candidate.
        self.base.seed=q.copy()
        # Permit exact vertical segment directions as well as tilted segments.
        lo=self.lower[:self.k*22].reshape(self.k,22);hi=self.upper[:self.k*22].reshape(self.k,22)
        lo[:,7:22:2]=-np.pi/2;hi[:,7:22:2]=np.pi/2
