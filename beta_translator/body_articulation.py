"""One source-checked body/contact model for every trial."""
import numpy as np
from .body_optimizer import AnatomicalProblem, audit as old_audit
from .body_reference import stage1_kinematics
from .body_contacts import foot_frames
from .body_constraints import DEFAULT_LIMITS

class PatchProblem(AnatomicalProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.foot_frames, self.foot_offset, self.foot_modes, self.foot_evidence = foot_frames(
            self.data, self.scene, self.regions, self.height)

    def body_kinematics(self, q):
        return stage1_kinematics(q, self.lengths, self.height)

    def geometry(self, y):
        g = super().geometry(y)
        foot = g['joints'][:, 2:, 2]+self.foot_offset
        points = np.concatenate([g['palm_contacts'], foot], axis=1)
        # One residual: p_joint + R_segment*r_patch - c_surface.
        # targets are the EXACT equivalent wrist/ankle locations, not a second
        # fixed-offset objective. Fit, audit, forces and renderer share points.
        g.update(contact_points=points, contact_targets=g['contacts'],
                 targets=g['contacts']-(points-g['joints'][:, :, 2]),
                 foot_contacts=foot, foot_frames=self.foot_frames,
                 foot_heel=foot-.07*self.height*self.foot_frames[:, :, :, 0])
        return g


class BodyProblem(PatchProblem):
    """The shared model with anatomical legs and articulated shoe patches."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for item in self.foot_evidence:
            item['prior_frame_world']=item.pop('frame_world')
            item.pop('ankle_to_patch_local_cm',None)
            item['role']='uncertain orientation prior, not actual foot transform'
            item['ankle_rom_status']='checked against assumed screening bounds'
            item['actual_patch']='trajectory.npz: foot_origins + foot_frames @ foot_patch_local_cm'

    def pose_bounds(self):
        from .body_legs import bounds_and_rates
        return bounds_and_rates()

    def body_kinematics(self,q):
        from .body_legs import kinematics
        return kinematics(q,self.lengths,self.height)

    def geometry(self,y):
        # Bypass the legacy surface-fixed foot offset; retain shared contacts.
        g=AnatomicalProblem.geometry(self,y)
        n=self.n;H=self.height;ground=self.data['ground'][:,2:]
        # Calcaneus-frame proxy dimensions in (toe,transverse,normal) axes.
        from .body_legs import source
        toe=np.array([source()['joints']['mtp_'+s]['parent']['translation'][0] for s in ('l','r')])*(H/180*100)
        local=np.zeros((n,2,3));local[:,:,0]=np.where(ground,.03*H,toe);local[:,:,2]=-.008*H
        foot=g['foot_origins']+np.einsum('tsij,tsj->tsi',g['foot_frames'],local)
        points=np.concatenate([g['palm_contacts'],foot],axis=1)
        heel=g['foot_origins']-.008*H*g['foot_frames'][:,:,:,2]
        tip=heel+(toe+.035*H)[None,:,None]*g['foot_frames'][:,:,:,0]
        g.update(contact_points=points,contact_targets=g['contacts'],
            targets=g['contacts']-(points-g['joints'][:,:,2]),foot_contacts=foot,foot_heel=heel,foot_toe=tip,foot_patch_local_cm=local)
        return g

    def local(self,y,g=None):
        g=self.geometry(y) if g is None else g
        r=super().local(y,g)
        # Surface frame is an uncertain soft prior, never an observed foot pose.
        prior=2*(g['foot_frames']-self.foot_frames)*self.known[:,2:,None,None]
        # Both heel/toe visibility and image positions remain independent evidence.
        pts=self.data['points'];scores=self.data['scores'];observed=[]
        for side,ids in enumerate(([19,17,18],[22,20,21])):
            target=np.stack([pts[:,ids[0]],pts[:,ids[1:]].mean(1)],axis=1)
            weight=np.stack([scores[:,ids[0]],scores[:,ids[1:]].min(1)],axis=1)
            valid=np.isfinite(target).all(-1)&(weight>.5)&self.data['identity_observed'][:,None]
            prediction=self.scene.pixels(np.stack([g['foot_heel'][:,side],g['foot_toe'][:,side]],axis=1))
            observed.append(((prediction-np.nan_to_num(target))/35*np.sqrt(np.where(valid,weight,0))[...,None]).reshape(self.n,-1))
        return np.c_[r,prior.reshape(self.n,-1),*observed]


def audit_body(fit, data, scene, regions, height=180., ratios=None):
    m, gaps, forces = old_audit(fit, data, scene, regions, height, ratios)
    m['hand_patch_gap_cm'] = np.max(np.where(data['known'][:, :2], gaps[:, :2], 0), axis=1)
    m['foot_patch_gap_cm'] = np.max(np.where(data['known'][:, 2:], gaps[:, 2:], 0), axis=1)
    m['grip_proxy_gap_cm'] = m.hand_patch_gap_cm
    m['grip_proxy_failed'] = m.hand_patch_gap_cm > DEFAULT_LIMITS.contact_cm
    # A separate diagnostic retains the old target mismatch, never optimizes it.
    from .body_contacts import contact_geometry
    _, offsets, _, _ = contact_geometry(scene, regions, fit['episode_uv'], 'region_offset')
    old_gap = np.linalg.norm(fit['joints'][:, :, 2]-(fit['contacts']+offsets), axis=-1)
    m['legacy_joint_target_gap_cm'] = np.max(np.where(data['known'], old_gap, 0), axis=1)
    agreement = np.linalg.norm((fit['joints'][:, :, 2]-fit['targets'])-(fit['contact_points']-fit['contacts']), axis=-1)
    m['contact_definition_disagreement_cm'] = agreement.max(axis=1)
    m['foot_orientation_uncertain'] = True
    anatomical=fit['anatomical_q'].shape[1]==32
    m['ankle_rom_checked'] = anatomical
    if anatomical:
        from .body_legs import bounds_and_rates
        lo,hi,_,_=bounds_and_rates();q=fit['anatomical_q']
        excess=np.maximum(lo-q,0)+np.maximum(q-hi,0)
        m['lower_limb_rom_failed']=excess[:,20:].max(1)>1e-7
        m['knee_coupling_displacement_cm']=np.linalg.norm(fit['joints'][:,2:,1]-fit['femur_distal'],axis=-1).max(1)
        m['foot_orientation_prior_error_deg']=np.rad2deg(np.arccos(np.clip((np.trace(np.swapaxes(fit['foot_frames'],-1,-2)@foot_frames(data,scene,regions,height)[0],axis1=-2,axis2=-1)-1)/2,-1,1))).max(1)
    switches = np.zeros(len(m), bool)
    switches[1:] = np.any(data['known'][1:] & data['known'][:-1] & (data['labels'][1:] != data['labels'][:-1]), axis=1)
    m['direct_contact_switch_uncertain'] = switches
    m['issues'] = m.issues.astype(str)+(';foot_pose_inferred;lower_limb_rom_assumed' if anatomical else ';foot_frame_inferred;ankle_rom_unchecked')
    m.loc[switches, 'issues'] += ';contact_switch_without_free_frame'
    return m, gaps, forces


