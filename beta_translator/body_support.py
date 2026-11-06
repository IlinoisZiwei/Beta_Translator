"""Shared 3D support checks; numerical feasibility remains conditional."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from .inverse_kinematics import make_body
from .body_constraints import DEFAULT_LIMITS as LIMITS

def force_audit(points,com,active,normals,height=180.,minimum_hands=True):
    """Independent six-equation LP; forces act at surface contacts, not wrists."""
    active=np.asarray(active,bool)
    if not active.any():return dict(success=False,hand_load=np.nan,forces=np.zeros((4,3)))
    A=np.zeros((6,18));A[:3,:12]=np.tile(np.eye(3),(1,4))
    for i,p in enumerate((np.asarray(points)-com)/height):
        a,b,c=p;A[3:,3*i:3*i+3]=[[0,-c,b],[c,0,-a],[-b,a,0]]
    ub=[]
    for i in [2,3]:
        n=normals[i];t=np.array([1.,0.,0.]);t-=n*(t@n);t/=np.linalg.norm(t)
        row=np.zeros(18);row[3*i:3*i+3]=-n;ub.append(row)
        for tangent in [t,np.cross(n,t)]:
            for sign in [-1,1]:
                row=np.zeros(18);row[3*i:3*i+3]=sign*tangent-LIMITS.foot_friction/np.sqrt(2)*n;ub.append(row)
    for k in range(6):
        for sign in [-1,1]:
            row=np.zeros(18);row[k]=sign;row[12+k]=-1;ub.append(row)
    objective=np.r_[np.zeros(12),np.ones(6) if minimum_hands else np.zeros(6)]
    bounds=[(-LIMITS.force_component_bw,LIMITS.force_component_bw) if active[i//3] else (0.,0.) for i in range(12)]+[(0.,None)]*6
    fit=linprog(objective,A_ub=np.array(ub),b_ub=np.zeros(len(ub)),A_eq=A,b_eq=[0,1,0,0,0,0],bounds=bounds,method='highs')
    return dict(success=bool(fit.success),hand_load=float(fit.x[12:].sum()) if fit.success else np.nan,
        forces=fit.x[:12].reshape(4,3) if fit.success else np.zeros((4,3)))


def audit_plan(fit,data,scene,regions,height_cm=180.,ratios=None,robust=True):
    j=fit['joints'];known=data['known']
    # Stage1 supplies physical hand/foot patch points; legacy fits retain joint targets.
    gaps=np.linalg.norm(fit.get('contact_points',j[:,:,2])-fit.get('contact_targets',fit['targets']),axis=-1)
    lengths=make_body(height_cm,ratios);expected=np.array([[lengths['upper_arm'],lengths['forearm']]]*2+[[lengths['thigh'],lengths['shank']]]*2)
    vectors_rigid=np.diff(fit['rigid_segments'],axis=3)[...,0,:] if 'rigid_segments' in fit else np.diff(j,axis=2)
    bone=np.max(abs(np.linalg.norm(vectors_rigid,axis=-1)-expected),axis=(1,2))
    vectors=np.diff(j,axis=2);cos=np.sum(vectors[:,:,0]*vectors[:,:,1],axis=-1)/np.prod(expected,axis=1)
    bend=np.rad2deg(np.arccos(np.clip(cos,-1,1))).max(1)
    step=np.r_[0,np.max(np.linalg.norm(np.diff(j,axis=0),axis=-1),axis=(1,2))]
    acceleration=np.r_[0,0,np.max(np.linalg.norm(np.diff(j,n=2,axis=0),axis=-1),axis=(1,2))] if len(j)>1 else np.zeros(1)
    clearance=scene.wall_clearance(j).min(axis=(1,2));rows=[];forces=[]
    floor_known=getattr(scene,'floor_y_cm',0.) is not None
    below=np.maximum(0,-j[...,1].min(axis=(1,2))) if floor_known else np.zeros(len(j))
    panel=np.array([e['panel'] for e in regions['episodes']])[np.maximum(regions['mapping'],0)]
    for t,f in enumerate(data['frames']):
        matched=known[t]&(gaps[t]<=LIMITS.contact_cm);normal=scene.normals[panel[t]]
        nominal=force_audit(fit['contacts'][t],fit['com'][t],matched,normal,height_cm)
        source_available=bool(data['identity_observed'][t] and np.any(data['visibility'][t]>0))
        eligible=matched&(data['confidence'][t]>=LIMITS.trusted_confidence)&source_available&~data['ground'][t]
        trusted=force_audit(fit['contacts'][t],fit['com'][t],eligible,normal,height_cm,False)['success']
        reserve=nominal['success'] and robust and all(force_audit(fit['contacts'][t],fit['com'][t]+d,matched,normal,height_cm,False)['success'] for d in ([LIMITS.reserve_cm,0,0],[-LIMITS.reserve_cm,0,0],[0,0,LIMITS.reserve_cm],[0,0,-LIMITS.reserve_cm]))
        departing=data['unload'][t]>0
        unloaded=force_audit(fit['contacts'][t],fit['com'][t],matched&~departing,normal,height_cm,False)['success'] if departing.any() else None
        gap=float(np.max(np.where(known[t],gaps[t],0)));issues=[]
        for flag,name in [(gap>LIMITS.contact_cm,'contact_gap'),(bone[t]>LIMITS.bone_cm,'bone_length'),(bend[t]>LIMITS.leg_bend_deg+LIMITS.bend_tolerance_deg,'joint_bend'),(clearance[t]<-LIMITS.penetration_cm,'wall_penetration'),(below[t]>LIMITS.penetration_cm,'below_floor'),(step[t]>LIMITS.motion_step_cm or acceleration[t]>LIMITS.motion_second_difference_cm,'motion_warning'),(not nominal['success'],'support_failed'),(unloaded is False,'preunload_failed'),(not fit['solver']['success'],'not_converged'),(not source_available,'source_pose_missing')]:
            if flag:issues.append(name)
        rows.append(dict(frame=int(f),max_contact_gap_cm=gap,contact_failed=gap>LIMITS.contact_cm,bone_error_cm=float(bone[t]),max_bend_deg=float(bend[t]),
            wall_penetration_cm=float(max(0,-clearance[t])),below_floor_cm=float(below[t]),
            geometric_failed=bool(gap>LIMITS.contact_cm or bone[t]>LIMITS.bone_cm or bend[t]>LIMITS.leg_bend_deg+LIMITS.bend_tolerance_deg or clearance[t]<-LIMITS.penetration_cm or below[t]>LIMITS.penetration_cm),
            step_cm=float(step[t]),acceleration_cm=float(acceleration[t]),motion_warning=bool(step[t]>LIMITS.motion_step_cm or acceleration[t]>LIMITS.motion_second_difference_cm),
            conditional_balance=nominal['success'],eligible_balance=trusted,robust_2cm=bool(reserve),
            preunload=unloaded,hand_load_l1_bw=nominal['hand_load'],known_contacts=int(known[t].sum()),
            ground_proposals=int(data['ground'][t].sum()),upper_contacts=int(((panel[t]==0)&known[t]).sum()),
            middle_contacts=int(((panel[t]==1)&known[t]).sum()),issues=';'.join(issues),geometry_uncertain=True))
        forces.append(nominal['forces'])
    return pd.DataFrame(rows),gaps,np.asarray(forces)


