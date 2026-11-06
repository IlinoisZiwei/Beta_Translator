"""Shared personalized fitting: active mechanics plus explicit assumption audits."""
import numpy as np
from scipy.sparse import coo_matrix,vstack
from .body_articulation import BodyProblem
from .body_mechanics import segments,mechanics,fatigue_coupling


class PersonalizedProblem(BodyProblem):
    def __init__(self,*args,user_profile,advanced=True,**kwargs):
        self.user_profile=user_profile;self.dynamic_balance=advanced
        super().__init__(*args,**kwargs)
        D=np.gradient(np.eye(self.n),1/self.fps,axis=0,edge_order=2)
        self.mechanical_pattern=(abs(D)>0)|(abs(D@D)>0)|np.eye(self.n,dtype=bool)
        for color in range(5):
            if np.any(self.mechanical_pattern[:,color::5].sum(1)>1): raise ValueError('Invalid derivative coloring')

    def body_kinematics(self,q):
        g=super().body_kinematics(q);d=self.user_profile['dimensions_cm']
        for side in range(2):
            H=g['orientations'][:,3+side*3];w=g['joints'][:,side,2]
            g['hand_tips'][:,side]=w-d['hand_length']*H[:,:,1]
            g['palm_contacts'][:,side]=w-d['hand_patch_distance']*H[:,:,1]-d['hand_patch_depth']*H[:,:,2]
        s,_=segments(g,self.user_profile);b=self.user_profile['biomechanics']
        c=s[:,:,0]+np.array(b['com_fractions'])[None,:,None]*(s[:,:,1]-s[:,:,0])
        g['com']=100*np.sum(c*np.array(b['mass_fractions'])[None,:,None],axis=1)
        return g

    def geometry(self,y):
        g=super().geometry(y);d=self.user_profile['dimensions_cm'];F=g['foot_frames']
        local=np.zeros((self.n,2,3));local[:,:,0]=np.where(self.data['ground'][:,2:],.45,.65)*d['foot_length'];local[:,:,2]=-d['foot_depth']
        foot=g['foot_origins']+np.einsum('tsij,tsj->tsi',F,local)
        points=np.concatenate([g['palm_contacts'],foot],1)
        g.update(contact_points=points,contact_targets=g['contacts'],targets=g['contacts']-(points-g['joints'][:,:,2]),
                 foot_contacts=foot,foot_patch_local_cm=local,
                 foot_heel=g['foot_origins']-d['foot_depth']*F[:,:,:,2],
                 foot_toe=g['foot_origins']+d['foot_length']*F[:,:,:,0]-d['foot_depth']*F[:,:,:,2])
        return g

    def mechanical_values(self,y):
        g=self.geometry(y)
        m=mechanics(g,y[:,:self.qdim],y[:,self.qdim:self.statedim].reshape(self.n,4,3),g['contacts'],self.user_profile,self.fps)
        return np.c_[m['residual'],m['utilization']]

    def mechanical_jacobian(self,y):
        value=self.mechanical_values(y);width=value.shape[1];rows=[];cols=[];values=[]
        for color in range(min(5,self.n)):
            times=np.arange(color,self.n,5)
            affected,inputs=np.where(self.mechanical_pattern[:,times]);source=times[inputs]
            row=(affected[:,None]*width+np.arange(width)).ravel()
            for k in range(self.localdim):
                cy=y.astype(complex);cy[times,k]+=1e-25j
                derivative=self.mechanical_values(cy).imag/1e-25
                rows.extend(row);cols.extend(np.repeat(source*self.localdim+k,width));values.extend(derivative[affected].ravel())
        J=coo_matrix((values,(rows,cols)),shape=(self.n*width,self.n*self.localdim)).tocsr();J.eliminate_zeros()
        return value,J@self.expand

    def evaluate(self,flat):
        base,J=super().evaluate(flat)
        if not self.dynamic_balance:return base,J
        y=(self.expand@flat).reshape(self.n,self.localdim)
        values,M=self.mechanical_jacobian(y);width=values.shape[1]
        ir=(np.arange(self.n)[:,None]*width+np.arange(width-12)).ravel()
        iu=(np.arange(self.n)[:,None]*width+np.arange(width-12,width)).ravel()
        fatigue=fatigue_coupling(values[:,-12:],self.user_profile['biomechanics'],self.fps,True)
        return np.r_[base,values[:,:-12].ravel(),fatigue['residual'].ravel()],vstack([J,M[ir],fatigue['jacobian']@M[iu]],format='csr')


def audit_personalized(fit,metrics,data,scene,regions,profile):
    q=fit['anatomical_q'];qdim=q.shape[1];forces=fit['values'][:,qdim:qdim+12].reshape(-1,4,3)
    b=profile['biomechanics'];mech=mechanics(fit,q,forces,fit['contacts'],profile,data['fps'])
    fatigue=fatigue_coupling(mech['utilization'],b,data['fps'])
    fit.update(joint_moments_nm=mech['joint_moments_nm'],joint_capacity_nm=mech['capacity_nm'],
               fatigue_states=fatigue['states'],fatigue_utilization=fatigue['effective_utilization'],
               segment_centers_m=mech['segment_centers_m'],segment_inertia_kg_m2=mech['segment_inertia_kg_m2'],
               dynamic_com_acceleration_m_s2=mech['com_acceleration_m_s2'])
    m=metrics
    m['shoulder_envelope_failed']=mech['shoulder_excess_rad'].max(1)>1e-7
    m['self_collision_failed']=mech['collision_margin_m'].min(1)<-1e-6
    m['minimum_collision_margin_cm']=100*mech['collision_margin_m'].min(1)
    m['max_joint_moment_utilization']=mech['utilization'].max(1)
    m['torque_capacity_failed']=m.max_joint_moment_utilization>1+1e-6
    m['max_fatigue_utilization']=fatigue['effective_utilization'].max(1)
    m['max_fatigued_fraction']=fatigue['states'][:,:,2].max(1)
    m['fatigue_failed']=(m.max_fatigue_utilization>1+1e-6)|(m.max_fatigued_fraction>b['max_fatigued_fraction']+1e-6)
    m['dynamic_force_residual_bw']=np.linalg.norm(mech['force_residual_bw'],axis=-1)
    m['dynamic_moment_residual_bw_m']=np.linalg.norm(mech['moment_residual_bw_m'],axis=-1)
    m['dynamic_balance_failed']=(m.dynamic_force_residual_bw>b['force_tolerance_bw'])|(m.dynamic_moment_residual_bw_m>b['moment_tolerance_bw_m'])
    # Audit the SAME optimized forces used by inertia/torque/fatigue, not a fresh
    # static LP allocation that could hide a failure of the selected trajectory.
    panels=np.array([e['panel'] for e in regions['episodes']])[np.maximum(regions['mapping'],0)]
    normals=scene.normals[panels];nf=np.sum(forces*normals,axis=-1)
    t=np.broadcast_to([1.,0.,0.],normals.shape).copy();t-=normals*np.sum(t*normals,axis=-1,keepdims=True);t/=np.linalg.norm(t,axis=-1,keepdims=True)
    friction=np.any(nf[:,2:]< -1e-7,axis=1)
    for tangent in (t,np.cross(normals,t)):
        friction|=np.any(abs(np.sum(forces*tangent,axis=-1)[:,2:])>.8/np.sqrt(2)*nf[:,2:]+1e-7,axis=1)
    m['optimized_friction_failed']=friction
    m['optimized_force_bound_failed']=np.max(abs(forces),axis=(1,2))>1+1e-7
    m['free_force_failed']=np.any((np.linalg.norm(forces,axis=-1)>1e-7)&~data['known'],axis=1)
    m['quasi_static_balance']=m.conditional_balance
    m['quasi_static_preunload']=m.preunload
    m['quasi_static_eligible_balance']=m.eligible_balance
    m['conditional_balance']=~(m.dynamic_balance_failed|m.optimized_friction_failed|m.optimized_force_bound_failed|m.free_force_failed|m.contact_failed)
    used=np.linalg.norm(forces,axis=-1)>1e-7
    trustworthy=(data['confidence']>=.8)&~data['ground']&data['identity_observed'][:,None]
    m['eligible_balance']=m.conditional_balance&np.all(~used|trustworthy,axis=1)
    departing=data['unload']>=1-1e-9
    m['preunload']=[bool(m.conditional_balance.iloc[i] and np.all(np.linalg.norm(forces[i,departing[i]],axis=-1)<=.01)) if departing[i].any() else None for i in range(len(m))]
    keys=['shoulder_envelope','self_collision','torque_capacity','fatigue','dynamic_balance']
    for key in keys:
        m.loc[m[key+'_failed'],'issues']+=';'+key
    m['advanced_failure']=m[[k+'_failed' for k in keys]].any(axis=1)
    m['parameter_calibration_known']=False
    m['issues']+=';user_profile_assumptions;fatigue_initial_state_assumed;sampled_dynamics'
    mappings={'shoulder_envelope':'shoulder_envelope','self_collision':'self_collision','torque_capacity':'torque_capacity',
              'fatigue':'fatigue','strength_fatigue_coupling':'fatigue','centroidal_dynamics':'dynamic_balance'}
    fit['advanced_checks']=[dict(id=key,status='fail' if m[col+'_failed'].any() else 'pass',
        reason='Numerical check under user profile assumptions; no personal physiological calibration.',
        calibrated=False,failed_frames=int(m[col+'_failed'].sum())) for key,col in mappings.items()]
    return mech,fatigue
