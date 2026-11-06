"""Independent mechanics cases and all-trial derivatives, not human validation."""
import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
import numpy as np
from beta_translator.body_user import load_user
from beta_translator.body_model import BodyModel
from beta_translator.body_mechanics import fatigue_coupling,mechanics,segment_distance,shoulder_limits
from beta_translator.body_checks import segment_distance as oracle_distance


class UserProfiles(unittest.TestCase):
    def test_span_geometry_and_explicit_conflicts(self):
        p=load_user();d=p['dimensions_cm']
        self.assertEqual(p['height_cm'],175)
        self.assertEqual(d['shoulder_w']+2*(d['upper_arm']+d['forearm']+d['hand_length']),176)
        p['dimensions_cm']['forearm']+=1
        with self.assertRaises(ValueError):load_user(p)
        with self.assertRaises(ValueError):load_user(load_user(),height_cm=180)

    def test_validation_and_independent_snapshots(self):
        a=load_user();b=load_user();a['mass_kg']=0
        with self.assertRaises(ValueError):load_user(a)
        self.assertEqual(b['mass_kg'],70)
        b['biomechanics']['mass_fractions'][0]+=.1
        with self.assertRaises(ValueError):load_user(b)
        self.assertNotEqual(BodyModel().ratios['upper_arm'],.186)

    def test_nonfinite_envelope_and_clearance_rejected(self):
        p=load_user();p['biomechanics']['shoulder_envelope']['plane_deg'][1]=float('nan')
        with self.assertRaises(ValueError):load_user(p)
        p=load_user();p['biomechanics']['collision_clearance_cm']=float('nan')
        with self.assertRaises(ValueError):load_user(p)

    def test_changed_model_profile_cannot_use_stale_geometry(self):
        m=BodyModel();m.user_profile['dimensions_cm']['thigh']+=1
        with self.assertRaises(ValueError):m.fit('trial1',render=False)

    def test_envelope_depends_on_posture(self):
        b=load_user()['biomechanics']['shoulder_envelope']
        q=np.deg2rad([[20,60,-80],[20,180,-80]])
        lo,hi=shoulder_limits(q,b)
        self.assertGreater(q[0,2],lo[0]);self.assertLess(q[1,2],lo[1])

    def test_capsules_against_independent_scalar_oracle(self):
        rng=np.random.default_rng(5810);s=rng.normal(size=(80,2,3));t=rng.normal(size=(80,2,3))
        s[0,1]=s[0,0];t[1,1]=t[1,0];t[2]=s[2]+[0,2,0]
        actual=segment_distance(s,t)
        expected=[oracle_distance(*a,*b) for a,b in zip(s,t)]
        np.testing.assert_allclose(actual,expected,atol=1e-10)

    def test_fatigue_conservation_recovery_and_causal_jacobian(self):
        c=load_user()['biomechanics'];d=np.full((9,12),.7)
        r=fatigue_coupling(d,c,30,True);states=r['states']
        self.assertGreater(states[-1,0,2],0)
        np.testing.assert_allclose(states.sum(-1),1,atol=1e-12);self.assertTrue(np.all(states>=0))
        self.assertGreater(r['effective_utilization'][-1,0],d[-1,0])
        c={**c,'initial_fatigue_state':[0,.5,.5]}
        resting=fatigue_coupling(np.zeros((30,12)),c,30)
        self.assertLess(resting['states'][-1,0,2],.5)
        c={**c,'max_fatigued_fraction':.1};rng=np.random.default_rng(51);v=rng.normal(size=d.shape);eps=1e-6
        r=fatigue_coupling(d,c,30,True)
        fd=(fatigue_coupling(d+eps*v,c,30)['residual']-fatigue_coupling(d-eps*v,c,30)['residual'])/(2*eps)
        np.testing.assert_allclose(r['jacobian']@v.ravel(),fd.ravel(),atol=2e-7)
        self.assertEqual(r['jacobian'][:36,12:].nnz,0)

    def test_fatigue_jacobian_with_clipped_and_active_recruitment(self):
        c=load_user()['biomechanics'];rng=np.random.default_rng(976)
        demand=rng.choice([-.2,.35,.7,1.4],size=(29,12))
        direction=rng.normal(size=demand.shape);eps=1e-6
        result=fatigue_coupling(demand,c,30,True)
        fd=(fatigue_coupling(demand+eps*direction,c,30)['residual']
            -fatigue_coupling(demand-eps*direction,c,30)['residual'])/(2*eps)
        np.testing.assert_allclose(result['jacobian']@direction.ravel(),fd.ravel(),rtol=1e-6,atol=2e-7)
        for frame in range(len(demand)):
            self.assertEqual(result['jacobian'][frame*36:(frame+1)*36,(frame+1)*12:].nnz,0)


class IntegratedMechanics(unittest.TestCase):
    def test_curve_controls_preserve_native_anatomy_forces_and_derivatives(self):
        from beta_translator.temporal_control_basis import control_map
        from beta_translator.constrained_sequence_solver import project_seed
        p=self.problem('trial1',n=9)
        T,z,lower,upper,info=control_map(p,4)
        z,_=project_seed(z,lower,upper,p.hard_rates@T)
        native=T@z;y=(p.expand@native).reshape(p.n,p.localdim)
        self.assertEqual(y.shape,(9,52))
        self.assertLess(T.shape[1],T.shape[0])
        self.assertLessEqual(np.max(abs(p.hard_rates@native)),1+1e-7)
        force=y[:,p.qdim:p.statedim].reshape(p.n,4,3)
        np.testing.assert_array_equal(force[~p.known],0)
        np.testing.assert_array_less(p.lower-1e-7,native)
        np.testing.assert_array_less(native,p.upper+1e-7)
        # Check smooth branches away from zero-force/friction and rate hinges.
        rng=np.random.default_rng(71);z=z+.0005*rng.normal(size=len(z));native=T@z
        r,J=p.evaluate(native);v=rng.normal(size=len(z));eps=1e-7
        fd=(p.evaluate(T@(z+eps*v))[0]-p.evaluate(T@(z-eps*v))[0])/(2*eps)
        np.testing.assert_allclose((J@T)@v,fd,rtol=3e-5,atol=.005)

    @staticmethod
    def problem(trial,n=5):
        from beta_translator.trials import load_config,load_trial
        from beta_translator.body_pipeline import sliced
        from beta_translator.body_legs import initialize
        from beta_translator.body_personalized import PersonalizedProblem
        from beta_translator.inverse_kinematics import make_body
        root=Path(__file__).resolve().parents[1];model=BodyModel();cfg=load_config(root,trial)
        _,data,scene,regions,old=load_trial(root,cfg,175);count=len(data['frames'])
        ix=np.arange(cfg['verification_frames'][0],cfg['verification_frames'][0]+n)
        d=sliced(data,ix,count);l=sliced(old,ix,count);r={**regions,'mapping':regions['mapping'][ix]}
        q=initialize(l,make_body(175,model.ratios),175)
        return PersonalizedProblem(d,scene,r,l,q,175,ratios=model.ratios,user_profile=model.user_profile)

    def test_all_trials_native_derivative_and_contact_identity(self):
        for name in ('trial1','trial2','trial3'):
            with self.subTest(trial=name):
                p=self.problem(name,n=7);x=p.initial.copy();rng=np.random.default_rng(19)
                x+=.0003*rng.normal(size=len(x));v=rng.normal(size=len(x));eps=2e-7
                r,J=p.evaluate(x);fd=(p.evaluate(x+eps*v)[0]-p.evaluate(x-eps*v)[0])/(2*eps)
                np.testing.assert_allclose(J@v,fd,rtol=3e-5,atol=.003)
                y=(p.expand@x).reshape(p.n,p.localdim);g=p.geometry(y)
                np.testing.assert_allclose(g['joints'][:,:,2]-g['targets'],g['contact_points']-g['contacts'],atol=1e-12)
                np.testing.assert_allclose(np.linalg.norm(g['hand_tips']-g['joints'][:,:2,2],axis=-1),14,atol=1e-12)

    def test_static_freefall_and_mass_scaled_wrist_moment(self):
        p=self.problem('trial1');y=(p.expand@p.initial).reshape(p.n,p.localdim).copy();y[:]=y[0]
        q=y[:,:p.qdim];g=p.geometry(y);forces=np.zeros((p.n,4,3));forces[:,0,1]=1
        points=np.repeat(g['com'][:,None],4,1)
        m=mechanics(g,q,forces,points,p.user_profile,p.fps)
        np.testing.assert_allclose(m['force_residual_bw'],0,atol=1e-10)
        np.testing.assert_allclose(m['moment_residual_bw_m'],0,atol=1e-10)
        heavy=deepcopy(p.user_profile);heavy['mass_kg']*=2
        mm=mechanics(g,q,forces,points,heavy,p.fps)
        np.testing.assert_allclose(mm['joint_moments_nm'],2*m['joint_moments_nm'],atol=1e-9)
        t=np.arange(p.n)/p.fps;y[:,1]-=.5*981*t*t
        g=p.geometry(y);free=mechanics(g,y[:,:p.qdim],forces*0,g['contacts'],p.user_profile,p.fps)
        np.testing.assert_allclose(free['force_residual_bw'],0,atol=1e-10)
        np.testing.assert_allclose(free['moment_residual_bw_m'],0,atol=1e-10)

    def test_profile_and_advanced_audits_are_exported(self):
        from beta_translator.body_pipeline import verified_run
        import json
        with tempfile.TemporaryDirectory() as temp:
            r=BodyModel().fit('trial3',project=Path(__file__).resolve().parents[1],frames=[296,298],iterations=1,
                candidates=('anatomical_control','steady'),output_root=temp,render=False,progress=lambda *a,**k:None)
            out=Path(r['output_dir']);verified_run(out)
            self.assertEqual(json.loads((out/'user_profile.json').read_text())['arm_span_cm'],176)
            checks={x['id']:x for x in json.loads((out/'steady/constraint_results.json').read_text())}
            for name in ('shoulder_envelope','self_collision','torque_capacity','fatigue','centroidal_dynamics'):
                self.assertIn(checks[name]['status'],('pass','fail'));self.assertFalse(checks[name]['calibrated'])
            self.assertEqual(checks['parameter_calibration']['status'],'unknown')
            self.assertFalse(r['physical_validation'])

if __name__=='__main__':unittest.main()
