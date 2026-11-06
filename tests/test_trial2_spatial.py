import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from beta_translator.trial2_scene3d import Trial2Scene
from beta_translator.trial2_spatial_data import load_trial2,episode_regions
from beta_translator.trial2_spatial_optimizer import Trial2Problem,force_audit,contact_geometry,rank_plans,transported_helper
from beta_translator.ik3d import kinematics
from beta_translator.trial2_spatial_pipeline import common_load_comparison

ROOT=Path(__file__).resolve().parents[1]


class Trial2SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene=Trial2Scene(ROOT,15,0)
        cls.full=load_trial2(ROOT)
        n=len(cls.full['frames']);indices=np.arange(420,428)
        cls.data={k:(v[indices] if isinstance(v,np.ndarray) and v.ndim and len(v)==n else v) for k,v in cls.full.items()}
        cls.regions=episode_regions(cls.data,cls.scene,180)

    def test_native_time_and_missing_observations_are_preserved(self):
        source=pd.read_csv(ROOT/'outputs/trial2/tables/pose_coordinates.csv')
        np.testing.assert_array_equal(self.full['frames'],np.arange(976))
        missing=np.setdiff1d(self.full['frames'],source.frame)
        self.assertGreater(len(missing),0)
        self.assertTrue((self.full['visibility'][missing]==0).all())

    def test_panel_hinges_are_shared_in_3d(self):
        for scene in [self.scene,Trial2Scene(ROOT,30,15),Trial2Scene(ROOT,45,0)]:
            for line,a,b in [('upper_hinge_px',0,1),('lower_hinge_px',1,2)]:
                pixels=np.array(scene.spec[line]);pixels=np.vstack([pixels,pixels.mean(0)])
                np.testing.assert_allclose(scene.intersect(pixels,a),scene.intersect(pixels,b),atol=1e-9)

    def test_each_panel_roundtrips_and_has_its_own_normal(self):
        for i,xy in enumerate([[1100.,200.],[1000.,550.],[1000.,820.],[1000.,900.]]):
            p=self.scene.intersect(xy,i)
            np.testing.assert_allclose(self.scene.pixels(p),xy,atol=1e-9)
            self.assertAlmostEqual(float((p-self.scene.origins[i])@self.scene.normals[i]),0,places=9)
        self.assertFalse(np.allclose(self.scene.normals[0],self.scene.normals[1]))

    def test_ground_needs_positive_visible_floor_evidence(self):
        d=self.full
        self.assertFalse(d['ground'][:,:2].any())
        self.assertFalse((d['ground']&d['hold_known']).any())
        self.assertTrue((~d['identity_observed'][:,None]&d['ground']).sum()==0)
        # Many off-hold feet must remain unsupported, rather than becoming ground.
        self.assertGreater(((d['raw_labels'][:,2:]=='free')&~d['ground'][:,2:]).sum(),100)
        for e in d['ground_evidence']:
            if e['ground_proposed']:
                self.assertTrue(e['visible']);self.assertLessEqual(e['floor_edge_gap_px'],18)
                self.assertLessEqual(e['source_step_px'],4)

    def test_contacts_are_shared_offsets_are_separate(self):
        uv=np.full((len(self.regions['episodes']),2),.37)
        c,o,n,_=contact_geometry(self.scene,self.regions,uv,'region_offset')
        for i in range(len(uv)):
            mask=self.regions['mapping']==i
            np.testing.assert_allclose(c[mask],np.broadcast_to(c[mask][0],c[mask].shape),atol=0)
            np.testing.assert_allclose(o[mask],np.broadcast_to(o[mask][0],o[mask].shape),atol=0)
        self.assertTrue((c[~self.data['known']]==0).all())
        self.assertTrue((o[~self.data['known']]==0).all())

    def test_global_jacobian_includes_episode_and_temporal_terms(self):
        p=Trial2Problem(self.data,self.scene,self.regions,support=True,stride=2)
        rng=np.random.default_rng(18);x=np.clip(p.initial+.02*rng.normal(size=p.initial.size),p.lower,p.upper)
        r,J=p.evaluate(x);direction=rng.normal(size=x.size)
        direction[(x-p.lower<1e-3)|(p.upper-x<1e-3)]=0
        eps=1e-5
        yp=(p.expand@(x+eps*direction)).reshape(p.n,42)
        ym=(p.expand@(x-eps*direction)).reshape(p.n,42)
        fd=(p.local(yp)-p.local(ym)).ravel()/(2*eps)
        analytic=(J@direction)[:len(fd)]
        np.testing.assert_allclose(analytic,fd,atol=2e-3,rtol=2e-4)
        # Full residual directional derivative also includes nonlocal XYZ motion.
        rp,_=p.evaluate(x+eps*direction);rm,_=p.evaluate(x-eps*direction)
        np.testing.assert_allclose(J@direction,(rp-rm)/(2*eps),atol=2e-3,rtol=2e-4)

    def test_fixed_bones_and_zero_unknown_forces(self):
        p=Trial2Problem(self.data,self.scene,self.regions,support=True)
        y=(p.expand@p.initial).reshape(p.n,42);j,_,_,_,_=p.geometry(y)
        actual=np.linalg.norm(np.diff(j,axis=2),axis=-1)
        b=p.lengths;expected=np.array([[b['upper_arm'],b['forearm']]]*2+[[b['thigh'],b['shank']]]*2)
        np.testing.assert_allclose(actual,np.broadcast_to(expected,actual.shape),atol=1e-10)
        forces=y[:,22:34].reshape(p.n,4,3)*p.known[...,None]
        self.assertTrue((forces[~p.known]==0).all())

    def test_native_refinement_preserves_entire_initial_geometry(self):
        parent=Trial2Problem(self.data,self.scene,self.regions,support=True,stride=2)
        y=(parent.expand@parent.initial).reshape(parent.n,42)
        j,com,head,_,_=parent.geometry(y)
        initial=dict(values=y[:,:34],joints=j,episode_uv=parent.initial[parent.k*34:].reshape(parent.e,2))
        native=Trial2Problem(self.data,self.scene,self.regions,support=True,stride=1,initial=initial,analytic_contacts=False)
        ny=(native.expand@native.initial).reshape(native.n,42)
        nj,nc,nh,_,_=native.geometry(ny)
        np.testing.assert_allclose(nj,j,atol=1e-9)
        np.testing.assert_allclose(nc,com,atol=1e-9)
        np.testing.assert_allclose(nh,head,atol=1e-9)
        self.assertFalse(native.analytic.any());self.assertEqual(len(native.knots),len(self.data['frames']))

    def test_contact_swivel_pole_cannot_shorten_bones(self):
        p=Trial2Problem(self.data,self.scene,self.regions)
        y=(p.expand@p.initial).reshape(p.n,42)
        _,_,_,_,target=p.geometry(y)
        roots=kinematics(y[:,:22],p.lengths)[0][:,:,0]
        delta=target-roots
        axis=delta/np.linalg.norm(delta,axis=-1,keepdims=True)
        # Reach aligns with the original helper, but the original reference
        # axis was orthogonal: this reproduces the old fixed-helper singularity.
        p.basis_helper=axis.copy()
        p.reach_axis=np.cross(axis,np.eye(3)[np.argmin(abs(axis),axis=-1)])
        p.reach_axis/=np.linalg.norm(p.reach_axis,axis=-1,keepdims=True)
        for perturbation in [0.,1e-12,1e-6]:
            p.basis_helper[...,0]+=perturbation
            j=p.geometry(y)[0]
            actual=np.linalg.norm(np.diff(j,axis=2),axis=-1)
            b=p.lengths;expected=np.array([[b['upper_arm'],b['forearm']]]*2+[[b['thigh'],b['shank']]]*2)
            np.testing.assert_allclose(actual,np.broadcast_to(expected,actual.shape),atol=1e-9)

    def test_transported_swivel_is_regular_through_old_pole(self):
        a=np.array([[1.,0.,0.]]);helper=np.array([[0.,0.,1.]])
        for angle in [0.,np.pi/2-1e-8,np.pi/2,np.pi/2+1e-8,np.pi]:
            axis=np.array([[np.cos(angle),0.,np.sin(angle)]])
            h=transported_helper(axis,a,helper)
            self.assertGreater(np.linalg.norm(np.cross(axis,h)),.9)

    def test_support_lp_respects_normals_and_unloading(self):
        points=np.array([[0,150,0],[10,150,0],[-10,0,0],[10,0,0.]])
        normal=np.tile([0.,1.,0.],(4,1));active=np.array([False,False,True,True])
        fit=force_audit(points,np.array([0,90,0.]),active,normal,180)
        self.assertTrue(fit['success']);self.assertAlmostEqual(fit['hand_load'],0)
        np.testing.assert_array_equal(fit['forces'][:2],0)
        self.assertFalse(force_audit(points,np.array([0,90,0.]),[False]*4,normal)['success'])
        self.assertFalse(force_audit(points,np.array([0,90,0.]),active,np.tile([0.,0.,1.],(4,1)))['success'])

    def test_complete_plan_rank_prioritizes_failure(self):
        base=dict(geometric_failure_frames=0,motion_warning_frames=0,nonconverged=0,support_failure_frames=0,
            preunload_failure_frames=0,low_margin_frames=0,hand_load_integral_bw_s=10)
        good=dict(base,candidate='good');bad=dict(base,candidate='bad',geometric_failure_frames=1,hand_load_integral_bw_s=0)
        ranked,_=rank_plans([bad,good]);self.assertEqual(ranked[0]['candidate'],'good')

    def test_appending_candidate_keeps_own_and_common_load_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);rows=[]
            for name,loads in [('a',[1.,2.,3.]),('b',[np.nan,4.,5.]),('c',[np.nan,np.nan,6.])]:
                (out/name).mkdir()
                pd.DataFrame({'conditional_balance':np.isfinite(loads),'hand_load_l1_bw':loads}).to_csv(out/name/'metrics.csv',index=False)
                rows.append(dict(candidate=name,hand_load_integral_bw_s=float(np.nansum(loads)/30)))
            first=common_load_comparison(out,rows[:2])
            extended=common_load_comparison(out,first+[rows[2]])
            self.assertAlmostEqual(extended[0]['own_feasible_hand_integral_bw_s'],6/30)
            self.assertAlmostEqual(extended[0]['hand_load_integral_bw_s'],3/30)
            self.assertEqual(extended[0]['common_load_frames'],1)


if __name__=='__main__':unittest.main()
