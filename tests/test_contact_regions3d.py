"""Region topology, shared contact parameters, offsets and whole-plan ranking."""
import unittest
import numpy as np
from beta_translator.contact_regions3d import interior_rectangle,contacts_from_parameters,knot_basis
from beta_translator.region_ik3d import RegionProblem
from beta_translator.native_region3d import NativeRegionProblem
from beta_translator.trial3_region_pipeline import rank_candidates
from tests.test_spatial_experiment import synthetic


def fixture():
    d=synthetic(6);d['labels']=np.tile(['H1','H2','H3','H4'],(6,1))
    es=[]
    for i,p in enumerate(d['targets'][0]):
        es.append(dict(rectangle_px=[p[0]-3,p[1]-2,p[0]+3,p[1]+2],prior_px=p[:2].tolist(),
            center_cm=p.tolist(),offset_cm=[0.,-6. if i<2 else 8.,3.],evidence_available=True))
    return d,dict(episodes=es,mapping=np.tile(np.arange(4),(6,1)),H=np.eye(3))


class ContactRegionsTests(unittest.TestCase):
    def test_rectangle_does_not_bridge_concave_gap_or_hole(self):
        mask=np.ones((40,40),bool);mask[8:32,16:25]=False
        b=interior_rectangle(mask,[15,20],radius_px=30).astype(int)
        x0,y0,x1,y1=b
        self.assertTrue(mask[y0:y1+1,x0:x1+1].all())
        self.assertLess(x1,16)

    def test_contact_is_shared_and_offset_is_separate(self):
        d,r=fixture();uv=np.full((4,2),.5)
        c,o,e=contacts_from_parameters(uv,r,'region_offset')
        np.testing.assert_allclose(c,np.broadcast_to(e,c.shape))
        np.testing.assert_allclose(c[0],d['targets'][0])
        self.assertGreater(np.linalg.norm(o[0,0]),6)
        r['mapping'][2,0]=-1
        c,o,e=contacts_from_parameters(uv,r,'region_offset')
        np.testing.assert_array_equal(c[2,0],0);np.testing.assert_array_equal(o[2,0],0)

    def test_half_offset_is_an_explicit_sensitivity_candidate(self):
        _,r=fixture();uv=np.full((4,2),.5)
        a,b,_=contacts_from_parameters(uv,r,'region_offset')
        c,d,_=contacts_from_parameters(uv,r,'region_offset_half')
        np.testing.assert_array_equal(a,c);np.testing.assert_allclose(b,2*d)

    def test_every_native_frame_participates_in_basis(self):
        d,_=fixture();d['labels'][3:,0]='H5';k,B=knot_basis(d,4)
        np.testing.assert_allclose(np.asarray(B.sum(axis=1)).ravel(),1)
        self.assertTrue({2,3,4}.issubset(set(k)))
        np.testing.assert_allclose(B@k,np.arange(6))

    def test_full_global_jacobian_includes_shared_contact_and_temporal_derivatives(self):
        d,r=fixture();p=RegionProblem(d,r,'region_offset',stride=2)
        x=p.initial.copy();rr,J=p.evaluate(x)
        rng=np.random.default_rng(73);v=rng.normal(size=len(x));v/=np.linalg.norm(v)
        eps=1e-5;finite=(p.evaluate(x+eps*v)[0]-p.evaluate(x-eps*v)[0])/(2*eps)
        np.testing.assert_allclose(J@v,finite,rtol=2e-4,atol=2e-4)

    def test_direct_switch_relaxes_parameterization_but_preserves_contact_targets(self):
        d,r=fixture();d['labels'][3:,0]='H5';original=d['known'].copy()
        p=RegionProblem(d,r,'region_offset',stride=2)
        np.testing.assert_array_equal(p.known,original)
        self.assertFalse(p.analytic[:,0].any());self.assertTrue(p.direct_switch[:,0].all())
        x=p.initial.copy();x[:p.k*22].reshape(-1,22)[:,0]+=30
        y=(p.expand@x).reshape(p.n,30);j,_,_,c,t=p.geometry(y)
        self.assertGreater(np.linalg.norm(j[:,0,2]-t[:,0],axis=1).max(),.1)
        np.testing.assert_allclose(c[:,0],np.broadcast_to(c[0,0],c[:,0].shape))
        rr,J=p.evaluate(x);v=np.random.default_rng(21).normal(size=len(x));v/=np.linalg.norm(v)
        eps=1e-6;fd=(p.evaluate(x+eps*v)[0]-p.evaluate(x-eps*v)[0])/(2*eps)
        np.testing.assert_allclose(J@v,fd,rtol=3e-4,atol=3e-4)

    def test_exact_bones_with_offsets_and_unreachable_contact(self):
        d,r=fixture();r['episodes'][0]['rectangle_px']=[-3,900,3,910]
        r['episodes'][0]['prior_px']=[0,905]
        p=RegionProblem(d,r,'region_offset');y=(p.expand@p.initial).reshape(p.n,30)
        j,_,_,c,t=p.geometry(y)
        length=p.lengths;expected=np.array([[length['upper_arm'],length['forearm']]]*2+[[length['thigh'],length['shank']]]*2)
        np.testing.assert_allclose(np.linalg.norm(np.diff(j,axis=2),axis=-1),np.broadcast_to(expected,(6,4,2)),atol=1e-7)
        self.assertGreater(np.linalg.norm(j[0,0,2]-t[0,0]),100)

    def test_native_refinement_reconstructs_previous_skeleton_without_stitching(self):
        d,r=fixture();coarse=RegionProblem(d,r,'region_offset',stride=2)
        y=(coarse.expand@coarse.initial).reshape(coarse.n,30);j=coarse.geometry(y)[0]
        initial=dict(parameters=coarse.initial,values=np.c_[y[:,:22],np.zeros((coarse.n,12))],joints=j)
        p=NativeRegionProblem(d,r,initial);x=p.initial
        self.assertEqual(len(p.knots),len(d['frames']));self.assertFalse(p.analytic.any())
        actual=p.geometry((p.expand@x).reshape(p.n,30))[0]
        np.testing.assert_allclose(actual,j,atol=1e-9)
        rr,J=p.evaluate(x);v=np.random.default_rng(51).normal(size=len(x));v/=np.linalg.norm(v)
        eps=1e-6;fd=(p.evaluate(x+eps*v)[0]-p.evaluate(x-eps*v)[0])/(2*eps)
        np.testing.assert_allclose(J@v,fd,rtol=3e-4,atol=3e-4)

    def test_ranking_never_hides_failure_behind_support_score(self):
        base=dict(frames=10,geometric_failure_frames=0,temporal_warning_frames=0,converged=True,
            conditional_balance_frames=2,preunload_failure_frames=1,conditional_robust_frames=1,
            max_contact_gap_cm=.01,mean_offset_cm=8.)
        rows=[dict(base,profile='valid_contact'),dict(base,profile='missed_contact',geometric_failure_frames=1,conditional_balance_frames=10)]
        ranked,_=rank_candidates(rows);self.assertEqual(ranked[0]['profile'],'valid_contact')
        self.assertIn('not a physically validated',ranked[0]['selection_reason'])

if __name__=='__main__':unittest.main()
