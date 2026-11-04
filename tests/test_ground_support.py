import copy
import json
from pathlib import Path
import unittest
import numpy as np

from beta_translator.contact_geometry import contact_points
from beta_translator.full_sequence import build_schedule, diagnose_sequence, solve_full_sequence
from beta_translator.gravity_sequence import GravityProblem, diagnose_gravity_sequence, solve_gravity_sequence
from beta_translator.ground_support import classify_ground, overlay_ground, validate_annotation
from beta_translator.support_audit import release_events, force_reserve, audit_settings, audit_support_sequence
from beta_translator.support_optimization import SupportProblem, prospective_constraints
from beta_translator.trial1_sequence import TRIAL1_VIDEO_SHA256
from tests.test_full_sequence import fixture


class GroundSupportTests(unittest.TestCase):
    def fixture(self, n=8):
        q,b,e,table,g=fixture(n)
        s=build_schedule(table,np.arange(n),g)
        s['labels'][:,2:]='free';s['known'][:,2:]=False;s['supported'][:,2:]=False
        evidence=[]
        for t in range(n):
            for i in (2,3):
                evidence.append(dict(frame=t,limb_index=i,state='observed_ground' if t<5 else 'off_ground',
                    support_id=f'ground_{i}',contact_point_cm=(e[t,i]-[0,b['height_cm']*.025]).tolist()))
        a={'sole_offset_height_ratio':.025,'ground_fit_weight':.8}
        return q,b,e,s,overlay_ground(s,evidence,b['height_cm'],a)

    def test_free_and_in_polygon_never_prove_ground(self):
        self.assertEqual(classify_ground(None,True,True,True),'ground_candidate')
        self.assertEqual(classify_ground('off_ground',True,True,True),'off_ground')
        self.assertEqual(classify_ground(None,False,True,True),'unknown')
        self.assertEqual(classify_ground('observed_ground',False,True,True),'unknown')
        q,b,e,s,_=self.fixture()
        for state in ('ground_candidate','unknown','off_ground'):
            r=overlay_ground(s,[dict(frame=0,limb_index=2,state=state)],180,{'sole_offset_height_ratio':.025,'ground_fit_weight':.8})
            self.assertFalse(r['known'][0,2]);self.assertEqual(r['support_ids'][0,2],'')

    def test_labels_confidences_and_old_schedule_unchanged(self):
        q,b,e,s,r=self.fixture()
        np.testing.assert_array_equal(r['labels'],s['labels'])
        np.testing.assert_array_equal(r['confidence'],s['confidence'])
        self.assertFalse(s['known'][:,2:].any())
        self.assertTrue(r['known'][:5,2:].all())
        self.assertFalse(r['supported'][:,2:].any())

    def test_conflicting_hold_not_overwritten(self):
        q,b,e,s,r=self.fixture();s['known'][0,2]=True;s['labels'][0,2]='H01';s['supported'][0,2]=True
        rr=overlay_ground(s,[dict(frame=0,limb_index=2,state='observed_ground')],180,{'sole_offset_height_ratio':.025,'ground_fit_weight':.8})
        self.assertTrue(rr['ground_conflict'][0,2]);self.assertEqual(rr['contact_type'][0,2],'hold')
        np.testing.assert_array_equal(rr['targets'],s['targets'])

    def test_liftoff_detected_even_when_raw_free_is_unchanged(self):
        q,b,e,s,r=self.fixture();events=release_events(r,r['frames'])
        event=next(x for x in events if x['release_frame']==5)
        self.assertEqual(event['moving_indices'],[2,3]);self.assertFalse(event['timing_uncertain'])
        self.assertEqual(event['timing_reason'],'reviewed_ground_liftoff')
        rows,unload=prospective_constraints(r,r['frames'],1,.8,audit_settings({'unload_lookahead_frames':3}))
        self.assertEqual(set(x['index'] for x in rows if x['kind']=='unloading'),{2,3,4})
        self.assertTrue(all(not {2,3}&set(x['limbs']) for x in rows if x['kind']=='unloading'))
        self.assertFalse(unload[5:].any())
        r['ground_timing_uncertain'][5,2]=True
        self.assertTrue(next(x for x in release_events(r,r['frames']) if x['release_frame']==5)['timing_uncertain'])

    def test_sole_not_ankle_and_analytic_jacobian(self):
        q,b,e,s,r=self.fixture()
        ends=contact_points(e,r)
        np.testing.assert_allclose(ends[:5,2:,1],e[:5,2:,1]-.025*b['height_cm'])
        p=SupportProblem(q,np.ones((8,9)),e,r,b,30,70,.8,None,None);p.configure(q,None,None)
        x=np.c_[q+np.arange(11)*.01,np.tile([.1,.2,-.1,.3,.3,.1,-.2,.1],(8,1))]
        j=p.local_jacobian(x,slice(None))
        for k in range(19):
            a=x.copy();bb=x.copy();a[:,k]+=1e-6;bb[:,k]-=1e-6
            np.testing.assert_allclose(j[:,:,k],(p.local(a,slice(None))-p.local(bb,slice(None)))/2e-6,rtol=3e-5,atol=2e-5)
        # No departure force enters balance, moment, friction, coupling or unload.
        self.assertFalse(j[5:,27:32,15:].any())
        self.assertFalse(j[5:,40:64,15:].any())

    def test_geometry_only_solver_does_not_silently_use_ankle_for_ground(self):
        q,b,e,s,r=self.fixture()
        with self.assertRaisesRegex(ValueError,'gravity-sequence'):
            solve_full_sequence(q,np.ones((8,9)),e,r,b,30)

    def test_ground_force_is_push_only_and_friction_limited(self):
        targets=np.array([[0,0],[0,0],[-10,0],[10,0]],float)
        a=force_reserve(targets,[False,False,True,True],[0,100],100,100,.4,50)
        self.assertEqual(a['status'],'feasible')
        self.assertTrue((a['forces_n'][2:,1]>=0).all())
        self.assertTrue((abs(a['forces_n'][2:,0])<=.4*a['forces_n'][2:,1]+1e-8).all())
        self.assertFalse(force_reserve(targets,[False]*4,[0,100],100,100,.4,50)['forces_n'].any())

    def test_actual_solve_no_force_after_liftoff(self):
        q,b,e,s,r=self.fixture()
        result=solve_gravity_sequence(q,np.ones((8,9)),e,r,b,30,settings={'max_nfev':30})
        self.assertFalse(result['contact_forces_n'][5:,2:].any())
        d=diagnose_gravity_sequence(result)
        expected=np.linalg.norm(contact_points(result['joints'][:,:,2],r)-r['targets'],axis=2)
        np.testing.assert_allclose(d['contact_errors_cm'][r['known']],expected[r['known']])
        a=audit_support_sequence(result,d)
        self.assertFalse(a['eligible'][:,2:].any())

    def test_stale_annotation_and_overlaps_rejected(self):
        a=json.loads((Path(__file__).resolve().parents[1]/'data/trial1_ground_support.json').read_text())
        validate_annotation(a,TRIAL1_VIDEO_SHA256,417)
        with self.assertRaises(ValueError):validate_annotation(a,'wrong',417)
        aa=copy.deepcopy(a);aa['intervals']['left_foot'].append(aa['intervals']['left_foot'][0])
        with self.assertRaises(ValueError):validate_annotation(aa,TRIAL1_VIDEO_SHA256,417)


if __name__=='__main__':unittest.main()
