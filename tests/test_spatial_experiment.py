"""Independent geometry, derivative, physical-invariance and export checks."""
import json,struct,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from beta_translator.scene3d import project,tilt_rotation,upward_force,fit_camera,contact_span_certificate
from beta_translator.ik3d import kinematics,initial_pose,SpatialProblem,force_feasible,support_polygon
from beta_translator.inverse_kinematics import make_body
from beta_translator.scene3d_render import export_glb
from beta_translator.trial3_3d_pipeline import verified_experiment


def synthetic(n=3):
    reference=np.tile([0.,60.,0.,3.,.3,3.,-.3,0.,.2,0.,-.2],(n,1))
    lengths=make_body(180);q=initial_pose(reference,25,lengths)
    targets=np.tile([[-40.,150.,0.],[40.,150.,0.],[-20.,0.,0.],[20.,0.,0.]],(n,1,1))
    K=np.array([[600.,0,360],[0,600,640],[0,0,1.]])
    R=np.tile(np.diag([1.,-1.,-1.]),(n,1,1));t=np.tile([0.,70.,400.],(n,1))
    j=kinematics(q,lengths)[0];pixels,_=project(j,K,R,t)
    return dict(frames=np.arange(n),fps=30.,reference=reference,targets=targets,known=np.ones((n,4),bool),
        K=K,R=R,t=t,observed_px=pixels,visibility=np.ones((n,4,3)),supported=np.ones((n,4),bool),
        low_confidence=np.zeros((n,4),bool),unload=np.zeros((n,4)))


class SpatialTests(unittest.TestCase):
    def test_rigid_tilt_does_not_identify_gravity_or_change_image(self):
        rng=np.random.default_rng(3);X=rng.normal(size=(20,3))*40
        K=np.diag([600.,600.,1.]);R=cv2.Rodrigues(np.array([.1,.2,.3]))[0];t=np.array([0.,0.,500.])
        base,_=project(X,K,R,t)
        for angle in [0,15,30]:
            Q=tilt_rotation(angle);pred,_=project(X@Q.T,K,R@Q.T,t)
            np.testing.assert_allclose(pred,base,atol=1e-10)
            np.testing.assert_allclose(Q@upward_force(angle),[0,1,0],atol=1e-12)

    def test_bones_are_exact_with_reachable_and_unreachable_contacts(self):
        d=synthetic();body=make_body(180);q=initial_pose(d['reference'],25,body)
        d['targets'][1,0]=[0,500,0]
        j,_,_=kinematics(q,body,d['targets'],d['known'])
        actual=np.linalg.norm(np.diff(j,axis=2),axis=-1)
        target=np.array([[body['upper_arm'],body['forearm']]]*2+[[body['thigh'],body['shank']]]*2)
        np.testing.assert_allclose(actual,np.broadcast_to(target,actual.shape),atol=1e-7)
        self.assertGreater(np.linalg.norm(j[1,0,2]-d['targets'][1,0]),100)
        np.testing.assert_allclose(j[0,:,2],d['targets'][0],atol=1e-7)

    def test_span_certificate_survives_arbitrary_3d_rotation(self):
        b=make_body(180);p=np.array([[0.,250,0],[0,0,0],[0,0,0],[20,0,0]])
        a=contact_span_certificate(p,[True,False,True,False],b)[0]
        z=contact_span_certificate(p@tilt_rotation(30).T,[True,False,True,False],b)[0]
        self.assertGreater(a['excess_cm'],0);self.assertAlmostEqual(a['excess_cm'],z['excess_cm'])

    def test_camera_recovers_synthetic_observations(self):
        xy=np.array([(x,y) for x in range(-60,61,15) for y in range(-120,121,15)],float)
        X=np.c_[xy,np.zeros(len(xy))];K=np.array([[620.,0,360],[0,620,640],[0,0,1.]])
        R=cv2.Rodrigues(np.array([2.9,.15,.2]))[0];t=np.array([10.,50.,450.])
        px,_=project(X,K,R,t);fit=fit_camera(X,px,np.arange(len(X))%5!=0,[720,1280])
        self.assertLess(fit['reserved_rmse_px'],1e-5);self.assertAlmostEqual(fit['K'][0][0],620.,places=3)

    def test_no_contact_cannot_support_weight(self):
        self.assertFalse(force_feasible(np.zeros((4,3)),np.zeros(3),np.zeros(4,bool),0))

    def test_polygon_vertices_have_independent_force_solutions(self):
        points=synthetic(1)['targets'][0];poly=support_polygon(points,np.ones(4,bool),15)
        self.assertGreater(len(poly['vertices']),2)
        g=upward_force(15)
        for x,h in poly['vertices']:
            com=poly['center']+180*np.array([x,-g[2]*h,g[1]*h])
            self.assertTrue(force_feasible(points,com,np.ones(4,bool),15))

    def test_full_temporal_jacobian_matches_finite_difference(self):
        d=synthetic();d['known'][1,0]=False
        p=SpatialProblem(d,15,25);x=np.c_[p.seed,np.tile([.1,.25,.15],(3,4))].ravel()
        r,J=p.evaluate(x);rng=np.random.default_rng(19);direction=rng.normal(size=len(x));direction/=np.linalg.norm(direction)
        eps=1e-5;finite=(p.evaluate(x+eps*direction)[0]-p.evaluate(x-eps*direction)[0])/(2*eps)
        np.testing.assert_allclose(J@direction,finite,atol=2e-4,rtol=2e-4)

    def test_swivel_basis_does_not_jump_at_axis_selection_boundary(self):
        d=synthetic(1);body=make_body(180)
        d['targets'][0,0]=[-body['shoulder_w']/2+50*np.sqrt(1-.95**2),60+body['torso'],0]
        p=SpatialProblem(d,15,47.5,force_weight=0,support_weight=0)
        x=np.c_[p.seed,np.zeros((1,12))].ravel();r,J=p.evaluate(x)
        v=np.zeros_like(x);v[0]=1.;eps=1e-6
        finite=(p.evaluate(x+eps*v)[0]-p.evaluate(x-eps*v)[0])/(2*eps)
        np.testing.assert_allclose(J@v,finite,atol=1e-3,rtol=1e-3)

    def test_gapped_frames_cannot_be_presented_as_continuous(self):
        d=synthetic();d['frames']=np.array([1,3,4])
        with self.assertRaises(ValueError):SpatialProblem(d,15,25)

    def test_cached_artifact_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'summary.json').write_text(json.dumps({'provenance':{},'output_sha256':{'trajectory.npz':'bad'}}))
            (p/'trajectory.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):verified_experiment(p,{})

    def test_glb_is_real_mesh_with_meter_units(self):
        d=synthetic(1);j,_,h=kinematics(initial_pose(d['reference'],25,make_body(180)),make_body(180))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'scene.glb';export_glb(p,np.array([[-100,-100,0],[100,-100,0],[100,200,0],[-100,200,0]]),j[0],h[0],15,{})
            b=p.read_bytes();magic,version,length=struct.unpack('<III',b[:12]);self.assertEqual((magic,version,length),(0x46546C67,2,len(b)))
            size,kind=struct.unpack('<II',b[12:20]);doc=json.loads(b[20:20+size]);self.assertEqual(doc['extras']['units'],'meters')
            self.assertGreater(doc['accessors'][0]['count'],100);self.assertEqual(doc['scene'],0)


if __name__=='__main__':unittest.main()
