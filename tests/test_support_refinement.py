import unittest
import numpy as np
from scipy.sparse import csr_matrix, diags
from unittest.mock import patch
from tests.test_full_sequence import fixture
from beta_translator.full_sequence import build_schedule
from beta_translator.gravity_sequence import solve_gravity_sequence,batch_com_and_jacobian
from beta_translator.support_optimization import SupportProblem
from beta_translator.support_refinement import (sparse_direct_least_squares,stationarity,com_coordinates,
    physical_coordinates,singleton_com_constraints,RefinementObjective,refine_support_sequence)


class RefinementTests(unittest.TestCase):
    def result(self,n=6):
        q,b,e,table,g=fixture(n);s=build_schedule(table,np.arange(n),g)
        return solve_gravity_sequence(q,np.ones((n,9)),e,s,b,30,settings={'max_nfev':50})

    def test_direct_solver_scaled_quadratic_and_monotone_cost(self):
        a=diags([.05,2.,1000.],format='csr');target=np.array([3.,-2.,.1])
        result=sparse_direct_least_squares(lambda x:(a@(x-target),a),np.zeros(3),np.full(3,-np.inf),np.full(3,np.inf),
            {'max_inner_iterations':100,'projected_gradient_tolerance':1e-8})
        self.assertTrue(result['success'],result['message'])
        np.testing.assert_allclose(result['x'],target,atol=1e-6)
        self.assertTrue(np.all(np.diff([r['cost'] for r in result['history']])<0))

    def test_exact_kkt_bound_and_fixed_coordinate(self):
        a=diags([1.,1.],format='csr')
        result=sparse_direct_least_squares(lambda x:(x-[-2.,3.],a),[4.,7.],[0.,7.],[np.inf,7.])
        self.assertTrue(result['success'],result['message']);np.testing.assert_allclose(result['x'],[0.,7.])
        self.assertEqual(result['optimality'],0.)

    def test_iteration_limit_is_not_convergence(self):
        def rosen(x):
            return np.array([10*(x[1]-x[0]**2),1-x[0]]),csr_matrix([[-20*x[0],10],[-1,0]])
        r=sparse_direct_least_squares(rosen,[-1.2,1.],[-np.inf]*2,[np.inf]*2,{'max_inner_iterations':1})
        self.assertFalse(r['success']);self.assertGreater(r['optimality'],1e-5)

    def test_com_coordinate_roundtrip_and_transform(self):
        r=self.result();x=np.c_[r['poses'],r['contact_forces_n'].reshape(-1,8)/(70*9.81)]
        u=com_coordinates(x,r['lengths']);physical,t=physical_coordinates(u,r['lengths'])
        np.testing.assert_allclose(physical,x,atol=1e-12)
        for k in range(19):
            a=u.copy();b=u.copy();a[:,k]+=1e-6;b[:,k]-=1e-6
            derivative=(physical_coordinates(a,r['lengths'])[0]-physical_coordinates(b,r['lengths'])[0])/2e-6
            np.testing.assert_allclose(t[:,:,k],derivative,atol=1e-7,rtol=1e-6)

    def test_only_true_nominal_singletons_are_exact(self):
        r=self.result();s=r['schedule'];s['targets'][:,:2]=[20.,100.]
        s['known'][:,2:]=False;s['supported'][:,2:]=False;s['low_confidence'][:,:2]=False
        p=SupportProblem(r['reference'],r['visibility'],r['reference_endpoints'],s,r['body'],30,70,.8,None,r['settings'])
        p.configure(r['poses'],None,None);fixed,rows=singleton_com_constraints(p)
        self.assertEqual(len(fixed),len(r['frames']));self.assertTrue(all(abs(v-20)<1e-9 for v in fixed.values()))
        # A finite interval collapsed by the +/-2 cm sensitivity target is NOT
        # an exact nominal equality and must not be treated as one.
        with patch('beta_translator.support_refinement.support_interval',return_value={'bounds':[19.9,20.1],'status':'available'}):
            self.assertFalse(singleton_com_constraints(p)[0])

    def test_global_jacobian_includes_temporal_com_chain_rule(self):
        r=self.result();obj=RefinementObjective(r,30);u=obj.initial_coordinates(r)
        u.reshape(-1,19)[:,11:]=np.array([.15,.25,-.1,.3,.25,.12,-.23,.11])
        u.reshape(-1,19)[2,0]+=9.;u.reshape(-1,19)[3,3]+=.4
        f,j=obj.evaluate(u);rng=np.random.default_rng(43)
        for _ in range(3):
            v=rng.normal(size=u.shape);v/=np.linalg.norm(v)
            numeric=(obj.evaluate(u+1e-6*v)[0]-obj.evaluate(u-1e-6*v)[0])/2e-6
            np.testing.assert_allclose(j@v,numeric,rtol=1e-4,atol=5e-5)

    def test_conflicting_singletons_are_not_silently_enforced(self):
        r=self.result();s=r['schedule'];s['targets'][0,0]=[10.,100.];s['targets'][0,1]=[20.,100.]
        p=SupportProblem(r['reference'],r['visibility'],r['reference_endpoints'],s,r['body'],30,70,.8,None,r['settings'])
        p.constraints=[dict(index=0,kind='reserve',group='eligible',event_id='',limbs=[i],lower_cm=0.) for i in (0,1)]
        fixed,records=singleton_com_constraints(p)
        self.assertFalse(fixed);self.assertEqual(len(records),2)
        self.assertTrue(all(row['status']=='conflicting_singletons_not_enforced' for row in records))

    def test_unknown_forces_fixed_and_contact_evidence_preserved(self):
        r=self.result();r['schedule']['known'][3:,2]=False;r['schedule']['supported'][3:,2]=False
        labels=r['schedule']['labels'].copy();targets=r['schedule']['targets'].copy()
        obj=RefinementObjective(r,30)
        self.assertTrue((obj.lower[3:,15:17]==0).all());self.assertTrue((obj.upper[3:,15:17]==0).all())
        new=refine_support_sequence(r,30,refinement={'max_outer_iterations':2,'max_inner_iterations':50},progress=None)
        self.assertFalse(new['contact_forces_n'][3:,2].any())
        np.testing.assert_array_equal(labels,new['schedule']['labels']);np.testing.assert_array_equal(targets,new['schedule']['targets'])
        for i in range(4):
            for j,name in enumerate(('upper_arm','forearm') if i<2 else ('thigh','shank')):
                np.testing.assert_allclose(np.linalg.norm(new['joints'][:,i,j+1]-new['joints'][:,i,j],axis=1),new['lengths'][name],atol=1e-10)

    def test_outer_reweight_stationarity_is_required(self):
        r=self.result();r['poses'][:,1]+=5.
        def false_inner_success(evaluate,x0,*args,**kwargs):
            return dict(x=x0,success=True,nfev=1,cost=0.,optimality=0.,history=[],gradient_definition='fixture')
        with patch('beta_translator.support_refinement.sparse_direct_least_squares',side_effect=false_inner_success):
            result=refine_support_sequence(r,30,refinement={'max_outer_iterations':1},progress=None)
        self.assertFalse(result['optimizer']['success'])
        self.assertFalse(result['optimization_history'][0]['outer_converged'])


if __name__=='__main__':unittest.main()
