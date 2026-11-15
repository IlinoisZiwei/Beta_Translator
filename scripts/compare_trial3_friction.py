#!/usr/bin/env python3
"""Matched soft/hard friction experiment, restricted to Trial3 frames 296–331."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.body_model import BodyModel
from beta_translator.body_pipeline import verified_run,write_json


def independent_friction_audit(out):
    """Recompute force inequalities without the fitting/audit helper."""
    with np.load(out/'steady/trajectory.npz') as z:
        forces=z['values'][:,32:44].reshape(-1,4,3);frames=z['frames']
        known=z['known'];q=z['anatomical_q'];labels=z['labels']
    episodes=json.loads((out/'steady/contact_episodes.json').read_text())
    normals=np.array(json.loads((out/'scene.json').read_text())['normals'])
    panels=np.zeros(known.shape,int)
    for episode in episodes:
        active=(frames>=episode['start_frame'])&(frames<=episode['end_frame'])
        panels[active,episode['limb_index']]=episode['panel']
    normal=normals[panels];fn=(forces*normal).sum(-1)
    tangent=np.broadcast_to([1.,0.,0.],normal.shape).copy()
    tangent-=normal*(tangent*normal).sum(-1,keepdims=True)
    tangent/=np.linalg.norm(tangent,axis=-1,keepdims=True)
    excess=[-fn[:,2:]]
    for t in (tangent,np.cross(normal,tangent)):
        excess.append(abs((forces*t).sum(-1)[:,2:])-.8/np.sqrt(2)*fn[:,2:])
    violation=np.maximum(0,np.concatenate(excess,axis=1)).max(1)
    return violation,fn[:,2:],frames,known,q,labels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=args.output_root.resolve();out.mkdir(parents=True,exist_ok=False)
    references=[root/'outputs/trial3/shared_body'/name for name in
                ['target_175cm_0296_0331_0349d243c3f8','target_175cm_0000_0502_1f9a2446bc6b']]
    for reference in references:verified_run(reference)
    protected=[*root.glob('*.ipynb'),*(root/'configs').glob('*results.json')]
    protected_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    # Frozen across both arms; only the hard-constraint switch differs.
    profile=json.loads((root/'outputs/trial3/shared_body/target_175cm_0296_0331_0349d243c3f8/user_profile.json').read_text())
    results={};rows=[]
    for mode,hard in [('soft',False),('hard',True)]:
        print(f'=== {mode}: Trial3 frames 296–331, steady, 35 iterations ===',flush=True)
        model=BodyModel(user_profile=profile,hard_foot_friction=hard)
        result=model.fit('trial3',project=root,frames=[296,331],iterations=35,
                         candidates=('steady',),output_root=out/mode,render=False)
        path=Path(result['output_dir']);verified_run(path);results[mode]=result
        m=pd.read_csv(path/'steady/metrics.csv');s=json.loads((path/'steady/summary.json').read_text())
        violation,normal,frames,known,q,labels=independent_friction_audit(path)
        np.testing.assert_allclose(violation,m.max_optimized_friction_violation_bw,atol=1e-12)
        np.testing.assert_array_equal(violation>1e-7,m.optimized_friction_failed)
        if hard:assert violation.max()<=1e-7,'Hard friction failed independent audit'
        assert frames.tolist()==list(range(296,332))
        rows.append(dict(mode=mode,frames=len(m),contact_failure_frames=int(m.contact_failed.sum()),
            max_contact_gap_cm=float(m.max_contact_gap_cm.max()),friction_failure_frames=int((violation>1e-7).sum()),
            max_friction_violation_bw=float(violation.max()),left_foot_pulling_frames=int((normal[:,0]<-1e-7).sum()),
            right_foot_pulling_frames=int((normal[:,1]<-1e-7).sum()),dynamic_balance_failure_frames=int(m.dynamic_balance_failed.sum()),
            support_failure_frames=int((~m.conditional_balance).sum()),free_force_failure_frames=int(m.free_force_failed.sum()),
            force_bound_failure_frames=int(m.optimized_force_bound_failed.sum()),
            anatomy_temporal_failure_frames=s['metrics']['anatomy_temporal_failure_frames'],
            torque_capacity_failure_frames=int(m.torque_capacity_failed.sum()),fatigue_failure_frames=int(m.fatigue_failed.sum()),
            departing_force_failure_frames=s['metrics']['departing_force_failure_frames'],
            converged=s['solver']['success'],iterations=s['solver']['nit'],cost=s['solver']['cost'],
            joint_acceleration_rms_cm_s2=s['metrics']['joint_acceleration_rms_cm_s2'],
            output_dir=str(path.relative_to(root))))
    soft,hard=(Path(results[k]['output_dir']) for k in ('soft','hard'))
    a=results['soft']['provenance'];b=results['hard']['provenance']
    assert a['input_sha256']==b['input_sha256']
    assert {k:v for k,v in a['config'].items() if k!='hard_foot_friction'}=={k:v for k,v in b['config'].items() if k!='hard_foot_friction'}
    with np.load(soft/'initialization.npz') as x,np.load(hard/'initialization.npz') as y:
        for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    for name in ['scene.json','user_profile.json','steady/contact_episodes.json']:
        assert (soft/name).read_bytes()==(hard/name).read_bytes()
    with np.load(soft/'steady/trajectory.npz') as x,np.load(hard/'steady/trajectory.npz') as y:
        for key in ['frames','known','labels','ground']:np.testing.assert_array_equal(x[key],y[key])
    for reference in references:verified_run(reference)
    for name,digest in protected_hashes.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    write_json(out/'comparison.json',dict(trial='trial3',frames=[296,331],iterations=35,candidate='steady',smoothness=2.5,
        changed_factor='hard_foot_friction',same_inputs=True,same_initialization=True,same_contact_schedule=True,
        same_scene_and_profile=True,independent_friction_audit_passed=True,
        existing_trial3_outputs_verified=True,protected_file_sha256=protected_hashes,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=rows,
        interpretation='Matched finite-budget comparison; contact remains soft and convergence/physical feasibility are separate checks.'))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(f'Comparison saved to {out}',flush=True)


if __name__=='__main__':main()
