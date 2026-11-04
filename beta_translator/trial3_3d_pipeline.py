"""Reproducible Trial3 3D hypothesis experiment; all older outputs stay read-only."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from .full_sequence_pipeline import load_full_sequence_inputs
from .pose_adapter import LIMB_LANDMARKS
from .inverse_kinematics import LIMBS,make_body
from .scene3d import build_cameras,contact_span_certificate
from .ik3d import SpatialProblem,audit
from .inverse_kinematics import RATIOS
from .project_io import file_sha256
from .full_sequence_pipeline import input_provenance

BASELINE='outputs/trial3/support_optimized/target_180cm_a9a3f35dffe1'
KEYFRAMES=[92,147,250,296,314,331,428,486]
CLIPS={'reachable':(296,331),'span_conflict':(474,497)}


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_inputs(project):
    project=Path(project);old=load_full_sequence_inputs(project);frames=np.arange(len(old['reference']))
    camera,R,t,residual=build_cameras(project,frames)
    pose=pd.read_csv(project/'outputs/trial3/tables/pose_coordinates.csv').set_index('frame')
    names=[list(LIMB_LANDMARKS[l]) for l in LIMBS]
    px=np.array([[[[pose.loc[f,n+'_x_px'],pose.loc[f,n+'_y_px']] for n in chain] for chain in names] for f in frames])
    visibility=np.array([[[pose.loc[f,n+'_visibility'] for n in chain] for chain in names] for f in frames])
    s=old['schedule'];targets=np.concatenate([s['targets'],np.zeros((*s['targets'].shape[:2],1))],axis=2)
    unload=np.zeros(s['known'].shape)
    for i in range(4):
        for f in range(1,len(frames)):
            if s['known'][f-1,i] and s['labels'][f,i]!=s['labels'][f-1,i]:
                for j in range(max(0,f-6),f):
                    if s['labels'][j,i]==s['labels'][f-1,i]:unload[j,i]=max(unload[j,i],(j-(f-6)+1)/6)
    return dict(frames=frames,fps=old['fps'],reference=old['reference'],observed_px=px,
        visibility=np.clip(visibility,.02,1.),known=s['known'],supported=s['supported'],low_confidence=s['low_confidence'],
        labels=s['labels'],confidence=s['confidence'],targets=targets,unload=unload,
        K=np.asarray(camera['K']),R=R,t=t,camera_residual_px=residual,camera=camera,old=old)


def subset(data,frames):
    frames=np.asarray(frames,int)
    return {k:(v[frames] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(data['frames']) else v)
            for k,v in data.items() if k!='old'}


def failure_review(project,data,out,height_cm,ratios=None):
    old=pd.read_csv(Path(project)/BASELINE/'frame_diagnostics.csv');body=make_body(height_cm,ratios)
    pair_rows=[];rows=[]
    for f in data['frames']:
        pairs=contact_span_certificate(data['targets'][f],data['known'][f],body)
        for p in pairs:
            pair_rows.append(dict(frame=int(f),limb_a=LIMBS[p['limb_i']],limb_b=LIMBS[p['limb_j']],
                hold_a=str(data['labels'][f,p['limb_i']]),hold_b=str(data['labels'][f,p['limb_j']]),**p))
        excess=max([r['excess_cm'] for r in pairs],default=0.)
        r=old.iloc[int(f)];causes=[]
        if excess>.2:causes.append('fixed_contact_span_impossible_even_in_3D_at_0.1cm_tolerance')
        if r.hold_extrapolation:causes.append('hold_outside_checked_grid')
        if r.low_contact_confidence:causes.append('contact_label_needs_review')
        if r.low_pose_visibility:causes.append('source_pose_has_low_visibility')
        if r.contact_residual and excess<=.2:causes.append('solver_or_geometry_or_evidence_not_identified')
        rows.append(dict(frame=int(f),old_contact_gap_cm=float(r.max_contact_error_cm),
            conservative_span_excess_cm=excess,unavoidable_max_gap_lower_bound_cm=excess/2,
            exceeds_contact_tolerance_in_any_dimension=excess>.2,causes=';'.join(causes),
            diagnosis_status='Conditional on fixed hold IDs, centimeter grid and body ratios; not human impossibility.'))
    pd.DataFrame(rows).to_csv(out/'failure_review.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(out/'pair_span_certificates.csv',index=False)
    return rows


def run_case(data,frames,tilt,depth,height,out,name,max_iterations=1200,initial=None,contact_weight=20.,depth_weight=1.,force_weight=25.,progress=print,ratios=None,support_weight=300.,reuse=None,reused_from=None):
    local=subset(data,frames);case=out/name
    if case.exists() and any(case.iterdir()):raise FileExistsError('Previous case is preserved; select a new output run.')
    case.mkdir(parents=True,exist_ok=True)
    if progress:progress(f'{name}: {len(frames)} frames, {tilt:g} deg, depth prior {depth:g} cm',flush=True)
    if reuse is not None:
        if force_weight!=0 or support_weight!=0:raise ValueError('Only angle-invariant geometric poses may be reused.')
        if len(reuse['values'])!=len(frames):raise ValueError('Shared geometric trajectory frame mismatch.')
        result=reuse
    else:
        problem=SpatialProblem(local,tilt,depth,height,contact_weight,depth_weight,force_weight,ratios=ratios,support_weight=support_weight)
        result=problem.solve(initial=initial,max_iterations=max_iterations)
    table,gaps=audit(result,local,tilt,height,ratios)
    np.savez_compressed(case/'trajectory.npz',frames=local['frames'],values=result['values'],joints_cm=result['joints'],
                        center_of_mass_cm=result['com'],head_cm=result['head'],contact_gaps_cm=gaps,
                        targets_cm=local['targets'],K=local['K'],R=local['R'],t_cm=local['t'],basis_helper=result['basis_helper'])
    pd.DataFrame(table).to_csv(case/'metrics.csv',index=False)
    solver=json.loads(json.dumps(result['solver']));history=solver.pop('history',[])
    if 'initial_direct_solver' in solver:solver['initial_direct_solver'].pop('history',None)
    pd.DataFrame(history).to_csv(case/'solver_history.csv',index=False)
    if reused_from:solver['shared_pose_solve_seconds']=solver['seconds'];solver['seconds']=0.
    info=dict(name=name,tilt_deg=float(tilt),depth_prior_cm=float(depth),height_cm=float(height),
              first_frame=int(frames[0]),last_frame=int(frames[-1]),frames=len(frames),
              contact_weight=contact_weight,depth_weight=depth_weight,force_weight=force_weight,support_weight=support_weight,solver=solver,
              code_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['scene3d.py','ik3d.py','trial3_3d_pipeline.py']},
              kinematics_reused_from=reused_from,
              physical_validation=False,geometry='Single plane with unmeasured intrinsics, tilt and body depth.')
    write_json(case/'summary.json',info)
    if progress:progress(f"  converged={info['solver']['success']} gradient={info['solver']['optimality']:.3g} time={info['solver']['seconds']:.1f}s gap={max(r['max_contact_gap_cm'] for r in table):.3f}cm",flush=True)
    return info,result,table


def identity(project,config):
    project=Path(project)
    files=input_provenance(project)
    for name in ['scene3d.py','ik3d.py','scene3d_render.py','trial3_3d_pipeline.py','support_refinement.py']:
        files['beta_translator/'+name]=file_sha256(project/'beta_translator'/name)
    for name in ['summary.json','trajectory.npz','frame_diagnostics.csv']:
        files[BASELINE+'/'+name]=file_sha256(project/BASELINE/name)
    return dict(config=config,inputs_sha256=files)


def verified_experiment(directory,expected_identity=None):
    directory=Path(directory);p=directory/'summary.json'
    if not p.exists():raise ValueError('Experiment is incomplete; select a new output run.')
    s=json.loads(p.read_text())
    if expected_identity is not None and s['provenance']!=expected_identity:raise ValueError('Experiment inputs or configuration changed.')
    for relative,digest in s['output_sha256'].items():
        if not (directory/relative).is_file() or file_sha256(directory/relative)!=digest:
            raise ValueError('Modified or missing experiment artifact: '+relative)
    return {**s,'output_dir':str(directory)}


def run_experiment(project,height_cm=180,ratios=None,angles=(0,15,30),depths=(10,25,40),
                   keyframes=None,clips=None,max_iterations=1200,progress=print,solver_mode='geometry'):
    project=Path(project).resolve();ratios=dict(RATIOS if ratios is None else ratios)
    make_body(height_cm,ratios)
    if solver_mode not in ('geometry','support'):raise ValueError('Choose geometry with independent force audits, or experimental support optimization.')
    keyframes=KEYFRAMES if keyframes is None else list(keyframes)
    clips=CLIPS if clips is None else dict(clips)
    if not keyframes and not clips:raise ValueError('Select at least one keyframe or clip.')
    angles=list(map(float,angles));depths=list(map(float,depths))
    if not angles or not depths or not np.isfinite(angles+depths).all() or min(angles)<-30 or max(angles)>60 or min(depths)<3 or max(depths)>80:
        raise ValueError('Use nonempty finite angles [-30,60] and depths [3,80].')
    if len(set(angles))!=len(angles) or len(set(depths))!=len(depths):raise ValueError('Duplicate scenarios.')
    if any(type(f) is not int or not 0<=f<=502 for f in keyframes):raise ValueError('Keyframes must be native frame integers 0..502.')
    for name,limits in clips.items():
        if not str(name).replace('_','').isalnum() or len(limits)!=2 or any(type(f) is not int for f in limits) or not 0<=limits[0]<=limits[1]<=502:
            raise ValueError('Clip name and native frame range are invalid.')
    if type(max_iterations) is not int or max_iterations<1:raise ValueError('Positive integer iteration limit required.')
    config=dict(height_cm=float(height_cm),ratios=ratios,angles_deg=angles,hip_depth_priors_cm=depths,
        keyframes=keyframes,clips={k:list(v) for k,v in clips.items()},max_iterations=max_iterations,solver_mode=solver_mode,
        contact_tolerance_cm=.1,depth_mode='soft hip-depth prior; actual solved depth reported',
        foot_normal='assumed wall outward normal',foot_friction=.8,force_component_limit_bw=1.,
        contact_force_model='hands pull/push; feet compression plus inscribed friction pyramid',
        prototype='articulated fixed-length 3D skeleton + planar metric wall; no measured 3D scan')
    provenance=identity(project,config)
    digest=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:12]
    out=project/f'outputs/trial3/spatial_experiments/target_{height_cm:g}cm_{digest}'
    if out.exists():
        cached=verified_experiment(out,provenance)
        if progress:progress('Verified cached 3D experiment: '+out.name,flush=True)
        return cached
    out.mkdir(parents=True);write_json(out/'run_inputs.json',provenance)
    start=time.monotonic();data=load_inputs(project)
    write_json(out/'camera.json',data['camera'])
    pd.DataFrame({'frame':data['frames'],'camera_tracking_fit_rmse_px':data['camera_residual_px']}).to_csv(out/'camera_tracking.csv',index=False)
    review=failure_review(project,data,out,height_cm,ratios)
    # Directly demonstrate the gravity gauge ambiguity on observed grid geometry.
    from .scene3d import tilt_rotation,project as project_points
    hold_values=np.array(list(data['old']['geometry']['hold_positions_cm'].values()))
    wall_holds=np.c_[hold_values,np.zeros(len(hold_values))]
    R=np.asarray(data['camera']['R']);t=np.asarray(data['camera']['t_cm']);K=data['K']
    baseline=project_points(wall_holds,K,R,t)[0];invariance=[]
    for angle in angles:
        Q=tilt_rotation(angle);same=project_points(wall_holds@Q.T,K,R@Q.T,t)[0]
        invariance.append(dict(tilt_deg=angle,max_projected_pixel_change=float(np.linalg.norm(same-baseline,axis=1).max()),
            interpretation='Rotating wall and camera together changes relation to gravity, not wall image; gravity is unobserved.'))
    write_json(out/'tilt_identifiability.json',invariance)
    case_dirs=[];case_rows=[];frame_rows=[];key_dirs={};shared={}
    weights=dict(force_weight=0.,support_weight=0.) if solver_mode=='geometry' else dict(force_weight=25.,support_weight=300.)
    for angle in angles:
        for depth in depths:
            for frame in keyframes:
                name=f'key_{frame:04d}_tilt{angle:g}_depth{depth:g}'
                cached=shared.get((depth,frame)) if solver_mode=='geometry' else None
                info,result,table=run_case(data,[frame],angle,depth,height_cm,out,name,max_iterations=max_iterations,progress=progress,ratios=ratios,
                    reuse=cached[0] if cached else None,reused_from=cached[1] if cached else None,**weights)
                shared.setdefault((depth,frame),(result,name))
                key_dirs[frame,angle,depth]=out/name
                for row in table:frame_rows.append(dict(case=name,kind='keyframe',tilt_deg=angle,depth_prior_cm=depth,**row))
                case_rows.append(case_summary(info,table,'keyframe'))
            for clip,(lo,hi) in clips.items():
                name=f'{clip}_tilt{angle:g}_depth{depth:g}'
                cached=shared.get((depth,clip)) if solver_mode=='geometry' else None
                info,result,table=run_case(data,np.arange(lo,hi+1),angle,depth,height_cm,out,name,max_iterations=max_iterations,progress=progress,ratios=ratios,
                    reuse=cached[0] if cached else None,reused_from=cached[1] if cached else None,**weights)
                shared.setdefault((depth,clip),(result,name))
                case_dirs.append(out/name)
                for row in table:frame_rows.append(dict(case=name,kind='clip',tilt_deg=angle,depth_prior_cm=depth,**row))
                case_rows.append(case_summary(info,table,'clip'))
    pd.DataFrame(case_rows).to_csv(out/'scenario_metrics.csv',index=False)
    pd.DataFrame(frame_rows).to_csv(out/'all_frame_metrics.csv',index=False)
    from .scene3d_render import render_comparison,make_viewer,wall_vertices,export_glb,case_panel
    import cv2
    middle_depth=depths[len(depths)//2];middle_angle=angles[len(angles)//2];videos=[]
    for clip in clips:
        paths=[out/f'{clip}_tilt{a:g}_depth{middle_depth:g}' for a in angles]
        video=out/f'{clip}_angles.mp4';render_comparison(project,data,paths,video);videos.append(video.name)
        paths=[out/f'{clip}_tilt{middle_angle:g}_depth{d:g}' for d in depths]
        video=out/f'{clip}_depths.mp4';render_comparison(project,data,paths,video);videos.append(video.name)
    cap=cv2.VideoCapture(str(project/'videos/trial3.mp4'));models=[]
    for frame in keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
        if not ok:raise RuntimeError('Missing keyframe.')
        panels=[]
        for angle in angles:
            case=key_dirs[frame,angle,middle_depth];info=json.loads((case/'summary.json').read_text())
            with np.load(case/'trajectory.npz') as z:
                row=pd.read_csv(case/'metrics.csv').iloc[0];panels.append(case_panel(image,data,z,row,info,0))
                name=f'scene_f{frame:04d}_tilt{angle:g}.glb'
                export_glb(out/name,wall_vertices(data),z['joints_cm'][0],z['head_cm'][0],angle,
                    dict(frame=frame,height_cm=height_cm,hip_depth_prior_cm=middle_depth,status=str(row['status']),
                         hold_markers_wall_xy_cm=data['old']['geometry']['hold_positions_cm']))
                models.append(name)
        cv2.imwrite(str(out/f'keyframe_{frame:04d}_angles.jpg'),np.hstack(panels))
    cap.release();make_viewer(out,data,case_dirs or list(key_dirs.values()))
    if identity(project,config)!=provenance:raise RuntimeError('Inputs or code changed during experiment; publication is incomplete.')
    summary=dict(schema_version=1,provenance=provenance,elapsed_seconds=time.monotonic()-start,
        keyframes=keyframes,clips=config['clips'],scenario_count=len(angles)*len(depths),case_count=len(case_rows),
        frames_with_dimension_independent_span_conflict=sum(r['exceeds_contact_tolerance_in_any_dimension'] for r in review),
        videos=videos,video_playback_fps=10,source_fps=data['fps'],models=models,
        camera_tilt_identifiable=False,all_frames_geometry_uncertain=True,
        limitations=['Tilt, intrinsics, body depth and contact normals are unmeasured.',
          'Unknown source height; source pixel discrepancy is a guide, not retargeting accuracy.',
          'No true hold protrusion/shape, skin, joint torque limits, self-collision or inertia.',
          'Same contact schedule retained; missed contacts excluded from independent force audits.',
          'Pre-unload timing comes from automatic labels and remains uncertain.',
          'Sampled inner force-support polygons aid optimization; independent wrench LPs determine support.',
          'Depth is a soft prior. Actual optimized depth, unreachable targets and nonconvergence remain visible.',
          'Single-plane assumption applies only to the white wall; no floor or second wall reconstruction.'])
    summary['output_sha256']={str(p.relative_to(out)):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'summary.json',summary)
    return {**summary,'output_dir':str(out)}


def case_summary(info,rows,kind):
    return dict(case=info['name'],kind=kind,tilt_deg=info['tilt_deg'],depth_prior_cm=info['depth_prior_cm'],
        frames=len(rows),converged=info['solver']['success'],projected_gradient=info['solver']['optimality'],seconds=info['solver']['seconds'],
        contact_failed_frames=sum(r['max_contact_gap_cm']>.1 for r in rows),max_contact_gap_cm=max(r['max_contact_gap_cm'] for r in rows),
        conditional_balance_frames=sum(r['conditional_balance'] for r in rows),
        alternate_foot_normal_balance_frames=sum(r['conditional_balance_upward_footholds'] for r in rows),
        conditional_robust_frames=sum(r['conditional_robust_2cm'] for r in rows),
        eligible_balance_frames=sum(r['eligible_balance'] for r in rows),
        median_hip_depth_cm=float(np.median([r['hip_depth_cm'] for r in rows])),
        max_segment_error_cm=max(r['max_segment_error_cm'] for r in rows),
        min_wall_clearance_cm=min(r['min_wall_clearance_cm'] for r in rows),
        max_source_guide_rmse_px=max(r['source_guide_rmse_px'] for r in rows))
