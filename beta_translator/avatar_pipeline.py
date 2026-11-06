"""Reusable Anny adapter for verified shared-body runs from any trial."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import pandas as pd
from .avatar_model import AnnyAvatar,group_frames,GROUPS,bone_group
from .body_pipeline import verified_run,write_json
from .project_io import file_sha256
from .trials import load_trial


def run_avatar(project, source_run, output_root=None, render=True, cache_dir=None, progress=print):
    """Preserve the saved solver trajectory; add a separately audited mesh/video."""
    project=Path(project).resolve();source_run=Path(source_run).resolve()
    source=verified_run(source_run);name=source['selected_candidate']
    settings=source['provenance']['config'];profile=settings.get('user_profile')
    if profile is None:raise ValueError('Use a saved personalized shared-body run, not a legacy stick trajectory')
    config=settings['trial_config']
    for path in [config['video'],config['input_root']+'/state/alignment_cache.npz']:
        expected=source['provenance']['input_sha256'].get(path)
        if expected is None or file_sha256(project/path)!=expected:
            raise ValueError('Source video/alignment no longer matches the saved motion: '+path)
    _,_,scene,_,_=load_trial(project,config,source['height_cm'])
    if scene.report()!=json.loads((source_run/'scene.json').read_text()):
        raise ValueError('Current scene differs from the saved run; preserve its trial config and camera inputs')
    with np.load(source_run/name/'trajectory.npz',allow_pickle=False) as z:fit={k:z[k] for k in z.files}
    n=len(fit['frames']);fps=source['fps']
    if not np.all(np.diff(fit['frames'])==1):raise ValueError('Avatar requires consecutive native frames')
    if fit['anatomical_q'].shape!=(n,32):raise ValueError('Use the anatomical arms AND legs shared model')
    avatar=AnnyAvatar(profile,cache_dir or os.environ.get('ANNY_CACHE_DIR',project/'.cache/anny'))
    identity={'source_summary_sha256':file_sha256(source_run/'summary.json'),
              'source_trajectory_sha256':file_sha256(source_run/name/'trajectory.npz'),
              'source_metrics_sha256':file_sha256(source_run/name/'metrics.csv'),
              'user_profile':profile,'anny_version':avatar.version,'render':bool(render),
              'adapter_sha256':{p.name:file_sha256(p) for p in Path(__file__).parent.glob('avatar_*.py')}}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    root=Path(output_root) if output_root else project/f"outputs/{source['trial']}/avatars"
    out=root/f"anny_{profile['height_cm']:g}cm_{int(fit['frames'][0]):04d}_{int(fit['frames'][-1]):04d}_{key}"
    if (out/'summary.json').exists():
        saved=json.loads((out/'summary.json').read_text())
        for f,h in saved['output_sha256'].items():
            if file_sha256(out/f)!=h:raise ValueError('Avatar output changed: '+f)
        return {**saved,'output_dir':str(out)}
    out.mkdir(parents=True,exist_ok=False);write_json(out/'provenance.json',identity)
    origins,rotations=group_frames(fit)
    np.savez_compressed(out/'avatar_bind.npz',vertices_cm=avatar.vertices,faces=avatar.faces,
                        weights=avatar.weights,group_names=GROUPS,origins_cm=avatar.bind_origins,rotations=avatar.bind_rotations,
                        anchor_faces=[x[0] for x in avatar.anchors],anchor_barycentric=[x[1] for x in avatar.anchors])
    np.savez_compressed(out/'avatar_motion.npz',frames=fit['frames'],origins_cm=origins,rotations=rotations,fps=fps)
    write_json(out/'profile_measurements.json',avatar.measurements)
    write_json(out/'surface_anchor_calibration.json',avatar.calibration)
    pd.DataFrame([{'anny_bone':b,'source_group':GROUPS[bone_group(b)],'finger_motion':'fixed template' if b.startswith(('finger','metacarpal')) else ''} for b in avatar.bone_labels]).to_csv(out/'bone_mapping.csv',index=False)
    solver=json.loads((source_run/name/'summary.json').read_text())['solver']
    original=pd.read_csv(source_run/name/'metrics.csv');records=[];surface_points=[]
    failures=[c for c in original if c.endswith('_failed') or c=='advanced_failure']
    for start in range(0,n,32):
        batch={k:v[start:start+32] for k,v in fit.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==n}
        V=avatar.pose(batch);surface=avatar.contact_points(V,batch['ground'])
        surface_points.append(surface)
        drift=np.linalg.norm(surface-batch['contact_points'],axis=-1)
        gap=np.linalg.norm(surface-batch['contacts'],axis=-1)
        clearance=scene.wall_clearance(V)
        for k in range(len(V)):
            t=start+k;active=batch['known'][k];row=original.iloc[t]
            records.append(dict(frame=int(fit['frames'][t]),mesh_patch_drift_cm=float(drift[k].max()),
                mesh_hold_gap_cm=float(np.max(np.where(active,gap[k],0))),
                source_hold_gap_cm=float(np.max(np.where(active,batch['contact_gaps_cm'][k],0))),
                mesh_wall_penetration_cm=float(max(0,-clearance[k].min())),
                mesh_contact_failed=bool(np.any(active&(gap[k]>.1))),
                mapping_failed=bool(drift[k].max()>.1),source_failure=bool(row[failures].fillna(False).any()),
                source_solver_converged=bool(solver['success']),
                support_failed=bool(not row.conditional_balance),
                finger_pose_uncertain=True,camera_wall_uncertain=True,
                source_issues=str(row.get('issues',''))))
    metrics=pd.DataFrame(records);metrics.to_csv(out/'surface_metrics.csv',index=False)
    np.savez_compressed(out/'contact_audit.npz',frames=fit['frames'],mesh_points_cm=np.concatenate(surface_points),
                        solver_points_cm=fit['contact_points'],targets_cm=fit['contacts'],known=fit['known'],ground=fit['ground'])
    # The adapter copies physical frames exactly, so all relative rotation steps
    # and twist continuity are inherited rather than re-inferred from endpoints.
    frame_error=float(np.max(abs(rotations@np.swapaxes(rotations,-1,-2)-np.eye(3))))
    mapping_pass=(not metrics.mapping_failed.any() and frame_error<1e-5
                  and abs(avatar.measurements['height_cm']-profile['height_cm'])<.1
                  and abs(avatar.measurements['arm_span_cm']-profile['arm_span_cm'])<.1)
    summary=dict(trial=source['trial'],source_run=f"outputs/{source['trial']}/shared_body/{source_run.name}",
        source_candidate=name,source_selection_status=source['selection_status'],
        source_solver_converged=bool(solver['success']),
        height_cm=profile['height_cm'],arm_span_cm=profile['arm_span_cm'],frame_count=n,
        first_frame=int(fit['frames'][0]),last_frame=int(fit['frames'][-1]),fps=fps,
        mapping_pass=bool(mapping_pass),max_patch_drift_cm=float(metrics.mesh_patch_drift_cm.max()),
        source_failed_frames=int(metrics.source_failure.sum()),mesh_contact_failed_frames=int(metrics.mesh_contact_failed.sum()),
        mesh_wall_penetration_frames=int((metrics.mesh_wall_penetration_cm>.5).sum()),
        max_wall_penetration_cm=float(metrics.mesh_wall_penetration_cm.max()),
        physical_validation=False,finger_grasp_validation=False,
        note='Mapping checks concern the display adapter only. Source failures remain. No mesh collision avoidance or person removal.',
        provenance=identity)
    if render:
        from .avatar_render import render as render_video
        render_video(out,avatar,fit,metrics,scene,scene.report(),config,project,fps,progress)
    # Detect any concurrent source modification before publishing a completed run.
    verified_run(source_run)
    if file_sha256(source_run/'summary.json')!=identity['source_summary_sha256']:raise ValueError('Source changed during avatar generation')
    summary['output_sha256']={str(f.relative_to(out)):file_sha256(f) for f in out.iterdir() if f.is_file()}
    write_json(out/'summary.json',summary)
    return {**summary,'output_dir':str(out)}


def show_avatar(project, output):
    """Display saved results without regenerating meshes, embedding video or solving."""
    from IPython.display import display,Markdown,Video
    from urllib.parse import quote
    project=Path(project);out=Path(output);s=json.loads((out/'summary.json').read_text())
    for f,h in s['output_sha256'].items():
        if file_sha256(out/f)!=h:raise ValueError('Avatar output changed: '+f)
    display(Markdown(f"**Anny display adapter:** {'pass' if s['mapping_pass'] else 'FAIL'}. "
        f"Frames {s['first_frame']}–{s['last_frame']}; height {s['height_cm']:g} cm; arm span {s['arm_span_cm']:g} cm. "
        f"**Source frames with constraint failures:** {s['source_failed_frames']}/{s['frame_count']}. "
        f"**Source solver converged:** {s['source_solver_converged']}. "
        "Fingers, wall/camera and foot orientation remain assumptions; this is not validated grasp or physical feasibility."))
    display(pd.DataFrame([dict(metric=k,value=s[k]) for k in
        ['max_patch_drift_cm','mesh_contact_failed_frames','mesh_wall_penetration_frames','max_wall_penetration_cm']]))
    display(pd.read_csv(out/'surface_metrics.csv').head(8))
    for file in ['avatar_comparison_3d.mp4','avatar_source_overlay.mp4']:
        if (out/file).exists():display(Video(url=quote((out/file).relative_to(project).as_posix()),embed=False,width=1000))

