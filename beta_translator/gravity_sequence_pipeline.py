"""Verified Trial3 inputs, configurable target body, and separate gravity outputs."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import tempfile
import textwrap
import uuid

import cv2
import numpy as np
import pandas as pd

from . import gravity_sequence, support_audit
from .support_audit import audit_settings, audit_support_sequence, attach_audit
from .full_sequence_pipeline import (
    load_full_sequence_inputs, input_provenance, _tables, _intervals, ISSUE_TEXT, COLORS,
)
from .gravity_sequence import DEFAULT_SETTINGS, solve_gravity_sequence, diagnose_gravity_sequence
from .inverse_kinematics import RATIOS, GRAVITY, make_body, make_mass_model
from .pose_adapter import project_plane
from .project_io import file_sha256, convert_video


GRAVITY_ISSUES = {**ISSUE_TEXT,
    'force_imbalance': 'Contact forces do not balance body weight.',
    'moment_imbalance': 'Residual moment exceeds numeric tolerance.',
    'foot_friction_violation': 'Simplified foot force constraint exceeded.',
    'no_known_support': 'No located hold support; no forces invented.',
    'force_at_missed_contact': 'Force applied at a limb that misses its hold.',
    'force_on_uncertain_contact': 'Force depends on uncertain/extrapolated contact.',
    'rapid_force_change': 'Large force transfer between adjacent frames.',
    'support_balance_failed': 'Located usable contacts cannot balance this pose.',
    'support_balance_uncertain': 'Balance needs uncertain contact assumptions.',
    'support_margin_low': 'Small reserve or COM perturbation fails.',
    'preunload_failed': 'Remaining contacts fail before predicted release.',
    'preunload_uncertain': 'Unloading depends on uncertain support/timing.',
    'preunload_timing_uncertain': 'Predicted release timing is ambiguous.',
    'preunload_margin_low': 'Small reserve after removing departing limbs.',
    'support_audit_not_evaluated': 'Support audit could not be evaluated.',
}


def gravity_provenance(project):
    return {**input_provenance(project),
        'beta_translator/support_audit.py': file_sha256(support_audit.__file__),
        'beta_translator/gravity_sequence.py': file_sha256(gravity_sequence.__file__),
        'beta_translator/gravity_sequence_pipeline.py': file_sha256(__file__)}


def run_identity(project, height_cm, ratios, mass_kg, foot_friction, max_force_n, settings, audit_config=None):
    body = dict(height_cm=float(height_cm),ratios=dict(RATIOS if ratios is None else ratios))
    make_body(**body)
    make_mass_model(mass_kg)
    if not np.isfinite(foot_friction) or foot_friction < 0:
        raise ValueError('Foot friction must be nonnegative and finite.')
    if max_force_n is not None and (not np.isfinite(max_force_n) or max_force_n <= 0):
        raise ValueError('Force component limit must be positive and finite.')
    config = {**DEFAULT_SETTINGS,**(settings or {})}
    if (set(config) != set(DEFAULT_SETTINGS) or any(not np.isfinite(v) or v <= 0 for v in config.values())
            or type(config['max_nfev']) is not int or config['outside_hull_weight'] > 1):
        raise ValueError('Invalid gravity-sequence settings.')
    physics = dict(mass_kg=float(mass_kg),gravity_m_s2=GRAVITY,foot_friction=float(foot_friction),
        max_force_component_n=float(mass_kg*GRAVITY if max_force_n is None else max_force_n))
    return dict(inputs=gravity_provenance(project),body=body,physics=physics,settings=config,
                support_audit_settings=audit_settings(audit_config))


def _destination(project, identity, result_root=None):
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    root = Path(result_root) if result_root is not None else Path(project)/'outputs/trial3/gravity_sequence'
    return root/f'target_{identity["body"]["height_cm"]:g}cm_{key}'


def _verified_cached(destination, identity):
    try:
        summary = json.loads((destination/'summary.json').read_text())
        if summary['provenance'] != identity:
            return None
        for name,digest in summary['output_sha256'].items():
            path = Path(name)
            if path.is_absolute() or len(path.parts)!=1 or file_sha256(destination/path)!=digest:
                return None
        return {**summary,'output_dir':str(destination),'video_file':str(destination/summary['video_name'])}
    except (OSError,ValueError,KeyError):
        return None


from .contact_geometry import contact_points


def render_gravity_sequence(inputs, result, diagnostic, destination):
    """Render all native frames with metric force arrows and explicit warnings."""
    destination = Path(destination)
    inverse = np.linalg.inv(np.asarray(inputs['geometry']['image_to_wall_cm']))
    pixels = project_plane(result['joints'].reshape(-1,2),inverse).reshape(-1,4,3,2)
    com_pixels = project_plane(result['center_of_mass_cm'],inverse)
    weight = result['mass_kg']*GRAVITY
    # A one-body-weight arrow is drawn as a 50 cm vector in calibrated wall coordinates.
    contacts = contact_points(result['joints'][:,:,2], result['schedule'])
    contact_pixels = project_plane(contacts.reshape(-1,2), inverse).reshape(-1,4,2)
    force_tip = contacts + result['contact_forces_n']/weight*50.
    force_pixels = project_plane(force_tip.reshape(-1,2),inverse).reshape(-1,4,2)
    gravity_pixels = project_plane(result['center_of_mass_cm']+[0.,-50.],inverse)
    background = inputs['background'].copy()
    h,w = background.shape[:2]
    all_pixels = np.concatenate([pixels.reshape(-1,2),com_pixels,force_pixels.reshape(-1,2),gravity_pixels])
    lo = np.minimum([0,0],np.floor(all_pixels.min(axis=0)-25)).astype(int)
    hi = np.maximum([w,h],np.ceil(all_pixels.max(axis=0)+25)).astype(int)
    if np.any(hi-lo > 8000):
        raise ValueError('Projection diverged; no meaningful wall view can be rendered.')
    for record in inputs['records']:
        hold = record['hold_id']
        if hold not in inputs['selected']:continue
        contours,_ = cv2.findContours(inputs['masks'][hold].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        color = COLORS['purple'] if hold not in inputs['geometry']['supported_hold_ids'] else (110,110,110)
        cv2.drawContours(background,contours,-1,color,1)
        cv2.putText(background,hold,(int(record['cx_px'])+8,int(record['cy_px'])-7),0,.5,color,1,cv2.LINE_AA)
    padded = np.full((hi[1]-lo[1],hi[0]-lo[0],3),235,np.uint8)
    padded[-lo[1]:h-lo[1],-lo[0]:w-lo[0]] = background
    panel_w,panel_h,side,header,footer = 660,1120,540,125,115
    width,height = panel_w+side,panel_h+header+footer
    display_scale = min(panel_w/padded.shape[1],panel_h/padded.shape[0])
    draw_w,draw_h = round(padded.shape[1]*display_scale),round(padded.shape[0]*display_scale)
    ox,oy = (panel_w-draw_w)//2,(panel_h-draw_h)//2
    raw = destination/'encoding.mp4'
    name = f'trial3_gravity_{result["body"]["height_cm"]:g}cm_full_sequence.mp4'
    video = destination/name
    writer = cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),inputs['fps'],(width,height))
    if not writer.isOpened():raise RuntimeError('Could not create gravity sequence video.')
    status_colors = dict(constraint_warning=COLORS['red'],uncertain=COLORS['amber'],within_numeric_tolerance=COLORS['green'])
    snapshots = []
    sample_indices = set(np.linspace(0,len(result['frames'])-1,7,dtype=int).tolist())
    sample_indices.add(min(314,len(result['frames'])-1))
    def put(canvas,text,x,y,scale=.51,color=None,bold=False):
        cv2.putText(canvas,text,(x,y),0,scale,color or COLORS['ink'],2 if bold else 1,cv2.LINE_AA)
    try:
        for t,frame in enumerate(result['frames']):
            wall = padded.copy()
            joints = np.rint(pixels[t]-lo).astype(int)
            s = result['schedule']
            cv2.polylines(wall,[joints[[0,1,3,2,0],0]],False,(70,125,105),4,cv2.LINE_AA)
            for i,limb in enumerate(('left_hand','right_hand','left_foot','right_foot')):
                if s['known'][t,i] and not s['supported'][t,i]:color=COLORS['purple']
                elif s['known'][t,i] and diagnostic['contact_errors_cm'][t,i]>result['settings']['contact_tolerance_cm']:color=COLORS['red']
                elif not s['known'][t,i] or s['low_confidence'][t,i]:color=COLORS['amber']
                else:color=COLORS['green']
                cv2.polylines(wall,[joints[i]],False,color,4,cv2.LINE_AA)
                for xy in joints[i]:cv2.circle(wall,tuple(xy),5,color,-1,cv2.LINE_AA)
                contact_xy = np.rint(contact_pixels[t,i]-lo).astype(int)
                if s.get('contact_type', np.full(s['known'].shape, 'hold'))[t,i] == 'ground':
                    cv2.line(wall,tuple(joints[i,2]),tuple(contact_xy),COLORS['purple'],3,cv2.LINE_AA)
                    cv2.line(wall,tuple(contact_xy+[-8,0]),tuple(contact_xy+[8,0]),COLORS['purple'],3,cv2.LINE_AA)
                if s['known'][t,i]:
                    target = project_plane(s['targets'][t,i][None],inverse)[0]-lo
                    cv2.drawMarker(wall,tuple(np.rint(target).astype(int)),color,cv2.MARKER_CROSS,15,2)
                    if diagnostic['contact_errors_cm'][t,i]>result['settings']['contact_tolerance_cm']:
                        for u in np.arange(0,1,.15):
                            a=contact_xy*(1-u)+target*u;b=contact_xy*(1-min(u+.07,1))+target*min(u+.07,1)
                            cv2.line(wall,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),color,2,cv2.LINE_AA)
                    cv2.arrowedLine(wall,tuple(contact_xy),tuple(np.rint(force_pixels[t,i]-lo).astype(int)),
                        ((20,105,235) if diagnostic['support_audit']['eligible'][t,i] else (145,145,145)),3,cv2.LINE_AA,tipLength=.22)
            com = tuple(np.rint(com_pixels[t]-lo).astype(int))
            cv2.circle(wall,com,9,(210,190,10),-1,cv2.LINE_AA)
            cv2.arrowedLine(wall,com,tuple(np.rint(gravity_pixels[t]-lo).astype(int)),
                (210,190,10),3,cv2.LINE_AA,tipLength=.15)
            canvas = np.full((height,width,3),250,np.uint8)
            canvas[header+oy:header+oy+draw_h,ox:ox+draw_w] = cv2.resize(wall,(draw_w,draw_h),interpolation=cv2.INTER_AREA)
            put(canvas,f'{inputs.get("trial_label","Trial3")} | {result["body"]["height_cm"]:g} cm | gravity IK + support audits',18,30,.78,bold=True)
            put(canvas,f'Frame {frame:03d} / {result["frames"][-1]} | {frame/inputs["fps"]:.2f} s | {inputs["fps"]:g} FPS | fixed segment lengths',18,59,.57)
            assumptions=inputs.get('render_assumptions','QUASI-STATIC 2D | assumed mass {mass:g} kg | not a dynamic/physical success test')
            put(canvas,assumptions.format(mass=result['mass_kg']),18,85,.52)
            put(canvas,diagnostic['status'][t].upper().replace('_',' '),18,110,.59,status_colors[diagnostic['status'][t]],True)
            x=panel_w+15;y=header+27
            audit=diagnostic['support_audit']; row=audit['frames'][t]
            def reserve(value):
                return 'N/A' if value is None else f'{100*value:+.1f}%'
            def status_line(label,value):
                readable = value.replace('_',' ')
                color = COLORS['red'] if value=='fails_located_support_model' else (COLORS['green'] if value=='passes_model_checks' else COLORS['amber'])
                return label+readable,color
            put(canvas,'REACHED-CONTACT SUPPORT AUDIT',x,y,.57,bold=True)
            y+=28
            text,color=status_line('',row['balance_audit_status']);put(canvas,text,x,y,.53,color,True)
            for line in [f'Eligible / conditional contacts: {row["eligible_support_count"]} / {row["conditional_support_count"]}',
                         f'Capacity reserve: {reserve(row["eligible_margin"])} | conditional: {reserve(row["conditional_margin"])}',
                         f'COM +/-{audit["settings"]["com_perturbation_cm"]:g} cm worst: {reserve(row["eligible_worst_margin"])} / {reserve(row["conditional_worst_margin"])}',
                         f'Max gap: {diagnostic["max_contact_error_cm"][t]:.3f} cm (limit 0.1)']:
                y+=24;put(canvas,line,x,y,.46)
            y+=32;put(canvas,'PRE-RELEASE UNLOADING',x,y,.57,bold=True)
            y+=26;text,color=status_line('',row['unload_audit_status']);put(canvas,text,x,y,.51,color,True)
            upcoming=[v for v in audit['unloading'] if v['frame']==int(frame)]
            if upcoming:
                for u in upcoming[:2]:
                    y+=22;put(canvas,f'{u["event_id"]}: remove {u["moving_limbs"].replace("left_","L ").replace("right_","R ")} -> frame {u["release_frame"]}',x,y,.43)
                    timing='TIMING UNCERTAIN' if u['timing_uncertain'] else 'predicted release'
                    y+=21;put(canvas,f'Reserve after: {reserve(u["eligible_margin"])} / {reserve(u["conditional_margin"])} | {timing}',x,y,.40,
                        COLORS['amber'] if u['timing_uncertain'] else None)
                if len(upcoming)>2:
                    y+=21;put(canvas,'Additional overlapping checks: see CSV',x,y,.43)
            else:
                y+=23;put(canvas,'No predicted release in the look-ahead window.',x,y,.44)
            y+=29;put(canvas,'CONTACT VALIDITY / CANDIDATE FORCES',x,y,.53,bold=True)
            for i,limb in enumerate(('Left hand','Right hand','Left foot','Right foot')):
                y+=25
                display = s.get('contact_type', np.full(s['known'].shape, 'hold'))[t,i]
                label = f'ground (raw {s["labels"][t,i]})' if display == 'ground' else str(s['labels'][t,i])
                put(canvas,f'{limb}: {label} | conf {s["confidence"][t,i]:.2f}',x,y,.44,bold=True)
                y+=21
                validity=str(audit['contact_state'][t,i]).replace('_',' ')
                gap=f'{diagnostic["contact_errors_cm"][t,i]:.2f} cm' if s['known'][t,i] else 'N/A'
                color=COLORS['red'] if validity=='invalid gap' else (COLORS['green'] if audit['eligible'][t,i] else COLORS['amber'])
                put(canvas,f'{validity} | gap {gap}',x,y,.43,color)
                y+=20;fx,fy=result['contact_forces_n'][t,i]
                put(canvas,f'Candidate Fx {fx:+.1f} N | Fy {fy:+.1f} N',x,y,.43)
            y+=29;put(canvas,'OTHER FLAGS (full details in CSV)',x,y,.51,bold=True)
            flags=[key for key,mask in diagnostic['flags'].items() if mask[t] and not key.startswith(('support_','preunload_'))]
            available=max(0,int((header+960-y)//21))
            shown=flags if len(flags)<=available else flags[:max(0,available-1)]
            for key in shown:
                y+=21;put(canvas,key.replace('_',' '),x,y,.44)
            if len(shown)<len(flags):
                y+=21;put(canvas,f'+ {len(flags)-len(shown)} more flags in frame_diagnostics.csv',x,y,.42)
            y=header+994
            for line,color in [('Orange: candidate forces on eligible contacts',(20,105,235)),
                               ('Gray arrows: excluded / conditional candidate forces',(120,120,120)),
                               ('Cyan: COM / gravity; 1 body weight = 50 cm',(190,155,0)),
                               ('Audited forces are stored separately in CSV.',COLORS['ink'])]:
                put(canvas,line,x,y,.43,color);y+=23
            base=header+panel_h
            for j,status in enumerate(diagnostic['status']):
                a,b=round(j*width/len(pixels)),round((j+1)*width/len(pixels))
                cv2.rectangle(canvas,(a,base+4),(b,base+19),status_colors[status],-1)
            cursor=round(t*(width-1)/(len(pixels)-1));cv2.line(canvas,(cursor,base+1),(cursor,base+24),(0,0,0),3)
            motion_note='One global fit + independent support audits. Red = model failure; amber = uncertainty / low reserve.'
            if inputs.get('source_video'):
                motion_note=f'Background = fixed reference frame {inputs["geometry"]["reference_frame_number"]} (including the person). Review page shows the moving source video.'
            put(canvas,motion_note,16,base+48,.5)
            put(canvas,inputs.get('render_contact_legend','Eligible = reached, high-confidence, in-grid contact. Conditional adds uncertain reached contacts.'),16,base+73,.5)
            put(canvas,'Unloading holds the pose fixed and removes departing limbs. No dynamics or real-world success claim.',16,base+98,.5)
            writer.write(canvas)
            if t in sample_indices:
                image=destination/f'frame_{int(frame):04d}.png'
                if not cv2.imwrite(str(image),canvas):raise RuntimeError('Could not save key frame.')
                snapshots.append(cv2.resize(canvas,(360,408)))
                if t==min(314,len(pixels)-1):cv2.imwrite(str(destination/'poster.png'),canvas)
    finally:
        writer.release()
    convert_video(raw,video)
    capture=cv2.VideoCapture(str(video));count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=capture.get(cv2.CAP_PROP_FPS);capture.release()
    if count!=len(result['frames']) or not np.isclose(fps,inputs['fps']):
        raise RuntimeError('Encoded gravity sequence lost source frames or timing.')
    # Two rows keep the storyboard readable without a very wide contact sheet.
    while len(snapshots)<8:snapshots.append(np.full_like(snapshots[0],250))
    cv2.imwrite(str(destination/'storyboard.png'),np.vstack([np.hstack(snapshots[:4]),np.hstack(snapshots[4:8])]))
    return [video,destination/'poster.png',destination/'storyboard.png',*sorted(destination.glob('frame_*.png'))]


def publish_gravity_run(project, inputs, result, identity, result_root=None, progress=print):
    """Publish verified finite results, retaining every previous run directory."""
    destination=_destination(project,identity,result_root)
    if destination.exists():destination=destination.with_name(destination.name+'_'+uuid.uuid4().hex[:8])
    d=diagnose_gravity_sequence(result,inputs['reference_outside'])
    if progress:progress('Auditing reached contacts, force reserve and pre-release unloading.')
    audit=audit_support_sequence(result,d,identity['support_audit_settings'])
    attach_audit(d,audit)
    frames,limbs=_tables(result,d,inputs['fps'])
    frames=frames.merge(pd.DataFrame(audit['frames']),on='frame',validate='one_to_one')
    frames['legacy_numerical_checks_pass']=d['legacy_numerical_checks_pass']
    frames['net_force_n']=d['net_force_norm_n'];frames['net_moment_nm']=d['net_moment_nm']
    frames['balanced']=d['balanced'];frames['numerical_checks_pass']=d['numerical_checks_pass']
    frames['max_foot_friction_excess_n']=d['foot_friction_excess_n'].max(axis=1)
    frames['force_step_bw']=d['force_step_bw']
    frames['com_x_cm']=result['center_of_mass_cm'][:,0];frames['com_y_cm']=result['center_of_mass_cm'][:,1]
    limbs['force_x_n']=result['contact_forces_n'][:,:,0].ravel()
    limbs['force_y_n']=result['contact_forces_n'][:,:,1].ravel()
    limbs['contact_validity']=audit['contact_state'].ravel()
    limbs['eligible_support']=audit['eligible'].ravel()
    limbs['conditional_support']=audit['conditional'].ravel()
    for group in ('eligible','conditional'):
        for axis,i in (('x',0),('y',1)):
            limbs[f'{group}_audit_force_{axis}_n']=audit['forces_n'][group][:,:,i].ravel()
    with tempfile.TemporaryDirectory(prefix='beta-gravity-export-') as temp:
        stage=Path(temp)
        frames.to_csv(stage/'frame_diagnostics.csv',index=False)
        limbs.to_csv(stage/'limb_diagnostics.csv',index=False)
        pd.DataFrame(audit['events'],columns=['event_id','release_index','release_frame','moving_indices',
            'moving_limbs','from_labels','to_labels','timing_uncertain','timing_reason']).to_csv(stage/'release_events.csv',index=False)
        unload=pd.DataFrame(audit['unloading'])
        if unload.empty:unload=pd.DataFrame(columns=['event_id','frame','release_frame','status','support_status'])
        unload.to_csv(stage/'preunload_checks.csv',index=False)
        np.savez_compressed(stage/'support_checks.npz',eligible=audit['eligible'],conditional=audit['conditional'],
            contact_validity=audit['contact_state'],eligible_forces_n=audit['forces_n']['eligible'],
            conditional_forces_n=audit['forces_n']['conditional'])
        pd.DataFrame(_intervals(d,inputs['fps'])).to_csv(stage/'issue_intervals.csv',index=False)
        np.savez_compressed(stage/'trajectory.npz',frames=result['frames'],poses=result['poses'],
            joints_cm=result['joints'],reference_poses=result['reference'],visibility=result['visibility'],
            reference_endpoints_cm=result['reference_endpoints'],decoded_labels=result['schedule']['labels'],
            targets_cm=result['schedule']['targets'],active=result['schedule']['known'],
            in_coverage=result['schedule']['supported'],confidence=result['schedule']['confidence'],
            contact_errors_cm=d['contact_errors_cm'],contact_forces_n=result['contact_forces_n'],
            center_of_mass_cm=result['center_of_mass_cm'],net_force_n=d['net_force_n'],
            net_moment_nm=d['net_moment_nm'],balanced=d['balanced'],numerical_checks_pass=d['numerical_checks_pass'])
        if progress:progress('Rendering every source frame with contact, force and uncertainty diagnostics.')
        files=render_gravity_sequence(inputs,result,d,stage)
        video=files[0]
        files.extend(stage/name for name in ('frame_diagnostics.csv','limb_diagnostics.csv','issue_intervals.csv','trajectory.npz',
            'release_events.csv','preunload_checks.csv','support_checks.npz'))
        final_unload={row['event_id']:row for row in audit['unloading']}
        summary=dict(schema_version=2,provenance=identity,body=result['body'],physics=identity['physics'],
            support_audit_settings=audit['settings'],
            contact_validity_counts=dict(Counter(audit['contact_state'].ravel().tolist())),
            balance_audit_counts=dict(Counter(row['balance_audit_status'] for row in audit['frames'])),
            preunload_frame_counts=dict(Counter(row['unload_audit_status'] for row in audit['frames'])),
            preunload_check_counts=dict(Counter(row['status'] for row in audit['unloading'])),
            release_event_count=len(audit['events']),
            release_timing_uncertain_count=sum(e['timing_uncertain'] for e in audit['events']),
            last_pre_release_check_counts=dict(Counter(row['status'] for row in final_unload.values())),
            legacy_numerical_pass_frame_count=int(d['legacy_numerical_checks_pass'].sum()),
            settings=result['settings'],method=result['method'],optimizer=result['optimizer'],
            video_name=video.name,frame_count=len(result['frames']),first_frame=int(result['frames'][0]),
            last_frame=int(result['frames'][-1]),fps=inputs['fps'],duration_seconds=len(result['frames'])/inputs['fps'],
            balanced_frame_count=int(d['balanced'].sum()),numerical_pass_frame_count=int(d['numerical_checks_pass'].sum()),
            max_contact_error_cm=float(d['max_contact_error_cm'].max()),
            max_force_residual_n=float(d['net_force_norm_n'].max()),max_moment_residual_nm=float(abs(d['net_moment_nm']).max()),
            max_segment_length_error_cm=d['max_segment_length_error_cm'],
            max_angle_step_deg=float(d['angle_step_deg'].max()),max_hip_step_cm=float(d['hip_step_cm'].max()),
            flag_counts=d['counts'],status_counts=dict(Counter(map(str,d['status']))),
            source_climber_height_cm=None,reference_frame=inputs['geometry']['reference_frame_number'],
            candidate_run_id=inputs['geometry']['candidate_run_id'],
            model_threshold_coverage=inputs['prediction']['accepted_frame_fraction'],
            interpretation='Continuous quasi-static diagnostic candidate. Failed contacts, forces and uncertain support remain flagged; not a physically validated climb.',
            limitations=[
                'Support audits diagnose the jointly optimized trajectory; they do not replan or repair failed poses.',
                'Eligible contacts meet gap/confidence/calibration filters, not human-confirmed grasp validity.',
                'Audit forces act at reached hold centres, separate from the original candidate endpoint forces.',
                'Reserve = 1 - minimum peak force-component utilization; assumed capacity, not a safety probability.',
                'COM +/- horizontal perturbation is a chosen sensitivity test, not measured pose uncertainty.',
                'Preunload removes all simultaneous departing limbs at each fixed pre-release pose; no future supports.',
                'Unloading is counterfactual support readiness, not simulated or enforced dynamic load transfer.',
                'Contact labels/timing are unchanged unreviewed predictions; confidence is not accuracy.',
                'The target mass is assumed unless supplied by the user; source mass and height are unknown.',
                'Only located decoded hold contacts receive forces. Ground/wall/air support of free limbs is unresolved.',
                'H053 remains flagged extrapolation; the white-wall calibration was not extended.',
                'Forces act at fitted limb endpoints; force on a missed target is explicitly flagged and is not valid support.',
                'Hands allow arbitrary in-plane forces within component limits; feet use fy>=0 and abs(fx)<=mu*fy.',
                'No measured hold normals, grip/strength limits, joint torques, collision, or body/hold depth.',
                'Per-frame static equilibrium omits inertial forces and out-of-plane dynamics; rapid changes are flagged.',
                'All frames and forces are optimized jointly; temporal smoothing does not establish dynamic feasibility.',
                'Default limits scale with body weight, so mass alone does not change poses.',
                'Fixed lengths are retained even where contacts cannot be met. Finite unconverged output is diagnostic only.',
                'Existing geometric and single-move results are preserved. No alternative-beta planner or success probability.'],
            output_sha256={p.name:file_sha256(p) for p in files})
        (stage/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        files.append(stage/'summary.json')
        if gravity_provenance(project)!=identity['inputs']:
            raise RuntimeError('Inputs or code changed during fitting; no outputs installed.')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.mkdir(exist_ok=False)
        for path in files:(destination/path.name).write_bytes(path.read_bytes())
    if progress:progress(f'Gravity sequence saved: {destination}')
    return {**summary,'output_dir':str(destination),'video_file':str(destination/video.name)}


def run_trial3_gravity_sequence(project, height_cm=180., ratios=None, mass_kg=70.,
                               foot_friction=.8, max_force_n=None, settings=None,
                               progress=print, audit_config=None):
    project=Path(project).resolve()
    identity=run_identity(project,height_cm,ratios,mass_kg,foot_friction,max_force_n,settings,audit_config)
    destination=_destination(project,identity)
    cached=_verified_cached(destination,identity)
    if cached is not None:
        if progress:progress('Reused verified full-sequence gravity result.')
        return cached
    inputs=load_full_sequence_inputs(project)
    if progress:progress('Verified 503 native Trial3 frames, contacts and white-wall calibration.')
    result=solve_gravity_sequence(inputs['reference'],inputs['visibility'],inputs['reference_endpoints'],
        inputs['schedule'],identity['body'],inputs['fps'],mass_kg=mass_kg,foot_friction=foot_friction,
        max_force_n=max_force_n,settings=identity['settings'],progress=progress)
    return publish_gravity_run(project,inputs,result,identity,progress=progress)
