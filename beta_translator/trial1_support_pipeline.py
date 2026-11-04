"""Trial1 full-video support optimization and synchronized failure review."""
from pathlib import Path
import hashlib,json,html
import numpy as np

from . import trial1_sequence
from .trial1_sequence import load_inputs,TRIAL1_SCALE
from .gravity_sequence import GravityProblem,solve_gravity_sequence
from .gravity_sequence_pipeline import _verified_cached,render_gravity_sequence
from .support_optimization import optimize_support_sequence,optimization_settings
from .support_optimization_pipeline import publish,diagnose
from .support_audit import audit_settings
from .inverse_kinematics import RATIOS
from .full_sequence_pipeline import _intervals
from .project_io import file_sha256


def provenance(project):
    return {**trial1_sequence.input_provenance(project),'beta_translator/trial1_support_pipeline.py':file_sha256(__file__)}


def trial1_diagnose(result,reference_outside,audit_config):
    d=diagnose(result,reference_outside,audit_config);n=len(result['frames'])
    disagreement=np.asarray(result['trial1_context']['reference_disagreement'],bool)
    if disagreement.shape!=(n,):raise ValueError('Invalid reviewed-reference alignment.')
    d['flags'].update(uniform_scale_approximation=np.ones(n,bool),image_up_gravity_assumed=np.ones(n,bool),
                      reviewed_contact_disagreement=disagreement)
    d['issues']=[';'.join(k for k,v in d['flags'].items() if v[t]) for t in range(n)]
    d['counts']={k:int(v.sum()) for k,v in d['flags'].items()}
    d['status']=np.where(d['status']=='constraint_warning','constraint_warning','uncertain')
    return d


def review_html(inputs,result,d,baseline_name,optimized_name):
    """Self-contained controls; videos are portable file references, never embedded."""
    s=result['schedule'];review=inputs['reference_comparison'];human={}
    for row in review.itertuples():human.setdefault(int(row.frame),{})[row.limb]=row.reviewed_contact
    limb_names=['left_hand','right_hand','left_foot','right_foot'];frames=[]
    for t,frame in enumerate(result['frames']):
        frames.append(dict(frame=int(frame),status=str(d['status'][t]),issues=d['issues'][t],
            balance=d['support_audit']['frames'][t]['balance_audit_status'],
            unload=d['support_audit']['frames'][t]['unload_audit_status'],
            limbs=[dict(limb=l,label=str(s['labels'][t,i]),confidence=float(s['confidence'][t,i]),
                gap=float(d['contact_errors_cm'][t,i]) if s['known'][t,i] else None,
                validity=str(d['support_audit']['contact_state'][t,i]),reviewed=human.get(t,{}).get(l)) for i,l in enumerate(limb_names)]))
    priorities={'contact_residual','support_balance_failed','preunload_failed','departing_load_remaining','reviewed_contact_disagreement'}
    intervals=[row for row in _intervals(d,inputs['fps']) if row['issue'] in priorities]
    buttons=''.join(f'<button data-frame="{r["start_frame"]}">{html.escape(r["issue"].replace("_"," "))}: {r["start_frame"]}–{r["end_frame"]}</button>' for r in intervals)
    payload=json.dumps(frames,allow_nan=False).replace('</','<\\/')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trial1 — full-sequence review</title><style>
body{font:15px system-ui,sans-serif;background:#f4f5f7;color:#17212c;margin:0}main{max-width:1400px;margin:auto;padding:24px}
h1{margin:0 0 10px}p{line-height:1.5}.note{padding:14px;background:#fff1cc;border-radius:8px}.videos{display:grid;grid-template-columns:1fr 1.3fr;gap:18px}
video{width:100%;height:68vh;object-fit:contain;background:#111;border-radius:8px}.card{background:white;padding:16px;border-radius:10px;margin:16px 0}
.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap}input[type=range]{flex:1;min-width:220px}button,select{padding:8px 12px;border:1px solid #bbc4cd;border-radius:6px;background:#fff;cursor:pointer}
#intervals button{margin:4px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid #ddd}.bad{color:#b52121;font-weight:600}.muted{color:#66727f}
@media(max-width:800px){.videos{grid-template-columns:1fr}video{height:50vh}main{padding:12px}.card{overflow:auto}}
</style><main><h1>Trial1 · 180 cm · full-sequence review</h1>
<p class="note">Uniform image scale (46 cm reference) and image-up gravity are assumptions. This is a 2D quasi-static diagnostic. Contacts are unreviewed predictions except the separately shown saved H44 → H28 reference. Green model checks do not establish physical success.</p>
<div class="card controls"><button id="play">Play / pause</button><button id="prev">−1 frame</button><input id="frame" type="range" min="0" max="416" value="0"><button id="next">+1 frame</button><strong id="time"></strong>
<label>Result <select id="variant"><option value="optimized">Optimized</option><option value="baseline">Gravity baseline</option></select></label></div>
<div class="videos"><div><h2>Original source video</h2><video id="source" src="../../../../videos/trial1.mp4" muted playsinline preload="metadata"></video></div>
<div><h2 id="resultTitle">Optimized target body</h2><video id="result" src="''' + html.escape(optimized_name,quote=True) + '''" muted playsinline preload="metadata"></video></div></div>
<div class="card"><h2>Optimized diagnostics at this source frame</h2><p id="state"></p><table><thead><tr><th>Limb</th><th>Prediction</th><th>Confidence</th><th>Gap (cm)</th><th>Contact check</th><th>Saved reviewed contact</th></tr></thead><tbody id="limbs"></tbody></table><p id="issues"></p></div>
<div class="card"><h2>Jump to failure / disagreement intervals</h2><div id="intervals">''' + buttons + '''</div><p class="muted">The table always describes the optimized run. Select baseline to compare the same source frame visually. This page does not edit or confirm contact annotations.</p></div>
<p><a href="comparison.csv">Before/after metrics</a> · <a href="frame_diagnostics.csv">All frame checks</a> · <a href="reviewed_reference_comparison.csv">Saved reference comparison</a></p>
</main><script>
const rows=''' + payload + ''', fps=30;
const files={optimized:''' + json.dumps(optimized_name) + ''',baseline:''' + json.dumps(baseline_name) + '''};
const source=document.getElementById('source'), result=document.getElementById('result'), slider=document.getElementById('frame');
let selected=0;
function paint(t){selected=Math.max(0,Math.min(rows.length-1,Math.round(t)));const r=rows[selected];slider.value=selected;
 document.getElementById('time').textContent=`Frame ${selected} / ${rows.length-1} · ${(selected/fps).toFixed(2)} s`;
 document.getElementById('state').textContent=`${r.status} | support: ${r.balance} | unloading: ${r.unload}`;
 const body=document.getElementById('limbs');body.replaceChildren();
 for(const l of r.limbs){const tr=document.createElement('tr');const values=[l.limb,l.label,l.confidence.toFixed(2),l.gap===null?'N/A':l.gap.toFixed(3),l.validity,l.reviewed??'Not reviewed'];
 values.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if((i===3&&l.gap>.1)||(i===5&&l.reviewed&&l.reviewed!==l.label))td.className='bad';tr.append(td)});body.append(tr)}
 document.getElementById('issues').textContent=r.issues.replaceAll('_',' ').replaceAll(';',' · ');
}
function seek(t){source.pause();result.pause();paint(t);source.currentTime=selected/fps;result.currentTime=selected/fps;}
slider.addEventListener('input',()=>seek(Number(slider.value)));
document.getElementById('prev').onclick=()=>seek(selected-1);document.getElementById('next').onclick=()=>seek(selected+1);
document.getElementById('play').onclick=()=>{if(source.paused){source.play();result.play()}else{source.pause();result.pause()}};
source.addEventListener('timeupdate',()=>{paint(Math.floor(source.currentTime*fps+1e-4));if(Math.abs(result.currentTime-source.currentTime)>.08)result.currentTime=source.currentTime});
source.addEventListener('ended',()=>{result.pause();paint(rows.length-1)});
document.getElementById('variant').onchange=e=>{source.pause();result.pause();result.src=files[e.target.value];result.onloadedmetadata=()=>{result.currentTime=selected/fps};document.getElementById('resultTitle').textContent=e.target.value==='baseline'?'Gravity baseline':'Optimized target body'};
document.querySelectorAll('[data-frame]').forEach(b=>b.onclick=()=>seek(Number(b.dataset.frame)));
paint(0);
</script></html>'''


def extra_artifacts(stage,inputs,initial,result,before,after):
    baseline_dir=stage/'baseline-render';baseline_dir.mkdir()
    baseline_files=render_gravity_sequence({**inputs,'trial_label':'Trial1 baseline'},initial,before,baseline_dir)
    baseline_video=stage/f'trial1_gravity_{result["body"]["height_cm"]:g}cm_baseline.mp4'
    baseline_files[0].rename(baseline_video)
    baseline_trajectory=stage/'baseline_trajectory.npz'
    np.savez_compressed(baseline_trajectory,poses=initial['poses'],joints_cm=initial['joints'],
        contact_forces_n=initial['contact_forces_n'],center_of_mass_cm=initial['center_of_mass_cm'])
    review=inputs['reference_comparison'];path=stage/'reviewed_reference_comparison.csv';review.to_csv(path,index=False)
    review_page=stage/'review.html';optimized=f'trial1_support_optimized_{result["body"]["height_cm"]:g}cm_full_sequence.mp4'
    page=review_html(inputs,result,after,baseline_video.name,optimized).replace('Trial1 · 180 cm',f'Trial1 · {result["body"]["height_cm"]:g} cm')
    review_page.write_text(page)
    return dict(files=[baseline_video,baseline_trajectory,path,review_page],summary=dict(trial='Trial1',
        geometry_model=inputs['geometry'],baseline_video_name=baseline_video.name,review_page='review.html',
        reviewed_reference_frame_range=[107,129],reviewed_reference_limb_comparisons=len(review),
        reviewed_reference_disagreements=int((~review.agrees).sum()),
        reviewed_reference_disagreement_frames=int(review.loc[~review.agrees,'frame'].nunique()),
        deployed_automatic_eligible_moves=inputs['prediction']['eligible_moves'],
        geometry_limitations=['Uniform reference-image scale does not rectify perspective or recover depth.',
            'Image-up gravity is an assumption; no camera-roll or wall-inclination calibration.',
            'Eligible support means reached/confident/mapped under this approximation; no calibrated planar coverage claim.',
            'The saved 107–129 human reference is compared but never copied into or used to optimize the model schedule.',
            'Automatic movement acceptance remains disabled by the deployment gate; this is an experimental diagnostic.']))


def make_identity(project,inputs,height_cm=180.,ratios=None,mass_kg=70.,foot_friction=.8,max_force_n=None,
                  settings=None,audit_config=None,optimization=None):
    body=dict(height_cm=float(height_cm),ratios=dict(RATIOS if ratios is None else ratios))
    p=GravityProblem(inputs['reference'],inputs['visibility'],inputs['reference_endpoints'],inputs['schedule'],
        body,inputs['fps'],mass_kg,foot_friction,max_force_n,settings)
    return dict(inputs=provenance(project),trial='trial1',body=body,
        physics=dict(mass_kg=float(mass_kg),foot_friction=float(foot_friction),max_force_component_n=p.max_force_n),
        geometry=dict(reference_frame=0,px_per_cm=TRIAL1_SCALE,model='uniform_image_scale',gravity='image_up'),
        settings=p.config,support_audit_settings=audit_settings(audit_config),optimization_settings=optimization_settings(optimization))


def run_trial1_support_optimization(project,height_cm=180.,ratios=None,mass_kg=70.,foot_friction=.8,max_force_n=None,
        settings=None,audit_config=None,optimization=None,progress=print):
    project=Path(project).resolve();inputs=load_inputs(project)
    ident=make_identity(project,inputs,height_cm,ratios,mass_kg,foot_friction,max_force_n,settings,audit_config,optimization)
    root=project/'outputs/trial1/support_optimized'
    key=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:12]
    target=root/f'target_{height_cm:g}cm_{key}';cached=_verified_cached(target,ident)
    if cached:
        if progress:progress('Reused verified Trial1 support-optimized full sequence.')
        return cached
    if progress:progress('Verified 417 native Trial1 frames; uniform-scale and image-up gravity assumptions retained.')
    initial=solve_gravity_sequence(inputs['reference'],inputs['visibility'],inputs['reference_endpoints'],inputs['schedule'],
        ident['body'],inputs['fps'],mass_kg,foot_friction,max_force_n,ident['settings'],progress)
    initial['trial1_context']=inputs['trial1_context']
    result=optimize_support_sequence(initial,inputs['fps'],audit_config,optimization,inputs['reference_outside'],progress)
    return publish(project,inputs,initial,result,ident,result_root=root,progress=progress,
        input_provenance_fn=provenance,diagnostic_fn=trial1_diagnose,artifact_builder=extra_artifacts)
