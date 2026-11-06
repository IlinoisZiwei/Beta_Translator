"""Small shared display helpers; notebooks hold settings, not model definitions."""
from pathlib import Path
import json
import pandas as pd
from IPython.display import display,Markdown,Video,Image
from .body_pipeline import verified_run
from .body_render import render_run


def project_root():
    for p in (Path.cwd(),*Path.cwd().parents):
        if (p/'configs/trials').is_dir():return p
    raise FileNotFoundError('Open the notebook in the Beta_Translator repository')


def saved_result(project,trial,height_cm,index_file="verification_results.json"):
    p=Path(project);index=json.loads((p/'configs'/index_file).read_text())
    result=verified_run(p/index[trial])
    if result['height_cm']!=height_cm:raise ValueError('Saved height differs; enable RUN_VERIFICATION to fit this height')
    return result


def show_video(project,path):
    path=Path(path);relative=path.relative_to(project)
    display(Video(str(relative),embed=False,width=900))


def evidence(project,config):
    root=Path(project)/config['input_root']
    files=[f for d in ['state','tables'] for f in (root/d).iterdir() if f.is_file()]
    display(pd.DataFrame([dict(file=str(f.relative_to(project)),size_kb=round(f.stat().st_size/1024,1)) for f in sorted(files)]))
    images=sorted((root/'hold_mapping').glob('*background*.jpg'))
    if images:display(Image(filename=str(images[-1]),width=700))


def show_result(result):
    out=Path(result['output_dir']);name=result['selected_candidate']
    display(Markdown(f"**Selected:** {name}. **Status:** {result['selection_status']}. "
        f"**Full specification:** {result['specification_status']}. "
        f"Frames {result['first_frame']}–{result['last_frame']}; {result['height_cm']:g} cm."))
    display(pd.read_csv(out/'candidate_comparison.csv'))
    display(Markdown(result['ranking'][0]['selection_reason']))
    display(pd.read_csv(out/name/'constraints.csv'))
    m=pd.read_csv(out/name/'metrics.csv')
    cols=['frame','max_contact_gap_cm','conditional_balance','preunload','departing_force_failed','issues']
    display(m[[c for c in cols if c in m]].head(12))
    return render_run(out)


def history(project,trial):
    p=Path(project);root=p/('outputs' if trial=='trial1' else 'outputs/'+trial)
    names={'trial1':['refined_support','technique_plans','ground_support','support_optimized'],
           'trial2':['spatial_plans','anatomy_plans','stage1_contacts'],
           'trial3':['spatial_experiments','contact_region_plans','support_optimized']}[trial]
    # Trial1's later adaptations are under outputs/trial1, original evidence is under outputs.
    bases=[root,p/'outputs'/trial]
    files=sorted(set(f for b in bases for name in names for f in (b/name).rglob('*.mp4')))
    rows='\n'.join(f'- [{f.relative_to(p)}]({str(f.relative_to(p)).replace(" ","%20")})' for f in files)
    display(Markdown(rows or 'Historical artifacts remain in outputs; see the archived notebook for earlier experiment links.'))
