"""Saved evidence, interactive review and historical displays; no model fitting.

Reading a notebook never rewrites evidence. Review buttons export new files;
saved videos retain their original body, scene and contact provenance.
"""
from pathlib import Path
from datetime import datetime, timezone
import html
import json
import uuid
import cv2
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, Image, Video, Markdown
from beta_translator.project_io import (
    file_sha256, read_frame, load_reviewed_holds, validate_reference_alignment,
    validate_annotations, save_reviewed_holds,
)
from beta_translator.contact_review import load_review_bundle, review_annotation_from_prediction
from beta_translator.inverse_kinematics import LIMBS, RATIOS
from beta_translator.pose_adapter import LIMB_LANDMARKS


def link(project, path, label=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    relative = path.relative_to(project).as_posix()
    display(Markdown(f'[{label or path.name}]({relative.replace(" ", "%20")})'))


def video(project, path, width=850):
    path = Path(path)
    link(project, path)
    display(Video(url=path.relative_to(project).as_posix(), embed=False, width=width))


def picture(path, width=850):
    display(Image(filename=str(path), width=width))


def table(path, rows=16):
    path = Path(path)
    if path.suffix == '.csv':
        data = pd.read_csv(path)
        display(data.head(rows))
        if len(data) > rows:
            print(f'{len(data)} rows total; showing the first {rows}.')
    else:
        data = json.loads(path.read_text())
        display(pd.json_normalize(data if isinstance(data, list) else [data]))
    return data


class Evidence:
    """One trial's immutable saved input bundle plus an in-memory route draft."""
    def __init__(self, project, config):
        self.project = Path(project)
        self.trial = config['trial']
        self.root = self.project / config['input_root']
        self.video = self.project / config['video']
        self.count, self.fps = config['frame_count'], config['fps']
        self.map_path = self.root / 'state/reviewed_hold_map.npz'
        self.alignment_path = self.root / 'state/alignment_cache.npz'
        self.records, self.masks, self.run_id, selected = load_reviewed_holds(self.map_path, self.video)
        self.saved_selected = list(selected)
        self.selected = list(selected)
        self.reference = validate_reference_alignment(self.map_path, self.alignment_path)
        self.input_hashes = {str(p.relative_to(self.project)): file_sha256(p)
                            for p in [self.video, self.map_path, self.alignment_path]}
        with np.load(self.map_path, allow_pickle=False) as z:
            self.wall = z['reference_frame'].copy()
        with np.load(self.alignment_path, allow_pickle=False) as z:
            self.transforms = {int(f): (m.copy(), str(s)) for f, m, s in zip(z['frames'], z['matrices'], z['statuses'])}
        self.poses = pd.read_csv(self.root / 'tables/pose_coordinates.csv').set_index('frame')
        self.prediction, self.contacts = load_review_bundle(self.video, self.root,
            self.root / 'automatic/wholebody.npz', self.project / 'models/contact_v1',
            self.root / 'automatic/predictions')
        self.guard = None
        if (self.root / 'state/identity_guard.npz').exists():
            with np.load(self.root / 'state/identity_guard.npz', allow_pickle=False) as z:
                self.guard = z['observed'].astype(bool)

    def check_current(self):
        for path, digest in self.input_hashes.items():
            if file_sha256(self.project / path) != digest:
                raise ValueError('Evidence changed on disk. Restart and reload the notebook: ' + path)

    def image(self, frame=None, ids=None, skeleton=False):
        image = self.wall.copy() if frame is None else read_frame(self.video, frame)
        ids = self.selected if ids is None else ids
        transform, status = (np.eye(3), 'ok') if frame is None else self.transforms[frame]
        if status == 'ok':
            for hold in ids:
                mask = self.masks[hold].astype(np.uint8)
                if frame is not None:
                    mask = cv2.warpPerspective(mask, transform, image.shape[1::-1], flags=cv2.INTER_NEAREST)
                contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contour, -1, (0, 220, 255), 2)
                yy, xx = np.nonzero(mask)
                if len(xx):
                    xy = (int(xx.mean()), int(yy.mean()))
                    cv2.putText(image, hold, xy, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
                    cv2.putText(image, hold, xy, cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        observed = frame in self.poses.index if frame is not None else False
        if observed and self.guard is not None:
            observed = bool(self.guard[frame])
        if skeleton and observed:
            row = self.poses.loc[frame]
            for names in LIMB_LANDMARKS.values():
                last = None
                for name in names:
                    xy = np.array([row[f'{name}_{a}_px'] for a in ('x', 'y')])
                    if not np.isfinite(xy).all() or row[f'{name}_visibility'] < .5:
                        last = None
                        continue
                    point = tuple(np.round(xy).astype(int))
                    cv2.circle(image, point, 4, (100, 240, 40), -1)
                    if last is not None:
                        cv2.line(image, last, point, (100, 240, 40), 3)
                    last = point
        if frame is not None:
            text = f'Frame {frame} | {frame/self.fps:.2f}s | observed pose: {observed}'
            cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
            cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        return image

    def show_image(self, **kwargs):
        ok, encoded = cv2.imencode('.jpg', self.image(**kwargs))
        if not ok:
            raise ValueError('Could not encode the evidence preview')
        display(Image(data=encoded.tobytes(), width=650))

    def export_route(self, directory=None):
        self.check_current()
        if not self.selected or len(set(self.selected)) != len(self.selected) or not set(self.selected) <= self.masks.keys():
            raise ValueError('Choose a nonempty set of candidate hold IDs.')
        directory = self.new_review_directory(directory)
        save_reviewed_holds(directory / 'reviewed_hold_map.npz', self.video, self.wall,
            self.records, self.masks, self.run_id, self.selected, reference_frame_number=self.reference)
        (directory / 'review.json').write_text(json.dumps(dict(trial=self.trial,
            source_hashes=self.input_hashes, selected_hold_ids=self.selected,
            role='route review export; saved predictions and fitted plans are unchanged'), indent=2) + '\n')
        return directory

    def new_review_directory(self, parent=None):
        parent = Path(parent) if parent is not None else self.project / 'outputs/notebook_reviews' / self.trial
        directory = parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
        directory.mkdir(parents=True, exist_ok=False)
        return directory


def route_picker(evidence):
    picker = widgets.SelectMultiple(options=[r['hold_id'] for r in evidence.records], value=tuple(evidence.selected),
        description='Route holds', rows=12, layout=widgets.Layout(width='280px'))
    preview, message = widgets.Output(), widgets.Output()
    save = widgets.Button(description='Export route review', layout=widgets.Layout(width='200px'))
    def update(change=None):
        evidence.selected = list(picker.value)
        with preview:
            preview.clear_output(wait=True)
            evidence.show_image()
            print(f'{len(evidence.selected)} selected / {len(evidence.records)} candidates retained')
    def export(_):
        with message:
            message.clear_output()
            try:
                path = evidence.export_route()
                link(evidence.project, path / 'reviewed_hold_map.npz')
                print('Exported a new review. Existing inference and fitted videos still use the saved route.')
            except ValueError as error:
                print(error)
    picker.observe(update, names='value');save.on_click(export)
    update()
    ui = widgets.VBox([widgets.HBox([picker, preview]), save, message])
    display(ui)
    return ui


def frame_viewer(evidence, initial=0):
    slider = widgets.IntSlider(value=initial, min=0, max=evidence.count-1, description='Frame', continuous_update=False)
    out = widgets.Output()
    def update(change=None):
        with out:
            out.clear_output(wait=True)
            evidence.show_image(frame=slider.value, skeleton=True)
            display(evidence.contacts.loc[evidence.contacts.frame == slider.value,
                ['limb', 'hold_id', 'confidence', 'status']])
    slider.observe(update, names='value');update()
    ui = widgets.VBox([slider, out]);display(ui)
    return ui


class ContactReview:
    """BEFORE/AFTER annotations with explicit draft and support-confirmation state."""
    def __init__(self, evidence, height_cm):
        self.evidence, self.height = evidence, height_cm
        self.route = list(evidence.selected)
        self.frames = dict(before=None, after=None)
        self.fields = {phase: {limb: widgets.Dropdown(options=['unknown', 'free', 'wall_contact'] + self.route,
            value='unknown', layout=widgets.Layout(width='150px')) for limb in LIMBS} for phase in self.frames}
        self.slider = widgets.IntSlider(min=0, max=evidence.count-1, description='Frame', continuous_update=False)
        self.moving = widgets.Dropdown(options=LIMBS, description='Moving limb')
        self.release = widgets.IntText(value=-1, description='Release')
        self.arrival = widgets.IntText(value=-1, description='Arrival')
        self.confirm = widgets.Checkbox(value=False, description='Other three contacts maintained throughout', indent=False)
        self.notes = widgets.Textarea(description='Notes', layout=widgets.Layout(width='500px'))
        self.annotation_path = widgets.Text(value=str((evidence.root / 'state/endpoint_annotations.json').relative_to(evidence.project)),
            description='Load file', layout=widgets.Layout(width='650px'))
        self.info, self.preview, self.message = widgets.HTML(), widgets.Output(), widgets.Output()
        self.moves = widgets.Dropdown(options=[(f"{m['moving_limb']}: {m['from_hold']} → {m['to_hold']}", i)
            for i, m in enumerate(evidence.prediction['moves'])], description='Model move')
        self.automatic_review = None
        for phase in self.fields.values():
            for field in phase.values():
                field.observe(self.invalidate, names='value')
        self.moving.observe(self.invalidate, names='value')
        for field in [self.release, self.arrival]:
            field.observe(lambda _: setattr(self.confirm, 'value', False), names='value')
        self.slider.observe(lambda _: self.refresh(), names='value')
        buttons = []
        actions = [('Set BEFORE', lambda: self.set_endpoint('before')), ('Set AFTER', lambda: self.set_endpoint('after')),
            ('Prefill automatic draft', self.prefill), ('Load annotations', self.load_saved), ('Export annotations', self.export)]
        for title, action in actions:
            button = widgets.Button(description=title, layout=widgets.Layout(width='200px'))
            button.on_click(self.callback(action));buttons.append(button)
        self.ui = widgets.VBox([self.moves, buttons[2], self.slider, self.preview, widgets.HBox(buttons[:2]), self.info, self.moving,
            widgets.HTML('<b>Limb | BEFORE contact | AFTER contact</b>'),
            *[widgets.HBox([widgets.Label(limb), self.fields['before'][limb], self.fields['after'][limb]]) for limb in LIMBS],
            widgets.HTML('Release = first free frame; arrival = first new-contact frame; −1 = unset.'),
            widgets.HBox([self.release, self.arrival]), self.confirm, self.notes, self.annotation_path,
            widgets.HBox(buttons[3:]), self.message])
        self.refresh()

    def callback(self, action):
        def call(_):
            with self.message:
                self.message.clear_output()
                try:
                    action()
                except (ValueError, FileNotFoundError) as error:
                    print(error)
        return call

    def check_route(self):
        self.evidence.check_current()
        if self.route != self.evidence.selected:
            raise ValueError('Route draft changed. Rerun this review cell before annotating.')

    def invalidate(self, change=None):
        self.confirm.value = False
        self.release.value = self.arrival.value = -1
        self.automatic_review = None

    def set_endpoint(self, phase):
        self.check_route()
        if self.frames[phase] != self.slider.value:
            self.frames[phase] = self.slider.value
            self.invalidate()
            for field in self.fields[phase].values():
                field.value = 'unknown'
        self.refresh()

    def state(self):
        data = dict(video_sha256=self.evidence.input_hashes[str(self.evidence.video.relative_to(self.evidence.project))],
            candidate_run_id=self.evidence.run_id, selected_hold_ids=self.route,
            reference_frame_number=self.evidence.reference, frames=dict(self.frames), moving_limb=self.moving.value,
            contacts={phase: {limb: field.value for limb, field in fields.items()} for phase, fields in self.fields.items()},
            notes=self.notes.value, body=dict(height_cm=self.height, ratios=dict(RATIOS), px_per_cm=None))
        if self.release.value != -1 or self.arrival.value != -1 or self.confirm.value:
            key = 'motion' if self.confirm.value else 'suggested_motion'
            data[key] = dict(release_frame=self.release.value, arrival_frame=self.arrival.value,
                support_contacts_maintained=self.confirm.value)
        if self.automatic_review is not None:
            data['automatic_review'] = self.automatic_review
        return data

    def prefill(self):
        self.check_route()
        if self.route != self.evidence.saved_selected:
            raise ValueError('Predictions belong to the saved route. Review the new route manually; regenerate inference before automatic prefill.')
        prediction, contacts = load_review_bundle(self.evidence.video, self.evidence.root,
            self.evidence.root / 'automatic/wholebody.npz', self.evidence.project / 'models/contact_v1',
            self.evidence.root / 'automatic/predictions')
        if self.moves.value is None:
            raise ValueError('No automatic move is available.')
        move = prediction['moves'][self.moves.value]
        draft = review_annotation_from_prediction(self.state(), move, prediction, contacts)
        self.populate(draft)
        self.release.value, self.arrival.value = move['release_frame'], move['arrival_frame']
        self.confirm.value = False
        self.automatic_review = dict(prediction_sha256=file_sha256(self.evidence.root/'automatic/predictions/automatic_contacts.json'),
            original_suggestion=move, prefill_supports_confirmed=False)
        print('Automatic suggestion loaded. Contact continuity still needs human review.')

    def populate(self, data):
        self.frames.update(data['frames']);self.moving.value = data['moving_limb']
        for phase in self.fields:
            for limb, field in self.fields[phase].items():
                field.value = data['contacts'][phase][limb]
        self.notes.value = data.get('notes', '')
        motion = data.get('motion') or data.get('suggested_motion') or {}
        self.release.value = motion.get('release_frame', -1)
        self.arrival.value = motion.get('arrival_frame', -1)
        self.confirm.value = motion.get('support_contacts_maintained', False)
        self.slider.value = self.frames['before'] or 0
        self.refresh()

    def load_saved(self):
        self.check_route()
        path = self.evidence.project / self.annotation_path.value
        data = validate_annotations(json.loads(path.read_text()), self.evidence.video,
            self.evidence.run_id, self.route, self.evidence.count)
        self.populate(data)
        self.automatic_review = data.get('automatic_review')
        print(f"Loaded historical contacts. Their stored height is {data['body']['height_cm']} cm; current target remains {self.height} cm.")

    def export(self, directory=None):
        self.check_route();data = self.state()
        validate_annotations(data, self.evidence.video, self.evidence.run_id, self.route, self.evidence.count)
        if any(f is None for f in self.frames.values()):
            raise ValueError('Choose BEFORE and AFTER before exporting.')
        timing = data.get('suggested_motion')
        if timing:
            release, arrival = timing['release_frame'], timing['arrival_frame']
            if any(v < -1 or v >= self.evidence.count for v in (release, arrival)):
                raise ValueError('Draft timing must be −1 or a valid frame number.')
            if release != -1 and arrival != -1 and not self.frames['before'] < release < arrival <= self.frames['after']:
                raise ValueError('Draft timing requires BEFORE < RELEASE < ARRIVAL <= AFTER.')
        folder = self.evidence.export_route(directory)
        path = folder / 'endpoint_annotations.json'
        path.write_text(json.dumps(data, indent=2) + '\n')
        if directory is None:
            link(self.evidence.project, path)
        return path

    def refresh(self):
        self.info.value = f"BEFORE: {self.frames['before']} | AFTER: {self.frames['after']}"
        with self.preview:
            self.preview.clear_output(wait=True)
            self.evidence.show_image(frame=self.slider.value, skeleton=True)
            display(self.evidence.contacts.loc[self.evidence.contacts.frame == self.slider.value,
                ['limb', 'hold_id', 'confidence', 'status']])


def saved_stage(project, item):
    """Display explicitly selected historical artifacts with integrity checks."""
    project = Path(project)
    display(Markdown(f"**Historical result:** {item['scope']}. {item.get('note', '')}"))
    for relative, expected in item['sha256'].items():
        if file_sha256(project / relative) != expected:
            raise ValueError('Historical artifact changed: ' + relative)
    if item.get('summary'):
        summary_path = project / item['summary']
        summary = json.loads(summary_path.read_text())
        for relative, expected in summary.get('output_sha256', {}).items():
            if file_sha256(summary_path.parent / relative) != expected:
                raise ValueError('Historical output integrity failure: ' + relative)
        fields = ['body', 'frame_count', 'first_frame', 'last_frame', 'selected_candidate', 'selected_profile',
                  'selection_status', 'optimizer', 'flag_counts', 'physical_validation']
        display(pd.Series({k: summary[k] for k in fields if k in summary}, name='Saved summary'))
    for relative in item.get('tables', []):
        table(project / relative);link(project, project / relative)
    for relative in item.get('images', []):
        picture(project / relative)
    for relative in item.get('videos', []):
        video(project, project / relative)
    for relative in item.get('links', []):
        link(project, project / relative)
    return item
