"""Assemble observed contact predictions without changing labels or running IK."""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .contact_review import load_review_bundle
from .inverse_kinematics import LIMBS
from .project_io import file_sha256


MOVE_COLUMNS = [
    'move_id', 'source_move_index', 'moving_limb', 'from_hold', 'to_hold',
    'before_frame', 'release_frame', 'arrival_frame', 'after_frame',
    'release_sec', 'arrival_sec', 'prediction_status', 'timing_status', 'overlap_move_ids',
    'supports_maintained', 'eligible_for_ik', 'prediction_notes',
]


def _contact_grid(prediction, contacts):
    """Validate native-frame coverage and preserve candidate/accepted distinctions."""
    required = {'frame', 'limb', 'candidate_hold', 'hold_id', 'confidence', 'status'}
    if contacts.empty or not required.issubset(contacts.columns):
        raise ValueError('Expected a nonempty learned contact table with prediction status.')
    frames = contacts['frame'].to_numpy(dtype=float)
    if not np.isfinite(frames).all() or (frames < 0).any() or (frames != np.floor(frames)).any():
        raise ValueError('Contact frame IDs must be nonnegative integers.')
    table = contacts.copy()
    table['frame'] = frames.astype(int)
    count = int(frames.max()) + 1
    expected = pd.MultiIndex.from_product([range(count), LIMBS], names=['frame', 'limb'])
    if table.duplicated(['frame', 'limb']).any():
        raise ValueError('Duplicate contact frame/limb rows.')
    indexed = table.set_index(['frame', 'limb'])
    if set(indexed.index) != set(expected):
        raise ValueError('Contacts must cover all four limbs on every native frame.')
    states = set(prediction['selected_hold_ids']) | {'free', 'wall_contact', 'unknown'}
    for column in ('candidate_hold', 'hold_id'):
        if not set(table[column]).issubset(states):
            raise ValueError(f'Unrecognized contact state in {column}.')
    confidence = table['confidence'].to_numpy(dtype=float)
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError('Contact confidence must lie between zero and one.')
    if not table['status'].isin(['accepted_model', 'review_required']).all():
        raise ValueError('Unrecognized learned prediction status.')
    accepted = table['status'] == 'accepted_model'
    if ((accepted & (table.hold_id != table.candidate_hold)) |
            (~accepted & (table.hold_id != 'unknown'))).any():
        raise ValueError('Accepted labels and prediction status disagree.')
    fps = float(prediction['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Prediction FPS must be positive.')
    return indexed.reindex(expected), count, fps


def _assemble_schedule(indexed, count, fps):
    """Group consecutive frames with identical four-limb states and acceptance."""
    wide = pd.DataFrame(index=range(count))
    confidence = {}
    for limb in LIMBS:
        rows = indexed.xs(limb, level='limb')
        wide[f'{limb}_candidate'] = rows.candidate_hold
        wide[f'{limb}_accepted'] = rows.hold_id
        wide[f'{limb}_status'] = rows.status
        confidence[limb] = rows.confidence.to_numpy(dtype=float)
    changed = wide.ne(wide.shift()).any(axis=1).to_numpy()
    boundaries = np.r_[np.flatnonzero(changed), count]
    records = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        state = wide.iloc[start].to_dict()
        uncertain = [limb for limb in LIMBS if state[f'{limb}_status'] != 'accepted_model']
        free = [limb for limb in LIMBS if state[f'{limb}_candidate'] == 'free']
        records.append(dict(
            segment_id=len(records) + 1, start_frame=int(start), end_frame=int(stop - 1),
            start_sec=start / fps, end_sec=stop / fps, **state,
            **{f'{limb}_min_confidence': float(confidence[limb][start:stop].min()) for limb in LIMBS},
            free_limbs=','.join(free), uncertain_limbs=','.join(uncertain),
            prediction_status='uncertain' if uncertain else 'model_accepted'))
    return pd.DataFrame(records)


def _assemble_moves(prediction, indexed, count, fps):
    """Sort existing proposals by release and report overlapping free intervals."""
    records = []
    for source_index, move in enumerate(prediction['moves']):
        before, release, arrival, after = (move[k] for k in
            ('before_frame', 'release_frame', 'arrival_frame', 'after_frame'))
        if (any(type(f) is not int for f in (before, release, arrival, after)) or
                not 0 <= before <= release <= arrival <= after < count):
            raise ValueError(f'Invalid timing in source move {source_index}.')
        limb = move['moving_limb']
        if limb not in LIMBS or any(move[k] not in prediction['selected_hold_ids']
                                   for k in ('from_hold', 'to_hold')):
            raise ValueError(f'Invalid limb or hold in source move {source_index}.')
        observed = indexed.xs(limb, level='limb').candidate_hold
        if observed.iloc[before] != move['from_hold'] or observed.iloc[arrival] != move['to_hold']:
            raise ValueError(f'Source move {source_index} disagrees with its contact table.')
        resolved = (arrival > release and (observed.iloc[release:arrival] == 'free').all())
        records.append(dict(
            source_move_index=source_index, moving_limb=limb,
            from_hold=move['from_hold'], to_hold=move['to_hold'],
            before_frame=before, release_frame=release, arrival_frame=arrival, after_frame=after,
            release_sec=release / fps, arrival_sec=arrival / fps,
            prediction_status='uncertain' if move['requires_review'] else 'model_accepted',
            timing_status='resolved_free_interval' if resolved else 'unresolved_transition',
            supports_maintained=bool(move['supports_maintained']),
            eligible_for_ik=bool(move['eligible_for_ik']),
            prediction_notes='; '.join(move.get('review_reasons', []))))
    records.sort(key=lambda r: (r['release_frame'], r['arrival_frame'],
                               LIMBS.index(r['moving_limb']), r['source_move_index']))
    for number, row in enumerate(records, 1):
        row['move_id'] = number
    for row in records:
        row['overlap_move_ids'] = ','.join(str(other['move_id']) for other in records
            if other is not row and max(row['release_frame'], other['release_frame']) <
            min(row['arrival_frame'], other['arrival_frame']))
    return pd.DataFrame(records, columns=MOVE_COLUMNS)


def assemble_move_sequence(prediction, contacts):
    """Return chronological moves and a lossless discrete contact schedule."""
    indexed, count, fps = _contact_grid(prediction, contacts)
    return (_assemble_moves(prediction, indexed, count, fps),
            _assemble_schedule(indexed, count, fps))


def plot_contact_timeline(schedule, moves, fps):
    """Draw candidates with hatching where the model did not accept a label."""
    if schedule.empty:
        raise ValueError('Cannot draw an empty contact schedule.')
    holds = sorted({state for limb in LIMBS for state in schedule[f'{limb}_candidate']
                    if state not in {'unknown', 'free', 'wall_contact'}})
    colors = {hold: plt.get_cmap('tab20')(i % 20) for i, hold in enumerate(holds)}
    colors.update(free='#ffffff', wall_contact='#b8c5cd', unknown='#777777')
    fig, ax = plt.subplots(figsize=(14, 5))
    duration = (int(schedule.end_frame.max()) + 1) / fps
    for lane, limb in enumerate(LIMBS):
        for row in schedule.to_dict('records'):
            start, width = row['start_sec'], row['end_sec'] - row['start_sec']
            candidate = row[f'{limb}_candidate']
            uncertain = row[f'{limb}_status'] != 'accepted_model'
            ax.broken_barh([(start, width)], (lane - .32, .64), facecolors=colors[candidate],
                           edgecolors='#555555', linewidth=.25, hatch='///' if uncertain else None)
        # Label whole candidate runs even when acceptance changes within them.
        values = schedule[f'{limb}_candidate'].to_numpy()
        changes = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
        for start, stop in zip(changes[:-1], changes[1:]):
            left, right = schedule.iloc[start].start_sec, schedule.iloc[stop - 1].end_sec
            if right - left >= duration * .025:
                ax.text((left + right) / 2, lane, values[start], ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=1))
    for row in moves.itertuples():
        lane = LIMBS.index(row.moving_limb)
        ax.plot([row.release_sec, row.arrival_sec], [lane - .43] * 2, color='#222222', linewidth=1.5)
        if row.timing_status == 'unresolved_transition':
            ax.plot(row.release_sec, lane - .43, marker='D', color='#222222', markersize=3)
        ax.text((row.release_sec + row.arrival_sec) / 2, lane - .45, f'M{row.move_id}',
                ha='center', va='bottom', fontsize=7)
    ax.set(yticks=range(4), yticklabels=[limb.replace('_', ' ').title() for limb in LIMBS],
           xlim=(0, duration), ylim=(3.6, -.85), xlabel='Video time (seconds)',
           title='Original climber: predicted contacts and hold-change moves')
    ax.grid(axis='x', alpha=.2)
    ax.legend(handles=[Patch(facecolor='white', edgecolor='#555555', hatch='///', label='Uncertain prediction'),
                       Patch(facecolor='white', edgecolor='#555555', label='Free'),
                       Patch(facecolor=colors['wall_contact'], label='Wall contact'),
                       Patch(facecolor=colors['unknown'], label='Unknown')],
              loc='upper center', bbox_to_anchor=(.5, -.15), ncol=4, fontsize=8)
    fig.text(.5, .015, 'Hatching: uncertain candidate. M numbers: move table. Diamonds: unresolved free-motion phase.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    return fig


def save_move_sequence(video, output_dir, pose_cache, package, predictions):
    """Verify saved inference, then export only new sequence artifacts."""
    root, predictions = Path(output_dir), Path(predictions)
    prediction, contacts = load_review_bundle(video, root, pose_cache, package, predictions)
    moves, schedule = assemble_move_sequence(prediction, contacts)
    fps, count = float(prediction['fps']), int(contacts.frame.max()) + 1
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
            raise ValueError('Prediction coverage does not match the complete video.')
        if not np.isclose(cap.get(cv2.CAP_PROP_FPS), fps):
            raise ValueError('Prediction FPS does not match the video.')
    finally:
        cap.release()
    destination = root / 'automatic/sequence'
    destination.mkdir(parents=True, exist_ok=True)
    moves.to_csv(destination / 'move_sequence.csv', index=False)
    schedule.to_csv(destination / 'contact_schedule.csv', index=False)
    fig = plot_contact_timeline(schedule, moves, fps)
    try:
        fig.savefig(destination / 'contact_timeline.png', dpi=160)
    finally:
        plt.close(fig)
    uncertain_frames = sum(row.end_frame - row.start_frame + 1 for row in schedule.itertuples()
                           if row.prediction_status == 'uncertain')
    summary = dict(
        schema_version=1, source='learned_predictions', frame_count=count, fps=fps,
        video_sha256=prediction['video_sha256'],
        prediction_sha256=file_sha256(predictions / 'automatic_contacts.json'),
        contacts_sha256=file_sha256(predictions / 'limb_contacts.csv'),
        assembler_sha256=file_sha256(__file__), move_count=len(moves),
        schedule_segments=len(schedule), frames_with_uncertainty=int(uncertain_frames),
        unresolved_transitions=int((moves.timing_status == 'unresolved_transition').sum()),
        overlapping_move_pairs=sum(len(r.overlap_move_ids.split(',')) if r.overlap_move_ids else 0
                                   for r in moves.itertuples()) // 2,
        frame_convention='Zero-based native frames; end_frame inclusive, end_sec exclusive. '
                         'Move intervals use [release_frame, arrival_frame); a zero-duration transition '
                         'has unresolved timing and is not counted as an overlapping interval.',
        interpretation='Original model predictions, not human-confirmed contacts or an IK-ready plan. '
                       'No smoothing, gap filling, manual timing overrides, or new moves applied.',
        output_sha256={name: file_sha256(destination / name) for name in
                       ('move_sequence.csv', 'contact_schedule.csv', 'contact_timeline.png')})
    (destination / 'sequence_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return moves, schedule, summary
