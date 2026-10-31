"""Refresh trial2 evaluation from the saved route and verified contact predictions."""
import argparse
import base64
import html
import json
from pathlib import Path
import re
import sys
import zipfile

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beta_translator.contact_review import load_review_bundle
from beta_translator.project_io import file_sha256, load_reviewed_holds
from evaluate_hold_maps import evaluate

STAGES = {
    'original100': 'Original frame 450, cap 100',
    'cap400': 'Frame 450, cap 400',
    'fusion': 'Reconstructed wall, cap 400',
    'fusion_tiled': 'Reconstruction plus overlapping tiles',
}


def save_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def refresh(repo, out):
    video = repo / 'videos/trial2.mp4'
    map_path = out / 'state/reviewed_hold_map.npz'
    selection = json.loads((out / 'state/hold_selection_status.json').read_text())
    prediction, contacts = load_review_bundle(video, out, out / 'automatic/wholebody.npz',
                                              repo / 'models/contact_v1', out / 'automatic/predictions')
    records, masks, run_id, ids = load_reviewed_holds(map_path, video, reference_frame_number=450)
    if (selection['selected_hold_ids'] != ids or selection['candidate_run_id'] != run_id
            or selection.get('reviewed_map_sha256') != file_sha256(map_path)):
        raise ValueError('Route selection metadata is stale. Rerun Section 4B and contact inference.')
    build_path = out / 'state/hold_map_build.json'
    build = json.loads(build_path.read_text())
    artifact_dir = out / 'hold_mapping'
    if (build['candidate_run_id'] != run_id or build['candidate_cache_sha256'] !=
            file_sha256(artifact_dir / 'fusion_tiled.npz')):
        raise ValueError('This historical detector comparison belongs to a different candidate map.')
    historical = json.loads((artifact_dir / 'hold_evaluation.json').read_text())
    previous_points = {m['hold_id']: m['point'] for m in historical['stages']['fusion_tiled']['matches']
                       if m['hold_id'] is not None}
    # Preserve independent source-review points when available. Additional points
    # are derived from selected masks and explicitly excluded from accuracy claims.
    points = []
    for hold in ids:
        if hold in previous_points:
            xy = previous_points[hold]
            source = 'Historical source-image review'
        else:
            ys, xs = np.nonzero(masks[hold])
            nearest = np.argmin((xs - xs.mean())**2 + (ys - ys.mean())**2)
            xy = [int(xs[nearest]), int(ys[nearest])]
            source = 'Interior pixel nearest the selected-mask centroid; not independent ground truth'
        points.append(dict(audit_id=hold, point=xy, point_source=source))
    audit = dict(reference_frame_number=450, candidate_run_id=run_id,
                 reviewed_map_sha256=file_sha256(map_path), selected_hold_ids=ids,
                 tolerance_px=3, holds=points,
                 scope='Coverage of currently selected hold points in frozen detector variants',
                 limitation='Selection comes from the final detected map. Missing unselected holds are not assessed. This is not independent detection recall, mask AP or contact accuracy.')
    stages = evaluate(artifact_dir, audit=audit, include_matches=True)
    count = len(ids)
    accepted = int((contacts.status == 'accepted_model').sum())
    coverage = accepted / len(contacts)
    move_count = len(prediction['moves'])
    frame_count = int(contacts.frame.max()) + 1
    with np.load(map_path, allow_pickle=False) as data:
        reference = data['reference_frame'].copy()
    drawing = reference.copy()
    selected_records = [r for r in records if r['hold_id'] in ids]
    for r in selected_records:
        contours, _ = cv2.findContours(masks[r['hold_id']].astype(np.uint8), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(drawing, contours, -1, (0, 180, 255), 2)
        xy = (int(r['cx_px']) + 8, int(r['cy_px']) - 4)
        for thickness, color in [(4, (0, 0, 0)), (1, (255, 255, 255))]:
            cv2.putText(drawing, r['hold_id'], xy, 0, .57, color, thickness, cv2.LINE_AA)
    bounds = np.array([[r[k] for k in ['x_min', 'y_min', 'x_max', 'y_max']] for r in selected_records])
    x0 = max(0, int(bounds[:, 0].min()) - 60)
    y0 = max(0, int(bounds[:, 1].min()) - 50)
    x1 = min(reference.shape[1], int(bounds[:, 2].max()) + 125)
    y1 = min(reference.shape[0], int(bounds[:, 3].max()) + 60)
    images = out / 'images'
    cv2.imwrite(str(images / 'confirmed_route_map.png'), drawing)
    cv2.imwrite(str(images / 'confirmed_route_detail.png'),
                cv2.resize(drawing[y0:y1, x0:x1], None, fx=1.7, fy=1.7))

    fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
    values = [s['matched_points'] / count * 100 for s in stages.values()]
    bars = ax.bar(['Original\ncap 100', 'Cap 400\nonly', 'Observed-pixel\nfusion', 'Fusion +\noverlapping tiles'],
                  values, color=['#aab2bd', '#5d95b2', '#248a9e', '#237a59'], width=.6)
    for bar, stage in zip(bars, stages.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{stage['matched_points']}/{count}", ha='center')
    ax.set(ylim=(0, 112), ylabel='Selected-point coverage (%)',
           title=f'Current {count}-hold route: coverage, not independent detection accuracy')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(images / 'current_route_hold_coverage.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 7), layout='constrained')
    for ax, key, label in zip(axes, ['original100', 'fusion_tiled'], ['Original map', 'Reconstruction + tiles']):
        ax.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
        for match in stages[key]['matches']:
            x, y = match['point']
            color = '#13c866' if match['hold_id'] else '#f33b42'
            ax.scatter([x], [y], s=42, facecolors='none', edgecolors=color, linewidths=1.5)
            ax.annotate(match['audit_id'], (x, y), xytext=(4, 3), textcoords='offset points',
                        color=color, fontsize=7, bbox=dict(facecolor='black', alpha=.55, edgecolor='none', pad=.4))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(f"{label}: {stages[key]['matched_points']}/{count}")
        ax.axis('off')
    fig.suptitle('Same selected points; green = covered, red = missed', fontsize=11)
    fig.savefig(images / 'current_route_point_comparison.png', dpi=160)
    plt.close(fig)

    build.update(selected_candidates=count, selected_hold_ids=ids,
                 human_route_confirmed=selection['status'] == 'user_confirmed_route',
                 route_selection_source=selection['source'],
                 current_route_evaluation='hold_mapping/current_route_evaluation.json')
    save_json(build_path, build)
    summary_path = out / 'state/trial2_run_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['hold_map'].update(selected_candidates=count, selected_hold_ids=ids,
        selection='Current manual selection from Section 4B; order is not a movement sequence')
    import pandas as pd
    pose_rows = pd.read_csv(out / 'tables/pose_coordinates.csv')
    summary['mediapipe'].update(detected_frames=len(pose_rows),
        missing_frames=sorted(set(range(frame_count)) - set(pose_rows.frame.astype(int))),
        frame_coverage=len(pose_rows)/frame_count,
        interpretation='Target-verified observation coverage; display-only estimates are excluded.')
    with np.load(out / 'state/alignment_cache.npz', allow_pickle=False) as alignment:
        inliers = alignment['inliers']
        summary['alignment'].update(frames=len(inliers), passed_inlier_gate=int((inliers>=25).sum()),
            minimum_inliers=int(inliers.min()), median_inliers=float(np.median(inliers)))
    if (out / 'state/identity_tracking_summary.json').exists():
        summary['route_selection_comparison'].pop('unchanged_inputs', None)
        summary['route_selection_comparison']['comparison_scope'] = 'Earlier route selection predates identity correction; current results also include corrected pose alignment and an identity observation guard.'
    summary['automatic_contacts'].update(limb_frames=len(contacts), accepted_limb_frames=accepted,
        confidence_threshold_coverage=coverage, review_required_limb_frames=len(contacts)-accepted,
        movement_proposals=move_count, ik_eligible_proposals=prediction['eligible_moves'])
    summary['route_selection_comparison'].update(selected_candidates=count,
        confidence_threshold_coverage=coverage, movement_proposals=move_count)
    summary['current_route_point_coverage'] = {key: {k: v for k, v in stage.items() if k != 'matches'}
                                               for key, stage in stages.items()}
    for name, details in summary['videos'].items():
        path = out / 'videos' / name
        cap = cv2.VideoCapture(str(path))
        expected_frames = details['source_end_frame'] - details['source_start_frame'] + 1 if 'source_start_frame' in details else frame_count
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != expected_frames:
            raise ValueError(f'Video frame count differs from predictions: {name}')
        cap.release()
        details['sha256'] = file_sha256(path)
    save_json(summary_path, summary)

    rows = '\n'.join(f"| {STAGES[key]} | {stage['candidate_count']} | {stage['matched_points']}/{count} | {stage['priority_point_recall']:.1%} |"
                     for key, stage in stages.items())
    old_rows = '\n'.join(f"| {STAGES[key]} | {stage['matched_points']}/{stage['audited_points']} |"
                         for key, stage in historical['stages'].items())
    derived = [p['audit_id'] for p in points if p['point_source'].startswith('Interior')]
    markdown = f'''### Current route: {count} selected holds

Read from the saved Section 4B selection and verified predictions:
{', '.join(ids)}.

| Current result | Value |
| --- | ---: |
| Stored detector candidates | {len(records)} |
| Selected route holds | {count} |
| Video frames analyzed | {frame_count} |
| Limb-frame predictions above confidence threshold | {accepted}/{len(contacts)} ({coverage:.2%}) |
| Movement proposals requiring review | {move_count} |
| Proposals eligible for unreviewed IK | {prediction['eligible_moves']} |

Confidence coverage is not contact accuracy or climbing-success probability.
Route membership does not confirm limb assignments, timing or support continuity.

### Current-route point coverage across detector variants

| Variant | Detector candidates | Selected points covered | Coverage |
| --- | ---: | ---: | ---: |
{rows}

Matching uses one-to-one mask assignments within 3 reference-image pixels.
Historical source-review points are reused when available; additional points
({', '.join(derived) if derived else 'none'}) use an interior pixel of the selected mask.
The route was selected from the final detections. This measures coverage of those
selected points, not independent detection recall or all missing wall holds.

### Historical experiment: fixed 15-point development audit

The original audit is preserved with its original points and denominator:

| Variant | Original audit points covered |
| --- | ---: |
{old_rows}

This historical sample is separate from the current route. The original spot
check of 40 raw candidates found 36 likely holds, 3 clear non-holds and 1 uncertain
object. Neither experiment establishes segmentation AP or contact accuracy.
'''
    evaluation = dict(audit=audit, stages=stages, contact_summary=summary['automatic_contacts'],
                      historical_audit='hold_mapping/hold_evaluation.json', notebook_markdown=markdown)
    save_json(artifact_dir / 'current_route_evaluation.json', evaluation)

    def embedded(name, caption):
        data = base64.b64encode((images / name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'

    table_rows = ''.join(f"<tr><td>{STAGES[key]}</td><td>{stage['candidate_count']}</td><td>{stage['matched_points']}/{count}</td><td>{stage['priority_point_recall']:.1%}</td></tr>"
                         for key, stage in stages.items())
    panel = f'''<section id="user-route"><h2>Current Section 4B route: {count} selected holds</h2>
<p>{', '.join(ids)}.</p>
<p>All {len(records)} candidate masks remain stored. Route membership is user-selected; contact order, release/arrival and support continuity still require review.</p>
<p>Verified inference covers {frame_count} frames: {accepted}/{len(contacts)} limb-frame predictions ({coverage:.2%}) pass the confidence threshold, and {move_count} movement proposals require review. These figures are not contact accuracy or climbing-success probability.</p>
<h3>Coverage of the current selected points</h3>
<table><tr><th>Variant</th><th>Candidates</th><th>Points covered</th><th>Coverage</th></tr>{table_rows}</table>
<p>One-to-one matches use a 3-pixel tolerance. Historical source-review points are reused when available. Additional points ({', '.join(derived) if derived else 'none'}) come from selected masks. Selection from the final detections makes this a coverage check, not an independent detection benchmark. Unselected missing holds are not assessed.</p>
{embedded('confirmed_route_detail.png', f'Current {count}-hold route; membership does not specify movement order.')}
{embedded('current_route_hold_coverage.png', 'Current-route point coverage; separate from the historical fixed 15-point audit.')}
{embedded('current_route_point_comparison.png', 'Original and repaired detector variants evaluated at the same currently selected points.')}
<p>The following sections preserve the earlier detector-repair experiment, including its fixed 15-point development audit and pre-selection inference results.</p></section>'''
    report_path = out / 'reports/hold_map_evaluation.html'
    report = report_path.read_text()
    report, replacements = re.subn(r'<section id="user-route">.*?</section>', lambda _: panel, report, count=1, flags=re.S)
    if replacements != 1:
        raise ValueError('Expected one current-route report section; refusing an ambiguous replacement.')
    report_path.write_text(report)

    archive_path = out / 'exports/reviewed_contact_data.zip'
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    names.update(['state/trial2_run_summary.json', 'state/hold_map_build.json',
                  'state/hold_selection_status.json', 'hold_mapping/current_route_evaluation.json',
                  'reports/hold_map_evaluation.html', 'images/confirmed_route_map.png',
                  'images/confirmed_route_detail.png', 'images/current_route_hold_coverage.png',
                  'images/current_route_point_comparison.png', 'images/all_hold_candidates.png',
                  'images/automatic_contact_storyboard.jpg'])
    names.update(str(p.relative_to(out)) for folder in ['hold_mapping', 'reference_runs', 'images']
                 for p in (out / folder).rglob('*') if p.is_file())
    temporary = archive_path.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.write(out / name, name)
    temporary.replace(archive_path)
    print(json.dumps(dict(selected_holds=count, accepted_fraction=coverage,
                          movement_proposals=move_count,
                          point_coverage={key: stage['matched_points'] for key, stage in stages.items()})))
    return evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=Path('outputs/trial2'))
    args = parser.parse_args()
    refresh(Path(__file__).resolve().parents[1], args.outputs.resolve())
