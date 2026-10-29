"""Create an English model card from measured results, never hard-coded scores."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beta_translator.wholebody_pose import sha256
import torch


def percentage(value):
    return 'N/A' if value is None else f'{value*100:.2f}%'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--project-check', type=Path)
    args = parser.parse_args()
    package = args.package
    config = json.loads((package/'deployment.json').read_text())
    test = json.loads((package/'test_report.json').read_text())
    validation = json.loads((package/'validation_report.json').read_text())
    manifest = json.loads((package/'dataset_manifest.json').read_text())
    baseline, model = test['baseline']['aggregate'], test['model']['aggregate']
    val = validation['model']['aggregate']
    runs = []
    for checkpoint in config['checkpoints']:
        saved = torch.load(package/checkpoint['file'], map_location='cpu', weights_only=True)
        training = saved['training']
        history = json.loads((args.runs/f"seed{training['seed']}"/'history.json').read_text())
        runs.append(dict(seed=training['seed'], selected_epoch=saved['epoch'],
                         epochs=len(history), updates=len(history)*training['steps'],
                         validation_nll=saved['validation_nll'], elapsed_sec=history[-1]['elapsed_sec'],
                         parameters=training['parameters'], trainer_sha256=training['trainer_sha256']))
    (package/'training_summary.json').write_text(json.dumps(runs, indent=2)+'\n')
    table = []
    for key, label in [('accuracy', 'Limb-frame accuracy'), ('macro_video_accuracy', 'Mean video accuracy'),
                       ('event_precision', 'Contact-event precision'), ('event_recall', 'Contact-event recall'),
                       ('event_f1', 'Contact-event F1 (IoU >= 0.5)')]:
        table.append(f'| {label} | {percentage(baseline[key])} | {percentage(model[key])} |')
    summary = '\n'.join(table)
    run_table = '\n'.join(f"| {r['seed']} | {r['epochs']} | {r['updates']} | {r['selected_epoch']} | {r['validation_nll']:.4f} |" for r in runs)
    timing = []
    for kind in ('release', 'arrival'):
        timing.append(f"- {kind.title()}: MAE {model[kind+'_mae_sec']:.3f} s; median {model[kind+'_median_sec']:.3f} s; "
                      f"{percentage(model[kind+'_within_0.2s'])} within 0.20 s; {model[kind+'_matched_boundaries']} matched, uncensored boundaries.")
    clip_table = '\n'.join(f"| {c['clip']} | {percentage(c['frame']['accuracy'])} | {percentage(c['event']['f1'])} | "
                           f"{percentage(c['frame']['coverage'])} | {percentage(c['frame']['accepted_accuracy'])} |" for c in test['model']['clips'])
    gate_text = (f"The maximum local timing-range width is {config['max_timing_width_sec']} s."
                 if config['automatic_ik_enabled'] else
                 'Automatic IK is disabled: no ensemble movement-gate setting met the validation requirement '
                 'of at least ten accepted proposals with at least 95% correct identities/supports and both boundaries within 0.20 s. '
                 'Use automatic preannotations followed by movement review.')
    text = f'''# Beta Translator contact model v1

This package contains a trained offline contact model and reproducible held-out evaluation. It predicts observed hand/foot contact with supplied holds and proposes release/arrival frames. It does not predict climbing success or physical feasibility.

## Measured held-out results

The test contains p9 and p10: four videos, 6,120 frames, on the same two public routes as training. Configuration was locked using p7 and p8 before test evaluation. There are {model['labeled_limb_frames']:,} scored limb-frames; unannotated margins and conflicting labels are excluded.

| Metric | Geometry + temporal baseline | Trained ensemble |
|---|---:|---:|
{summary}

The baseline uses the same frozen whole-body landmarks and reviewed hold boxes; distance and temporal settings are selected on validation. This comparison isolates learned contact inference rather than changing both the pose input and classifier.

At the deployed confidence threshold **{config['accept_threshold']:.3f}**, accepted test limb-frames have **{percentage(model['accepted_accuracy'])} accuracy** and cover **{percentage(model['coverage'])}** of labeled limb-frames. The remaining **{percentage(1-model['coverage'])}** require review under this confidence rule. This is a limb-frame coverage proxy, not a measured reduction in human working time.

There are {model['ik_eligible_proposals']} IK-eligible proposals out of {model['motion_proposals']} inferred hold-change proposals ({percentage(model['proposal_eligibility_fraction'])}). Of {model['scored_eligible_moves']} scorable accepted proposals, {model['correct_eligible_moves']} have correct hold/support identities and both boundaries within 0.20 s ({percentage(model['eligible_move_accuracy_at_0.2s'])}). This small conditional sample is not a physical-feasibility measurement.

Timing, conditional on successful event matching:

{chr(10).join(timing)}

Read timing together with event recall: missed events have no matched boundary error. Event endpoints are inclusive; release is the next frame after contact ends.

| Test clip | Frame accuracy | Event F1 | Accepted coverage | Accepted accuracy |
|---|---:|---:|---:|---:|
{clip_table}

![Evaluation](evaluation.png)

## Validation and model selection

Validation frame accuracy: {percentage(val['accuracy'])}; event F1: {percentage(val['event_f1'])}. Selected-label accuracy: {percentage(val['accepted_accuracy'])} at {percentage(val['coverage'])} coverage. Temperature: {config['temperature']}; decoder switch penalty: {config['switch_penalty']}.

| Seed | Completed epochs | Updates | Selected epoch | Validation NLL |
|---|---:|---:|---:|---:|
{run_table}

Total optimizer updates: {sum(r['updates'] for r in runs):,}. Each network has {runs[0]['parameters']:,} parameters. Training windows overlap and are sampled repeatedly; they are not independent new examples. Seeds share the same participant split. GPU execution may not be bitwise deterministic across hardware/library versions.

## Architecture and data

The frozen RTMW-x whole-body model supplies 133 keypoints, including fingertips and toes. A candidate-equivariant encoder and six residual dilated temporal convolution blocks score each supplied hold, free movement, and wall contact. The receptive field is 127 frames at 25 FPS (approximately 2.5 s of context on each side). Three validation-selected networks are averaged and temperature-calibrated. The system is offline and uses future frames.

The Way Up supplies 22 processed videos, 38,692 frames, 10 people and two routes. Train: p1–p6 (14 videos, including both p2 sessions). Validation: p7–p8 (four videos). Test: p9–p10 (four videos). The project's trial1 clip is excluded from training, selection, and calibration.

Of 940 source contact rows, one names an unspecified off-route hold. The parser masks its 11 limb-frames and 79 conflicting simultaneous-hold limb-frames. Terminal endFrame values one past the decoded video are clipped and logged. Occlusion labels are used for stratified evaluation, never as inputs. See dataset_manifest.json for per-clip audits and hashes.

The dataset has no anatomical keypoint ground truth. The pose backbone is publicly pretrained and remains frozen; this experiment trains the contact head and does not establish a new pose-localization accuracy figure.

## Integration and limitations

Open ../../Automatic_Contacts.ipynb or follow ../../docs/automatic_contacts.md. Input is a video, existing reviewed hold map, verified camera alignment, and the pretrained whole-body cache. Output is per-frame contact candidates, confidence, contact intervals, transition proposals and uncertain ranges. New candidates do not overwrite manually reviewed annotations.

Only transitions passing a separately validated movement gate, with three unchanged accepted supports, are eligible for automatic IK. **{gate_text}** Central 90% local path-mass ranges describe model ambiguity; they do not guarantee statistical coverage. Existing IK body proportions and wall scale remain necessary. Predictions are resampled back to native video frame IDs; temporal evidence remains limited by the 40 ms model grid. Visual contact is not a force measurement.

This evaluation measures unseen people on the same wall and routes. It does not establish unseen-gym accuracy, robust multi-person tracking, dynamic-camera accuracy, or reliability on steep walls. Confidence calibration can change under those conditions. Validate an independently labeled sample of new-wall moves before relying on automatic acceptance broadly.

## Sources and integrity

- Dataset: Anna Maschek and David Schedl, [The Way Up](https://zenodo.org/records/15196867), CC BY 4.0, DOI 10.5281/zenodo.15196867.
- Pose implementation: [RTMLib](https://github.com/Tau-J/rtmlib), Apache 2.0.
- Pose backbone: [MMPose RTMPose/RTMW](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose). Retain upstream model/data license conditions.
- Deployment configuration SHA-256: `{sha256(package/'deployment.json')}`.
- Dataset manifest SHA-256: `{config['manifest_sha256']}`.

Checkpoint hashes are verified at inference. Full numeric reports and training_summary.json accompany this card.
'''
    if args.project_check:
        smoke = json.loads(args.project_check.read_text())
        text += ('\n## Local project integration check\n\n'
            f"The model found {smoke['from_hold']} → {smoke['to_hold']}, but predicted release/arrival "
            f"at frames {smoke['predicted_release']}/{smoke['predicted_arrival']} versus the existing "
            f"reference {smoke['reference_release']}/{smoke['reference_arrival']}. "
            'Both reference boundaries fall outside the local model ambiguity ranges. '
            'This demonstrates new-wall timing error and overconfidence; do not interpret path-mass ranges '
            'as guaranteed coverage intervals. This clip was not used to tune the package.\n\n'
            '```json\n' + json.dumps(smoke, indent=2) + '\n```\n')
    (package/'MODEL_CARD.md').write_text(text)


if __name__ == '__main__': main()
