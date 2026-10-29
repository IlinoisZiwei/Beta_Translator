"""Calibrate on validation people, lock configuration, then evaluate held-out people."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.special import softmax
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beta_translator.contact_network import ContactTCN, predict_logits
from beta_translator.contact_decoding import decode_probabilities, ensemble_probabilities, export_predictions
from beta_translator.contact_evaluation import frame_metrics, event_metrics, aggregate, eligible_move_check
from beta_translator.wholebody_pose import sha256
from beta_translator.contact_baseline import geometric_probabilities


def clips_for(root, split):
    manifest = json.loads((root/'manifest.json').read_text())
    result = []
    for meta in manifest['clips']:
        if meta['split'] != split: continue
        with np.load(root/(meta['id']+'.npz'), allow_pickle=False) as data:
            clip = {key: data[key].copy() for key in ('features', 'distances', 'labels', 'occluded')}
        clip['meta'] = meta
        result.append(clip)
    return result


def model_logits(paths, clips, device):
    for clip in clips: clip['logits'] = []
    for path in paths:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = ContactTCN(**saved['model']).to(device)
        model.load_state_dict(saved['state_dict']); model.eval()
        for clip in clips:
            clip['logits'].append(predict_logits(model, clip['features'], device))
        del model


def report(clips, temperature, penalty, threshold, baseline=None, timing_width=.16, movement_threshold=None):
    results = []
    for clip in clips:
        if baseline is None:
            p = ensemble_probabilities(clip['logits'], temperature)
        else:
            p = geometric_probabilities(clip['distances'], baseline['distance'], baseline.get('window', 1))
        decoded = decode_probabilities(p, penalty)
        _, _, moves = export_predictions(p, decoded, [f'H{i}' for i in range(clip['meta']['holds'])],
                                          clip['meta']['fps'], threshold, max_timing_width_sec=timing_width,
                                          movement_threshold=movement_threshold)
        results.append({'clip': clip['meta']['id'],
            'frame': frame_metrics(p, clip['labels'], clip['occluded'], threshold, decoded),
            'event': event_metrics(decoded, clip['labels'], clip['meta']['holds'], clip['meta']['fps']),
            'motion_proposals': len(moves), 'ik_eligible_proposals': sum(m['eligible_for_ik'] for m in moves),
            'eligible_move_check': eligible_move_check(moves, clip['labels'], clip['meta']['fps'], clip['meta']['holds'])})
    return {'aggregate': aggregate(results), 'clips': results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, nargs='+')
    p.add_argument('--phase', choices=['calibrate', 'test'], required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args(); torch.set_num_threads(3)
    if args.phase == 'calibrate':
        if not args.checkpoints: raise ValueError('Provide trained checkpoints.')
        args.package.mkdir(parents=True, exist_ok=True)
        if (args.package/'deployment.json').exists():
            raise FileExistsError('Use a new package directory; do not silently change a locked evaluation.')
        clips = clips_for(args.data, 'validation')
        model_logits(args.checkpoints, clips, args.device)
        temperatures = [.5, .7, 1., 1.25, 1.5, 2., 2.5, 3.]
        calibration = []
        for temp in temperatures:
            loss = count = 0
            for clip in clips:
                prob = ensemble_probabilities(clip['logits'], temp)
                m = frame_metrics(prob, clip['labels'])
                loss += m['nll']*m['labeled_limb_frames']; count += m['labeled_limb_frames']
            calibration.append({'temperature': temp, 'nll': loss/count})
        temp = min(calibration, key=lambda x: x['nll'])['temperature']
        penalties = [0., 1., 2., 4., 6.]
        decoded_runs = [dict(penalty=cost, report=report(clips, temp, cost, .9)) for cost in penalties]
        chosen = max(decoded_runs, key=lambda x: x['report']['aggregate']['event_f1'])
        penalty = chosen['penalty']
        coverage = []
        for threshold in [.5, .6, .7, .8, .85, .9, .925, .95, .975, .99, .995, .999]:
            r = report(clips, temp, penalty, threshold)
            coverage.append({'threshold': threshold, **r['aggregate']})
        eligible = [r for r in coverage if r['accepted_accuracy'] is not None and r['accepted_accuracy'] >= .97
                    and r['coverage'] >= .1]
        threshold = max(eligible, key=lambda x: x['coverage'])['threshold'] if eligible else 1.01
        timing_runs = []
        for movement_threshold in sorted(set([threshold, .85, .9, .95, .975])):
            if movement_threshold < threshold: continue
            for width in [.0, .04, .08, .12, .16, .2]:
                r = report(clips, temp, penalty, threshold, timing_width=width, movement_threshold=movement_threshold)
                timing_runs.append({'width_sec': width, 'movement_threshold': movement_threshold, **r['aggregate']})
        usable = [r for r in timing_runs if r['scored_eligible_moves'] >= 10 and
                  r['eligible_move_accuracy_at_0.2s'] >= .95]
        selected_gate = max(usable, key=lambda x: x['scored_eligible_moves']) if usable else None
        timing_width = selected_gate['width_sec'] if usable else -1.
        movement_threshold = selected_gate['movement_threshold'] if usable else 1.01
        baseline_runs = []
        for distance in [.1, .2, .3, .4, .6, .8]:
            for window in [1, 3, 5, 9, 13]:
                for cost in [0., 2.]:
                    r = report(clips, 1., cost, 1.01, {'distance': distance, 'window': window})
                    baseline_runs.append({'distance': distance, 'window': window, 'penalty': cost, 'report': r})
        baseline = max(baseline_runs, key=lambda x: x['report']['aggregate']['event_f1'])
        checkpoints = []
        for i, path in enumerate(args.checkpoints):
            dest = args.package/f'model_{i+1}.pt'
            exported = torch.load(path, map_location='cpu', weights_only=True)
            exported['training'] = {k: v for k, v in exported['training'].items()
                                    if k not in ('data', 'output', 'device', 'resume')}
            torch.save(exported, dest)
            checkpoints.append({'file': dest.name, 'sha256': sha256(dest)})
        deployment = {'schema_version': 1, 'feature_version': 1, 'training_fps': 25.,
            'pose_backbone': json.loads((args.data/'manifest.json').read_text()).get('pose_backbone'),
            'temperature': temp, 'switch_penalty': penalty, 'accept_threshold': threshold,
            'max_timing_width_sec': timing_width,
            'movement_accept_threshold': movement_threshold,
            'automatic_ik_enabled': bool(usable),
            'move_validation_rule': 'At least 10 scored validation proposals, >=95% correct contacts/supports and both boundaries within 0.2s',
            'validation_accuracy_target': .97, 'checkpoints': checkpoints,
            'manifest_sha256': sha256(args.data/'manifest.json'),
            'baseline': {k: baseline[k] for k in ('distance', 'window', 'penalty')},
            'scope': 'Offline, one climber, reviewed hold map; unseen people on the same two routes',
            'release_definition': 'First frame no longer in contact',
            'arrival_definition': 'First frame in contact',
            'confidence_warning': 'Validation calibration is not a guarantee on a new wall.'}
        (args.package/'deployment.json').write_text(json.dumps(deployment, indent=2)+'\n')
        validation = {'calibration': calibration, 'coverage_curve': coverage, 'timing_gate_search': timing_runs,
            'decoder_search': [{'penalty': r['penalty'], **r['report']['aggregate']} for r in decoded_runs],
            'baseline_selection_metric': 'validation_event_f1',
            'baseline_search': [{**{k: r[k] for k in ('distance', 'window', 'penalty')},
                                 **r['report']['aggregate']} for r in baseline_runs],
            'model': report(clips, temp, penalty, threshold, timing_width=timing_width,
                            movement_threshold=movement_threshold), 'baseline': baseline['report']}
        (args.package/'validation_report.json').write_text(json.dumps(validation, indent=2)+'\n')
        shutil.copyfile(args.data/'manifest.json', args.package/'dataset_manifest.json')
        print(json.dumps({'deployment': deployment, 'validation': validation['model']['aggregate'],
                          'baseline': baseline['report']['aggregate']}, indent=2), flush=True)
    else:
        deployment = json.loads((args.package/'deployment.json').read_text())
        if sha256(args.data/'manifest.json') != deployment['manifest_sha256']:
            raise ValueError('Dataset changed after calibration.')
        paths = [args.package/c['file'] for c in deployment['checkpoints']]
        for c, path in zip(deployment['checkpoints'], paths):
            if sha256(path) != c['sha256']: raise ValueError('Checkpoint changed after calibration.')
        clips = clips_for(args.data, 'test'); model_logits(paths, clips, args.device)
        result = {'selection': 'Configuration locked on validation before held-out evaluation',
            'deployment_sha256': sha256(args.package/'deployment.json'),
            'model': report(clips, deployment['temperature'], deployment['switch_penalty'], deployment['accept_threshold'],
                            timing_width=deployment['max_timing_width_sec'],
                            movement_threshold=deployment['movement_accept_threshold']),
            'baseline': report(clips, 1., deployment['baseline']['penalty'], 1.01, deployment['baseline'])}
        (args.package/'test_report.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: result[k]['aggregate'] for k in ('model', 'baseline')}, indent=2), flush=True)


if __name__ == '__main__': main()
