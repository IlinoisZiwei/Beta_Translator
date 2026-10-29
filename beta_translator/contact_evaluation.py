"""Frame, event, timing and selective-automation metrics with explicit denominators."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def segments(sequence, valid=None):
    sequence = np.asarray(sequence)
    if valid is not None:
        sequence = sequence.copy(); sequence[~valid] = -100
    changes = np.r_[0, np.flatnonzero(sequence[1:] != sequence[:-1])+1, len(sequence)]
    return [(int(sequence[a]), int(a), int(b-1)) for a, b in zip(changes[:-1], changes[1:])]


def frame_metrics(probabilities, labels, occluded=None, threshold=.9, decoded=None):
    valid = labels >= 0
    pred = probabilities.argmax(-1) if decoded is None else decoded
    confidence = np.take_along_axis(probabilities, pred[..., None], axis=-1)[..., 0]
    correct = pred == labels
    selected = valid & (confidence >= threshold)
    holds = probabilities.shape[-1]-2
    contact = valid & (labels < holds)
    free = valid & (labels == holds)
    wall = valid & (labels == holds+1)
    metrics = {'labeled_limb_frames': int(valid.sum()),
        'correct_limb_frames': int((correct & valid).sum()),
        'accuracy': float(correct[valid].mean()),
        'contact_accuracy': float(correct[contact].mean()) if contact.any() else None,
        'free_recall': float(correct[free].mean()) if free.any() else None,
        'wall_recall': float(correct[wall].mean()) if wall.any() else None,
        'accepted_limb_frames': int(selected.sum()),
        'accepted_correct': int((selected & correct).sum()),
        'coverage': float(selected.sum()/valid.sum()),
        'accepted_accuracy': float(correct[selected].mean()) if selected.any() else None}
    if occluded is not None:
        for name, mask in [('occluded', occluded), ('visible', ~occluded)]:
            use = valid & mask
            metrics[name+'_frames'] = int(use.sum())
            metrics[name+'_accuracy'] = float(correct[use].mean()) if use.any() else None
    true_probability = probabilities[valid][np.arange(valid.sum()), labels[valid]]
    metrics['nll'] = float(-np.log(np.maximum(true_probability, 1e-8)).mean())
    return metrics


def event_metrics(predicted, labels, holds, fps, iou_threshold=.5):
    """One-to-one limb+hold+IoU matching; timing is conditional on matched events."""
    true_count = predicted_count = matched = 0
    arrival_errors, release_errors = [], []
    for limb in range(4):
        valid = labels[:, limb] >= 0
        truth = [e for e in segments(labels[:, limb], valid) if 0 <= e[0] < holds]
        guess = [e for e in segments(predicted[:, limb], valid) if 0 <= e[0] < holds]
        true_count += len(truth); predicted_count += len(guess)
        if not truth or not guess: continue
        ious = np.zeros((len(truth), len(guess)))
        for i, (h, a, b) in enumerate(truth):
            for j, (k, c, d) in enumerate(guess):
                if h == k:
                    overlap = max(0, min(b, d)-max(a, c)+1)
                    ious[i, j] = overlap / (max(b, d)-min(a, c)+1)
        # Threshold-aware assignment prioritizes match count, then IoU.
        reward = (ious >= iou_threshold).astype(float) * 1000 + ious
        row, col = linear_sum_assignment(-reward)
        for i, j in zip(row, col):
            if ious[i, j] < iou_threshold: continue
            matched += 1
            # Do not score boundaries censored by video ends or invalid labels.
            a, b = truth[i][1:]; c, d = guess[j][1:]
            if a > 0 and valid[a-1]: arrival_errors.append(abs(a-c)/fps)
            if b+1 < len(valid) and valid[b+1]: release_errors.append(abs(b-d)/fps)
    return {'true_events': true_count, 'predicted_events': predicted_count, 'matched_events': matched,
            'precision': matched/predicted_count if predicted_count else 0.,
            'recall': matched/true_count if true_count else 0.,
            'f1': 2*matched/(true_count+predicted_count) if true_count+predicted_count else 0.,
            'arrival_errors_sec': arrival_errors, 'release_errors_sec': release_errors}


def aggregate(reports):
    frames = sum(r['frame']['labeled_limb_frames'] for r in reports)
    accepted = sum(r['frame']['accepted_limb_frames'] for r in reports)
    correct = sum(r['frame']['correct_limb_frames'] for r in reports)
    accepted_correct = sum(r['frame']['accepted_correct'] for r in reports)
    gt = sum(r['event']['true_events'] for r in reports)
    pred = sum(r['event']['predicted_events'] for r in reports)
    match = sum(r['event']['matched_events'] for r in reports)
    result = dict(labeled_limb_frames=frames, accuracy=correct/frames,
                  coverage=accepted/frames, accepted_accuracy=accepted_correct/accepted if accepted else None,
                  true_events=gt, predicted_events=pred, matched_events=match,
                  event_precision=match/pred if pred else 0., event_recall=match/gt if gt else 0.,
                  event_f1=2*match/(gt+pred) if gt+pred else 0.,
                  macro_video_accuracy=float(np.mean([r['frame']['accuracy'] for r in reports])))
    result['motion_proposals'] = sum(r.get('motion_proposals', 0) for r in reports)
    result['ik_eligible_proposals'] = sum(r.get('ik_eligible_proposals', 0) for r in reports)
    result['proposal_eligibility_fraction'] = (result['ik_eligible_proposals']/result['motion_proposals']
                                              if result['motion_proposals'] else 0.)
    scored = sum(r.get('eligible_move_check', {}).get('scored', 0) for r in reports)
    correct_moves = sum(r.get('eligible_move_check', {}).get('correct', 0) for r in reports)
    result['scored_eligible_moves'] = scored
    result['correct_eligible_moves'] = correct_moves
    result['eligible_move_accuracy_at_0.2s'] = correct_moves/scored if scored else None
    for kind in ('arrival', 'release'):
        errors = np.array([e for r in reports for e in r['event'][kind+'_errors_sec']])
        result[kind+'_matched_boundaries'] = len(errors)
        result[kind+'_mae_sec'] = float(errors.mean()) if len(errors) else None
        result[kind+'_median_sec'] = float(np.median(errors)) if len(errors) else None
        result[kind+'_within_0.2s'] = float((errors <= .2+1e-7).mean()) if len(errors) else None
    return result


def eligible_move_check(moves, labels, fps, holds):
    """Score automatic proposals against held-out states and both boundaries, once each."""
    limbs = ('left_hand', 'right_hand', 'left_foot', 'right_foot')
    truth = []
    for limb in range(4):
        contacts = [s for s in segments(labels[:, limb]) if 0 <= s[0] < holds]
        for (old, start, end), (new, arrival, finish) in zip(contacts[:-1], contacts[1:]):
            if old != new and arrival > end+1 and (labels[end+1:arrival, limb] == holds).all():
                truth.append((limb, old, new, end+1, arrival))
    scored = correct = 0; used = set()
    for move in moves:
        if not move['eligible_for_ik']: continue
        a, b = move['before_frame'], move['after_frame']
        if (labels[a:b+1] < 0).any(): continue
        scored += 1
        limb = limbs.index(move['moving_limb'])
        old, new = int(move['from_hold'][1:]), int(move['to_hold'][1:])
        matches = [(abs(r-move['release_frame'])+abs(t-move['arrival_frame']), i)
                   for i, (l, h, k, r, t) in enumerate(truth) if i not in used and (l, h, k) == (limb, old, new)
                   and abs(r-move['release_frame']) <= .2*fps+1e-6 and abs(t-move['arrival_frame']) <= .2*fps+1e-6]
        supports_ok = all((labels[a:b+1, limbs.index(name)] == int(hold[1:])).all()
                          for name, hold in move['support_contacts'].items())
        if matches and supports_ok and labels[a, limb] == old and labels[b, limb] == new:
            used.add(min(matches)[1]); correct += 1
    return {'scored': scored, 'correct': correct, 'unscorable': sum(m['eligible_for_ik'] for m in moves)-scored}
