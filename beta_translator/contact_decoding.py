"""Temporal decoding and conservative review metadata for automated contacts."""
import numpy as np
from scipy.special import softmax

from .contact_evaluation import segments
from .contact_features import LIMBS


def decode_probabilities(probabilities, switch_penalty=2.):
    """Viterbi decode; changing holds costs more than entering/leaving free motion."""
    t, limbs, states = probabilities.shape
    output = np.empty((t, limbs), np.int64)
    cost = np.full((states, states), switch_penalty, np.float32)
    np.fill_diagonal(cost, 0)
    free = states-2
    cost[free, :] *= .5; cost[:, free] *= .5
    for limb in range(limbs):
        logp = np.log(np.maximum(probabilities[:, limb], 1e-9))
        current = logp[0]
        back = np.zeros((t, states), np.int64)
        for frame in range(1, t):
            candidates = current[:, None] - cost
            back[frame] = candidates.argmax(axis=0)
            current = candidates.max(axis=0) + logp[frame]
        output[-1, limb] = current.argmax()
        for frame in range(t-1, 0, -1):
            output[frame-1, limb] = back[frame, output[frame, limb]]
    return output


def ensemble_probabilities(logits, temperature=1.):
    if not logits or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Supply logits and a positive calibration temperature.')
    return np.mean([softmax(x/temperature, axis=-1) for x in logits], axis=0)


def boundary_ranges(probabilities, old, new, release, arrival, before, after):
    """Model ambiguity ranges conditional on old -> free -> new, not coverage guarantees."""
    free = probabilities.shape[-1]-2
    logp = np.log(np.maximum(probabilities[before:after+1][:, [old, free, new]], 1e-9))
    cumulative = np.pad(np.cumsum(logp, axis=0), ((1, 0), (0, 0)))
    pairs, scores = [], []
    radius = max(release-before, after-arrival)
    for r in range(max(before+1, release-radius), min(after-1, release+radius)+1):
        for a in range(max(r+1, arrival-radius), min(after, arrival+radius)+1):
            i, j = r-before, a-before
            scores.append(cumulative[i, 0] + cumulative[j, 1]-cumulative[i, 1]
                          + cumulative[-1, 2]-cumulative[j, 2])
            pairs.append((r, a))
    if not pairs:
        return [release, release], [arrival, arrival], False
    pairs = np.asarray(pairs); weights = softmax(scores)
    result = []
    for axis, chosen in enumerate((release, arrival)):
        values = np.unique(pairs[:, axis])
        mass = np.array([weights[pairs[:, axis] == value].sum() for value in values])
        cdf = np.cumsum(mass)
        low = int(values[min(np.searchsorted(cdf, .05), len(values)-1)])
        high = int(values[min(np.searchsorted(cdf, .95), len(values)-1)])
        result.append([min(low, chosen), max(high, chosen)])
    interior = result[0][0] > before+1 and result[1][1] < after
    return result[0], result[1], interior


def export_predictions(probabilities, decoded, hold_ids, fps, threshold, observed=None,
                       max_timing_width_sec=.16, movement_threshold=None):
    """Keep inferred candidates separate from accepted labels; never call unknown free."""
    if len(hold_ids) != probabilities.shape[-1]-2 or len(set(hold_ids)) != len(hold_ids):
        raise ValueError('Hold IDs must be unique and match model candidates.')
    names = list(hold_ids) + ['free', 'wall_contact']
    observed = np.ones(decoded.shape, bool) if observed is None else np.asarray(observed, bool)
    confidence = np.take_along_axis(probabilities, decoded[..., None], axis=-1)[..., 0]
    accepted = (confidence >= threshold) & observed
    movement_threshold = threshold if movement_threshold is None else movement_threshold
    movement_accepted = (confidence >= movement_threshold) & observed
    rows, events, moves = [], [], []
    for limb, name in enumerate(LIMBS):
        for frame in range(len(decoded)):
            state = names[decoded[frame, limb]]
            rows.append(dict(frame=frame, time_sec=frame/fps, limb=name,
                candidate_hold=state, hold_id=state if accepted[frame, limb] else 'unknown',
                confidence=float(confidence[frame, limb]),
                status='accepted_model' if accepted[frame, limb] else 'review_required'))
        contact_segments = [s for s in segments(decoded[:, limb]) if s[0] < len(hold_ids)]
        for state, start, end in contact_segments:
            events.append(dict(limb=name, hold_id=names[state], start_frame=start, end_frame=end,
                start_sec=start/fps, end_sec=(end+1)/fps,
                confidence=float(np.quantile(confidence[start:end+1, limb], .1)),
                accepted_fraction=float(accepted[start:end+1, limb].mean()),
                requires_review=not bool(accepted[start:end+1, limb].all())))
        for previous, following in zip(contact_segments[:-1], contact_segments[1:]):
            old, start, end = previous; new, arrival, finish = following
            if old == new: continue
            release = end+1
            moving_span = decoded[release:arrival, limb]
            # A geometric IK schedule requires a release phase and three verified supports.
            direct_jump = arrival <= release or (moving_span != len(hold_ids)).any()
            radius = max(3, int(round(.24*fps)))
            before, after = max(start, end-radius), min(finish, arrival+radius)
            uncertain = np.flatnonzero(~accepted[before:after+1, limb])
            release_range, arrival_range, interior = boundary_ranges(
                probabilities[:, limb], old, new, release, arrival, before, after)
            timing_ok = (not direct_jump and interior and
                         max(release_range[1]-release_range[0], arrival_range[1]-arrival_range[0])
                         <= max_timing_width_sec*fps + 1e-6)
            old_anchor = confidence[before:min(end+1, before+3), limb]
            new_anchor = confidence[max(arrival, after-2):after+1, limb]
            phase_ok = (len(old_anchor) > 0 and len(new_anchor) > 0 and
                        min(float(old_anchor.min()), float(new_anchor.min())) >= movement_threshold and
                        arrival > release and
                        float(probabilities[release:arrival, limb, len(hold_ids)].max()) >= movement_threshold and
                        observed[before:after+1, limb].all())
            supports, support_verified = {}, True
            for other, support in enumerate(LIMBS):
                if other == limb: continue
                span = decoded[before:after+1, other]
                ok = len(span) > 0 and (span == span[0]).all() and span[0] < len(hold_ids)
                ok = bool(ok and movement_accepted[before:after+1, other].all())
                support_verified &= ok
                supports[support] = names[span[0]] if ok else 'unknown'
            review = bool(direct_jump or not timing_ok or not phase_ok or not support_verified)
            reasons = []
            if direct_jump: reasons.append('No confidently resolved free-motion phase')
            if not timing_ok: reasons.append('Transition timing is insufficiently resolved')
            if not phase_ok: reasons.append('Contact/free/contact evidence needs review')
            if not support_verified: reasons.append('Support contacts are uncertain or change')
            moves.append(dict(moving_limb=name, from_hold=names[old], to_hold=names[new],
                before_frame=before, after_frame=after, release_frame=release, arrival_frame=arrival,
                release_frame_range=release_range, arrival_frame_range=arrival_range,
                timing_range_semantics='Central 90% local path mass; empirical coverage is not guaranteed',
                uncertain_frame_range=[int(uncertain[0]+before), int(uncertain[-1]+before)] if review and len(uncertain) else None,
                support_contacts=supports, supports_maintained=bool(support_verified),
                requires_review=review, eligible_for_ik=not review,
                review_reasons=reasons,
                evidence='Visual model prediction; not physical contact-force confirmation'))
    return rows, events, moves
