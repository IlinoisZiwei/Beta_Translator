"""Observed-pixel wall reconstruction and multi-scale hold candidate detection."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def person_mask(points, scores, shape, padding=55):
    """Conservatively exclude the tracked climber, including gaps between limbs."""
    mask = np.zeros(shape[:2], np.uint8)
    ids = list(range(23)) + [95, 99, 103, 107, 111, 116, 120, 124, 128, 132]
    p = points[ids]
    valid = np.isfinite(p).all(axis=1) & (scores[ids] > 1.0)
    if valid.sum() < 8:
        return np.ones(shape[:2], np.uint8)
    hull = cv2.convexHull(np.rint(p[valid]).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding*2+1,)*2))


def reconstruct_wall(video, pose_cache, alignment_cache, frames, reference, destination, person_padding=55):
    """Replace masked reference pixels only with visible pixels from source frames.

    Transform direction is reference -> source. Unobserved pixels remain explicit
    holes; this function does not synthesize or inpaint hold geometry.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with open(video, 'rb') as stream:
        video_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    with np.load(pose_cache, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata'].item()))
        if metadata.get('video_sha256') != video_hash:
            raise ValueError('Pose cache belongs to another video.')
        points, scores = data['keypoints'], data['scores']
    with np.load(alignment_cache, allow_pickle=False) as data:
        if str(data['video_sha256'].item()) != video_hash:
            raise ValueError('Alignment cache belongs to another video.')
        settings = json.loads(str(data['settings'].item()))
        if settings.get('reference_frame_number', 0) != reference:
            raise ValueError('Reconstruction and alignment must share a reference frame.')
        transforms = {int(f): (H, str(s)) for f, H, s in
                      zip(data['frames'], data['matrices'], data['statuses'])}
    cap = cv2.VideoCapture(str(video))
    ordered = [reference] + [f for f in dict.fromkeys(frames) if f != reference]
    result = provenance = missing = None
    diagnostics = []
    try:
        for index in ordered:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or index not in transforms or transforms[index][1] != 'ok':
                raise ValueError(f'No verified source frame {index}.')
            h, w = frame.shape[:2]
            exclusion = person_mask(points[index], scores[index], frame.shape, padding=person_padding)
            inverse = np.linalg.inv(transforms[index][0])
            warped = cv2.warpPerspective(frame, inverse, (w, h))
            valid = cv2.warpPerspective(1-exclusion, inverse, (w, h), flags=cv2.INTER_NEAREST)
            valid = cv2.erode(valid, np.ones((5, 5), np.uint8)).astype(bool)
            if result is None:
                result = warped.copy()
                provenance = np.full((h, w), -1, np.int32)
                missing = np.ones((h, w), bool)
                reference_occlusion = ~valid
            take = missing & valid
            result[take] = warped[take]
            provenance[take] = index
            missing[take] = False
            diagnostics.append({'source_frame': index, 'used_pixels': int(take.sum()),
                                'unresolved_pixels': int(missing.sum())})
            cv2.imwrite(str(destination / f'partial_after_{index:04d}.jpg'), result)
    finally:
        cap.release()
    # Gray areas mean unavailable evidence, never a hallucinated wall surface.
    result[missing] = (128, 128, 128)
    cv2.imwrite(str(destination / 'reconstructed_wall.png'), result)
    cv2.imwrite(str(destination / 'unresolved_mask.png'), missing.astype(np.uint8)*255)
    np.savez_compressed(destination / 'pixel_provenance.npz', source_frame=provenance,
                        reference_occlusion=reference_occlusion, unresolved=missing)
    report = {'reference_frame_number': reference, 'source_frames': ordered,
              'sources': diagnostics, 'unresolved_pixels': int(missing.sum()),
              'method': 'Reference-image coordinates; observed donor pixels only; convex-hull person exclusion.',
              'limitations': 'One global homography per source; parallax and surface-depth errors remain possible.'}
    (destination / 'reconstruction.json').write_text(json.dumps(report, indent=2)+'\n')
    return result, report


def compact_mask(mask, offset=(0, 0)):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
    return {'bbox': [x0+offset[0], y0+offset[1], x1+offset[0], y1+offset[1]],
            'mask': mask[y0:y1, x0:x1].astype(bool), 'area': int(len(xs))}


def overlap(a, b):
    lo = np.maximum(a['bbox'][:2], b['bbox'][:2])
    hi = np.minimum(a['bbox'][2:], b['bbox'][2:])
    if (hi <= lo).any():
        return 0.0, 0.0
    ax, ay = lo-np.array(a['bbox'][:2]); bx, by = lo-np.array(b['bbox'][:2])
    width, height = hi-lo
    inter = int(np.logical_and(a['mask'][ay:ay+height, ax:ax+width],
                               b['mask'][by:by+height, bx:bx+width]).sum())
    return inter / max(1, a['area']+b['area']-inter), inter / max(1, min(a['area'], b['area']))


def deduplicate(candidates, iou=.35, containment=.7):
    """Suppress duplicate masks, preferring unclipped detections before score."""
    kept = []
    for item in sorted(candidates, key=lambda c: (not c['clipped'], c['score']), reverse=True):
        if any((overlap(item, k)[0] >= iou or overlap(item, k)[1] >= containment) for k in kept):
            continue
        kept.append(item)
    return kept


def detect_candidates(image, predictor, tiled=False, tile_size=768, stride=576):
    """Run whole-image inference and optional overlapping tiles at native pixels."""
    h, w = image.shape[:2]
    regions = [(0, 0, w, h, 'whole')]
    if tiled:
        xs = sorted(set(list(range(0, max(1, w-tile_size+1), stride)) + [max(0, w-tile_size)]))
        ys = sorted(set(list(range(0, max(1, h-tile_size+1), stride)) + [max(0, h-tile_size)]))
        regions += [(x, y, min(w, x+tile_size), min(h, y+tile_size), f'tile_{x}_{y}') for y in ys for x in xs]
    candidates = []
    for x0, y0, x1, y1, source in regions:
        result = predictor(image[y0:y1, x0:x1])['instances'].to('cpu')
        masks, scores, classes = result.pred_masks.numpy(), result.scores.numpy(), result.pred_classes.numpy()
        for mask, score, cls in zip(masks, scores, classes):
            item = compact_mask(mask, (x0, y0))
            if item is None or item['area'] < 12:
                continue
            a, b, c, d = item['bbox']
            clipped = source != 'whole' and ((x0>0 and a<=x0+2) or (y0>0 and b<=y0+2)
                or (x1<w and c>=x1-2) or (y1<h and d>=y1-2))
            item.update(score=float(score), class_id=int(cls), source=source, clipped=clipped)
            candidates.append(item)
        print(source, len(masks), flush=True)
    return deduplicate(candidates), {'raw_candidates': len(candidates), 'inference_regions': len(regions)}


def export_candidates(candidates, image, path):
    """Export compact masks and a readable overlay; IDs are stable within this run."""
    candidates = sorted(candidates, key=lambda c: (c['bbox'][1]+c['bbox'][3], c['bbox'][0]+c['bbox'][2]))
    canvas = image.copy()
    records, masks = [], {}
    for number, item in enumerate(candidates, 1):
        hold = f'H{number:03d}'
        x0, y0, x1, y1 = item['bbox']
        ys, xs = np.nonzero(item['mask'])
        record = {k: v for k, v in item.items() if k != 'mask'}
        record.update(hold_id=hold, cx_px=float(xs.mean()+x0), cy_px=float(ys.mean()+y0))
        records.append(record)
        masks[hold] = item['mask']
        contours, _ = cv2.findContours(item['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c + np.array([[[x0, y0]]]) for c in contours]
        cv2.drawContours(canvas, contours, -1, (0, 220, 255), 1)
        location = (int(record['cx_px'])+3, int(record['cy_px'])-3)
        cv2.putText(canvas, hold, location, cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, hold, location, cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1, cv2.LINE_AA)
    path = Path(path)
    np.savez_compressed(path.with_suffix('.npz'), records=json.dumps(records), **masks)
    cv2.imwrite(str(path.with_suffix('.jpg')), canvas)
    return records


def filter_observed_candidates(candidates, unresolved, min_observed_fraction=.98):
    """Reject detections on reconstruction holes instead of inventing hidden holds."""
    retained = []
    for item in candidates:
        x0, y0, x1, y1 = item['bbox']
        fraction = float(np.count_nonzero(item['mask'] & ~unresolved[y0:y1, x0:x1]) / item['area'])
        if fraction >= min_observed_fraction:
            retained.append({**item, 'observed_pixel_fraction': fraction})
    return retained
