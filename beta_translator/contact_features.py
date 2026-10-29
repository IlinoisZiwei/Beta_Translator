"""Hold-order invariant whole-body geometry for learned contact association."""
import numpy as np
from scipy.ndimage import median_filter

FEATURE_VERSION = 1
LIMBS = ('left_hand', 'right_hand', 'left_foot', 'right_foot')
# COCO-WholeBody: wrist + five fingertips; ankle + toes/heel repeated to six.
LANDMARKS = ((9, 95, 99, 103, 107, 111), (10, 116, 120, 124, 128, 132),
             (15, 17, 18, 19, 17, 18), (16, 20, 21, 22, 20, 21))
BODY = tuple(range(5, 17))


def build_features(points, scores, hold_boxes, fps):
    """Return [frame, limb, hold+free+wall, feature] and raw geometric distances.

    Points and hold boxes must share a stabilized pixel coordinate system.
    The network uses relative geometry, never hold IDs or absolute image positions.
    """
    points, scores, boxes = (np.asarray(x, dtype=np.float32) for x in (points, scores, hold_boxes))
    if points.ndim != 3 or points.shape[1:] != (133, 2) or scores.shape != points.shape[:2]:
        raise ValueError('Expected COCO-WholeBody-133 coordinates and raw SimCC scores.')
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes) or not np.isfinite(boxes).all():
        raise ValueError('Expected finite nonempty XYXY hold boxes.')
    if (boxes[:, 2:] <= boxes[:, :2]).any() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid hold bounds or frame rate.')
    t, h = len(points), len(boxes)
    shoulder = np.mean(points[:, [5, 6]], axis=1)
    pelvis = np.mean(points[:, [11, 12]], axis=1)
    torso = np.linalg.norm(shoulder-pelvis, axis=1)
    usable = torso[np.isfinite(torso) & (torso > 5)]
    scale = float(np.median(usable)) if len(usable) else 100.0
    scale = max(scale, 10.0)
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    all_features, all_distances = [], []
    for limb, indices in enumerate(LANDMARKS):
        p = points[:, indices]
        s = np.nan_to_num(scores[:, indices], nan=0) / 5
        anchor = np.mean(p[:, 1:], axis=1)
        smooth = median_filter(np.nan_to_num(anchor), size=(3, 1))
        velocity = np.gradient(smooth, axis=0) * fps / scale if t > 1 else np.zeros_like(smooth)
        acceleration = np.gradient(velocity, axis=0) / fps if t > 1 else np.zeros_like(velocity)
        delta = (p[:, None] - center[None, :, None]) / scale
        outside = np.maximum(np.maximum(boxes[None, :, None, :2] - p[:, None],
                                        p[:, None] - boxes[None, :, None, 2:]), 0)
        distance = np.linalg.norm(outside, axis=-1) / scale
        # Low-score coordinates must not become a spurious zero-distance observation.
        valid = np.isfinite(p).all(axis=-1) & (s > .1)
        distance = np.where(valid[:, None], distance, 10)
        nearest = distance.min(axis=2)
        all_distances.append(nearest)
        context = np.stack([np.sort(nearest, axis=1)[:, 0],
                            np.sort(nearest, axis=1)[:, min(1, h-1)],
                            np.linalg.norm(velocity, axis=1),
                            np.linalg.norm(acceleration, axis=1)], axis=-1)
        common = np.concatenate([s, velocity, context, np.tile(np.eye(4)[limb], (t, 1))], axis=-1)
        body = (points[:, None, BODY] - center[None, :, None]) / scale
        size = np.broadcast_to((boxes[:, 2:]-boxes[:, :2])[None]/scale, (t, h, 2))
        spatial = np.concatenate([delta.reshape(t, h, 12), distance, body.reshape(t, h, 24), size], axis=-1)
        # Special states share the observed limb context but have no hold geometry.
        spatial = np.pad(spatial, ((0, 0), (0, 2), (0, 0)))
        types = np.zeros((t, h+2, 3), np.float32)
        types[:, :h, 0] = 1; types[:, h, 1] = 1; types[:, h+1, 2] = 1
        features = np.concatenate([spatial, np.broadcast_to(common[:, None], (t, h+2, common.shape[-1])), types], axis=-1)
        all_features.append(np.clip(np.nan_to_num(features, nan=0, posinf=10, neginf=-10), -10, 10))
    return np.stack(all_features, axis=1).astype(np.float32), np.stack(all_distances, axis=1)
