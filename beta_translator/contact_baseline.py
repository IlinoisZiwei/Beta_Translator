"""Nonlearned nearest-hold association with nominal-state temporal voting."""
import numpy as np
from scipy.ndimage import uniform_filter1d


def geometric_probabilities(distances, radius=.2, window=5):
    """Return smoothed vote fractions, not calibrated contact probabilities."""
    distances = np.asarray(distances, dtype=np.float32)
    t, limbs, holds = distances.shape
    selected = distances.argmin(axis=-1)
    nearest = distances.min(axis=-1)
    radii = np.array([radius, radius, max(.1, radius*.5), max(.1, radius*.5)])
    selected[nearest > radii] = holds
    votes = np.zeros((t, limbs, holds+2), np.float32)
    np.put_along_axis(votes, selected[..., None], 1., axis=-1)
    if window > 1:
        votes = uniform_filter1d(votes, int(window), axis=0, mode='nearest')
    votes = np.maximum(votes, 1e-4)
    return votes / votes.sum(axis=-1, keepdims=True)
