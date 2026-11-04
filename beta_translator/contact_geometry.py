"""Optional contact geometry, separate from anatomical ankle endpoints.

Offsets are fixed vectors in the model's world axes, so their pose Jacobian is
the endpoint Jacobian. They do not add a foot joint or a foot mass model.
"""
import numpy as np


def contact_points(ends, schedule, indices=slice(None)):
    offsets = schedule.get('contact_offsets_cm')
    return ends if offsets is None else ends + np.asarray(offsets)[indices]


def support_ids(schedule):
    return schedule.get('support_ids', schedule['labels'])


def validate_contact_geometry(schedule, n):
    if 'contact_offsets_cm' not in schedule:
        return
    offsets = np.asarray(schedule['contact_offsets_cm'])
    if offsets.shape != (n, 4, 2) or not np.isfinite(offsets).all():
        raise ValueError('Contact offsets must be finite (frames, 4, 2) vectors.')
    ids = np.asarray(schedule['support_ids'])
    if ids.shape != (n, 4) or np.any(schedule['known'] & (ids == '')):
        raise ValueError('Every located contact needs a support identity.')
    fit = np.asarray(schedule['fit_weight'])
    if fit.shape != (n, 4) or not np.isfinite(fit).all() or np.any((fit < 0) | (fit > 1)):
        raise ValueError('Invalid contact fitting weights.')
