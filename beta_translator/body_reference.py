"""Pinned source rotations and independent verification; no simulator import."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from . import body_kinematics as arms

@lru_cache(maxsize=1)
def source_reference():
    return json.loads((Path(__file__).resolve().parents[1]/'data/trial2_stage1_chain_reference.json').read_text())


def source_curve(x, knots, values):
    """Two/three-knot SimmSpline special case, with endpoint tangent extrapolation.

    OpenSim 4.5's two-point curve is linear; its three-point end conditions
    give a quadratic. No general SimmSpline implementation is claimed here.
    Real-valued branch selection preserves complex-step derivatives.
    """
    knots, values = np.asarray(knots), np.asarray(values)
    if len(knots) not in (2, 3):
        raise ValueError('Only the verified two/three-knot source curves are supported')
    coef = np.polyfit(knots, values, len(knots)-1)
    y = np.polyval(coef, x)
    derivative = np.polyder(coef)
    for end, mask in [(0, np.real(x)<knots[0]), (-1, np.real(x)>knots[-1])]:
        y = np.where(mask, values[end]+(x-knots[end])*np.polyval(derivative, knots[end]), y)
    return y


def coupled_arm_frames(q):
    """Source rotational chain from scapphant to hand; project lengths retained."""
    A, F, _ = arms.arm_frames(q)
    _, _, _, _, s, d, f = np.moveaxis(q, -1, 0)
    ref = source_reference()
    joints = {j['name']:j for j in ref['joints']}
    couplers = {j['coordinate']:j for j in ref['couplers']}
    def rot(angle, axis):
        a = np.asarray(axis); a = a/np.linalg.norm(a)
        R = arms.axis_rotation(angle, a)
        return arms.OSIM_TO_BODY@R@arms.OSIM_TO_BODY.T
    carpal = joints['radiocarpal']['rotations']
    distal = joints['wrist_hand']['rotations']
    rc = rot(d, carpal[0]['axis'])@rot(source_curve(f, carpal[1]['x'], carpal[1]['y']), carpal[1]['axis'])
    dc = couplers['wrist_hand_r1']; fc = couplers['wrist_hand_r3']
    hand = rot(source_curve(d, dc['x'], dc['y']), distal[0]['axis'])@rot(source_curve(f, fc['x'], fc['y']), distal[1]['axis'])
    H = F@rot(s, joints['radioulnar']['rotations'][0]['axis'])@rc@hand
    return A, F, H


def stage1_kinematics(q, lengths, height):
    """Correct distal hand frames without stretching or repositioning arm bones."""
    g = arms.kinematics(q, lengths, height)
    q = np.atleast_2d(q); torso = g['orientations'][:, 0]
    for side in range(2):
        _, _, H = coupled_arm_frames(q[:, 6+7*side:13+7*side])
        if side == 0:
            H = arms.MIRROR@H@arms.MIRROR
        H = torso@H
        g['orientations'][:, 3+3*side] = H
        g['hand_tips'][:, side] = g['joints'][:, side, 2]-.035*height*H[:, :, 1]
        g['palm_contacts'][:, side] = g['hand_tips'][:, side]-.012*height*H[:, :, 2]
    return g


def verify_chain(samples=128):
    """Independent source-parameter matrix composition (SciPy rotation oracle).

    Verifies the ROTATIONAL subchain; translational source offsets are explicitly
    excluded. Does not substitute for an OpenSim simulation or human validation.
    """
    ref = source_reference(); rng = np.random.default_rng(5810)
    bounds = np.deg2rad(arms.ARM_LIMITS_DEG)
    qs = rng.uniform(bounds[:, 0], bounds[:, 1], (samples, 7))
    # Independent three-point polynomial interpolation using a Vandermonde solve.
    def curve(x, spec):
        if spec['function'] == 'LinearFunction':
            return spec['coefficients'][0]*x+spec['coefficients'][1]
        xx, yy = np.array(spec['x']), np.array(spec['y'])
        coef = np.linalg.solve(np.vander(xx), yy)
        y = np.polyval(coef, x)
        for k in (0, -1):
            if (k == 0 and x < xx[k]) or (k == -1 and x > xx[k]):
                y = yy[k]+(x-xx[k])*np.polyval(np.polyder(coef), xx[k])
        return y
    expected = []
    for q in qs:
        coords = dict(zip(arms.ARM_NAMES, q))
        for c in ref['couplers']:
            coords[c['coordinate']] = curve(coords[c['independent']], c)
        R = np.eye(3); frames = {}
        for j in ref['joints']:
            assert not np.any(j['orientation_in_parent']) and not np.any(j['orientation'])
            for a in j['rotations']:
                axis = np.asarray(a['axis']); axis /= np.linalg.norm(axis)
                R = R@Rotation.from_rotvec(axis*curve(coords[a['coordinate']], a)).as_matrix()
            frames[j['child']] = arms.OSIM_TO_BODY@R@arms.OSIM_TO_BODY.T
        expected.append([frames[n] for n in ['humerus', 'ulna', 'hand']])
    expected = np.asarray(expected)
    actual = np.stack(coupled_arm_frames(qs), axis=1)
    old = np.stack(arms.arm_frames(qs), axis=1)
    error = np.max(abs(actual-expected))
    old_angles = arms.rotation_steps_deg(np.stack([old[:, 2], expected[:, 2]], axis=0), 1)
    return dict(source_sha256=ref['source_sha256'], random_poses=samples,
        rotation_matrix_max_absolute_error=float(error), rotational_chain_pass=bool(error<1e-10),
        previous_hand_orientation_max_difference_deg=float(np.max(old_angles)),
        fixed_length_reduction=True, source_translation_offsets_imported=False,
        source_rest_offsets_m={j['name']:j['location_in_parent'] for j in ref['joints']},
        source_girdle_imported=False, opensim_engine_crosscheck=False,
        conclusion='Source rotational subchain checked independently; omitted translations and girdle motion remain a declared reduction.')
