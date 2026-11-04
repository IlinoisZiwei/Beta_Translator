"""Conditional metric 3D cameras for one calibrated wall; never infer gravity from it."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares


def project(points, K, R, t):
    """Project wall-local XYZ (cm), supporting batched cameras and complex derivatives."""
    points = np.asarray(points)
    if np.ndim(R) == 2:
        camera = points @ np.asarray(R).T + np.asarray(t)
        pixels = camera @ np.asarray(K).T
    else:
        camera = np.einsum('t...j,tij->t...i', points, R) + np.asarray(t).reshape((-1,) + (1,)*(points.ndim-2) + (3,))
        pixels = camera @ np.asarray(K).T
    return pixels[..., :2] / pixels[..., 2:3], camera[..., 2]


def tilt_rotation(degrees):
    """Wall -> world. Positive tilt: top overhangs toward the climber; world Y is up."""
    a = np.deg2rad(degrees)
    return np.array([[1., 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def upward_force(degrees):
    """Contact resultant / body weight required to oppose world gravity, in wall axes."""
    return tilt_rotation(degrees).T @ np.array([0., 1., 0.])


def fit_camera(xyz, image, training_mask, image_size, principal_x=None, fit_principal_y=False):
    """Zero distortion, square pixels and zero skew are explicit, unmeasured priors.

    The held-out grid was discovered with shared consensus: consistency, not an
    independent physical calibration accuracy measurement.
    """
    xyz, image = np.asarray(xyz, float), np.asarray(image, float)
    mask = np.asarray(training_mask, bool)
    width, height = image_size
    cx = width / 2 if principal_x is None else float(principal_x)
    cy = height / 2
    def unpack(q):
        k = np.array([[np.exp(q[0]), 0., cx], [0., np.exp(q[0]), q[7] if fit_principal_y else cy], [0., 0., 1.]])
        return k, q[1:4], q[4:7]
    K = np.array([[800., 0, cx], [0, 800., cy], [0, 0, 1.]])
    ok, r, t = cv2.solvePnP(xyz[mask], image[mask], K, None)
    if not ok:
        raise ValueError('Planar camera initialization failed.')
    q0 = np.r_[np.log(800), r.ravel(), t.ravel(), [cy] if fit_principal_y else []]
    def residual(q):
        k, r, t = unpack(q)
        return (cv2.projectPoints(xyz[mask], r, t, k, None)[0][:, 0] - image[mask]).ravel()
    lo = np.r_[np.log(150), [-np.inf]*6, [0.] if fit_principal_y else []]
    hi = np.r_[np.log(6000), [np.inf]*6, [float(height)] if fit_principal_y else []]
    fit = least_squares(residual, q0, bounds=(lo, hi), max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-9)
    K, r, t = unpack(fit.x)
    R = cv2.Rodrigues(r)[0]
    px, z = project(xyz, K, R, t)
    error = np.linalg.norm(px-image, axis=1)
    center = -R.T @ t
    if not fit.success or np.min(z) <= 0 or center[2] <= 0:
        raise ValueError('Camera must converge, see the wall in front, and be on its outward side.')
    return dict(K=K.tolist(), R=R.tolist(), t_cm=t.tolist(), camera_center_cm=center.tolist(),
                fitting_rmse_px=float(np.sqrt(np.mean(error[mask]**2))),
                reserved_rmse_px=float(np.sqrt(np.mean(error[~mask]**2))),
                reserved_p95_px=float(np.percentile(error[~mask], 95)),
                principal_x_px=cx, fit_principal_y=fit_principal_y,
                measured_intrinsics=False, measured_wall_tilt=False,
                distortion='Assumed zero; resized/cropped video intrinsics not supplied.',
                solver_success=bool(fit.success))


def build_cameras(project_root, frames):
    project_root = Path(project_root)
    root = project_root/'outputs/trial3/state'
    data = json.loads((root/'white_wall_grid_points.json').read_text())
    obs = json.loads((project_root/'data/trial3_wall_grid.json').read_text())
    xy = np.asarray(data['grid_column_row']) * [obs['spacing_cm'], -obs['spacing_cm']]
    xyz = np.c_[xy, np.zeros(len(xy))]
    pixels = np.asarray(data['image_xy'])
    train = ~np.asarray(data['reserved_from_final_fit'])
    primary = fit_camera(xyz, pixels, train, obs['image_size'])
    # Same planar image can admit different intrinsics: expose rather than hide it.
    alternatives = [fit_camera(xyz, pixels, train, obs['image_size'], principal_x=x, fit_principal_y=True)
                    for x in [obs['image_size'][0]*.45, obs['image_size'][0]*.55]]
    K = np.asarray(primary['K'])
    with np.load(root/'alignment_cache.npz', allow_pickle=False) as z:
        indices = {int(f):i for i,f in enumerate(z['frames'])}
        matrices = np.array([z['matrices'][indices[int(f)]] for f in frames])
        if any(str(z['statuses'][indices[int(f)]]) != 'ok' for f in frames):
            raise ValueError('An experiment frame has invalid wall tracking.')
    rotations, translations, residuals = [], [], []
    for H in matrices:
        current = cv2.perspectiveTransform(pixels.astype(float)[None], H)[0]
        ok, r, t = cv2.solvePnP(xyz[train], current[train], K, None)
        if not ok: raise ValueError('Per-frame planar pose failed.')
        R = cv2.Rodrigues(r)[0]
        pred, depth = project(xyz, K, R, t.ravel())
        if depth.min() <= 0: raise ValueError('Wall behind camera.')
        rotations.append(R); translations.append(t.ravel())
        residuals.append(float(np.sqrt(np.mean(np.sum((pred-current)**2, axis=1)))))
    primary.update(alternatives=alternatives, frame_pose_note='Tracked wall homographies supply correspondences; not independent multi-view depth observations.',
                   tilt_identifiable=False, grid_point_count=len(xyz), reserved_count=int((~train).sum()))
    return primary, np.asarray(rotations), np.asarray(translations), np.asarray(residuals)


def contact_span_certificate(targets, known, body):
    """Triangle-inequality impossibility certificate, invariant under any wall tilt.

    Rigid torso shoulder/hip root separations and fixed limb lengths provide a
    conservative upper bound valid in both 2D and 3D. Depth cannot shorten the
    measured Euclidean separation between fixed contacts on the same plane.
    """
    roots = np.array([[-body['shoulder_w']/2, body['torso']], [body['shoulder_w']/2, body['torso']],
                      [-body['hip_w']/2, 0], [body['hip_w']/2, 0]])
    length = np.array([body['upper_arm']+body['forearm']]*2 + [body['thigh']+body['shank']]*2)
    rows = []
    for i in range(4):
        for j in range(i+1, 4):
            if known[i] and known[j]:
                distance = float(np.linalg.norm(np.asarray(targets[i])-targets[j]))
                maximum = float(length[i]+length[j]+np.linalg.norm(roots[i]-roots[j]))
                rows.append(dict(limb_i=i, limb_j=j, distance_cm=distance, maximum_span_cm=maximum,
                                 excess_cm=max(0.,distance-maximum), min_possible_max_endpoint_error_cm=max(0.,distance-maximum)/2))
    return rows
