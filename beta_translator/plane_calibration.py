"""Metric calibration of one wall plane from a user-specified square bolt grid."""
import json
from pathlib import Path

import cv2
import numpy as np

from .project_io import file_sha256, load_reviewed_holds, read_frame
from .trial_pipeline import checkpoint, valid_cache, write_json


def project(points, matrix):
    points = np.asarray(points, float).reshape(-1, 2)
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix, float).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError('Points project to an invalid location or the plane horizon.')
    return homogeneous[:, :2] / homogeneous[:, 2:]


def fit_grid(candidates, seed_pixels, seed_grid, spacing_cm):
    """Grow grid correspondences from a small identified patch, then reserve final-fit checks."""
    if not np.isfinite(spacing_cm) or spacing_cm <= 0:
        raise ValueError('Grid spacing must be positive.')
    pixels, grid = np.asarray(seed_pixels, float), np.asarray(seed_grid, float)
    points = np.asarray(candidates, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 12 or not np.isfinite(points).all():
        raise ValueError('Provide at least twelve finite candidate hole centers.')
    if (pixels.shape != grid.shape or pixels.ndim != 2 or pixels.shape[1] != 2
            or len(pixels) < 6 or np.linalg.matrix_rank(grid-grid.mean(axis=0)) < 2
            or not np.isfinite(pixels).all() or not np.isfinite(grid).all()
            or not np.array_equal(grid, np.rint(grid))):
        raise ValueError('Provide at least six finite seed holes on a two-dimensional integer grid.')
    matrix, _ = cv2.findHomography(grid, pixels, 0)
    if matrix is None:
        raise ValueError('Seed grid does not determine a homography.')
    cv2.setRNGSeed(0)
    low, high = grid.min(axis=0), grid.max(axis=0)
    for step in range(22):
        coordinates = np.rint(project(points, np.linalg.inv(matrix)))
        residual = np.linalg.norm(project(coordinates, matrix)-points, axis=1)
        margin = np.array([step+1, 2*step+1])
        keep = (residual < 2.) & np.all(coordinates >= low-margin, axis=1) & np.all(coordinates <= high+margin, axis=1)
        unique = {}
        for index in np.flatnonzero(keep):
            key = tuple(coordinates[index])
            if key not in unique or residual[index] < residual[unique[key]]:
                unique[key] = index
        indices = np.array(list(unique.values()), int)
        if len(indices) < 12:
            raise ValueError('Too few consistent grid holes; inspect the seed patch.')
        updated, inliers = cv2.findHomography(coordinates[indices], points[indices], cv2.RANSAC, 1.25)
        if updated is None:
            raise ValueError('Grid fitting failed.')
        matrix = updated
    indices = indices[inliers.ravel().astype(bool)]
    grid, pixels = coordinates[indices], points[indices]
    # These checks test geometric consistency, not independent detection accuracy:
    # correspondence discovery used grid consensus across the image.
    reserved = ((grid[:, 0]+2*grid[:, 1]).astype(int) % 5 == 0)
    if len(grid) < 40 or reserved.sum() < 8:
        raise ValueError('Insufficient grid coverage for a checked plane calibration.')
    matrix, fitted = cv2.findHomography(grid[~reserved], pixels[~reserved], cv2.RANSAC, 1.25)
    if matrix is None or fitted.sum() < 25:
        raise ValueError('Too few fitting inliers for plane calibration.')
    errors_px = np.linalg.norm(project(grid, matrix)-pixels, axis=1)
    pixel_to_cm = np.diag([spacing_cm, -spacing_cm, 1.]) @ np.linalg.inv(matrix)
    errors_cm = np.linalg.norm(project(pixels, pixel_to_cm)-grid*[spacing_cm, -spacing_cm], axis=1)
    metrics = dict(grid_holes=len(grid), fitting_holes=int((~reserved).sum()),
        fitting_inliers=int(fitted.sum()), reserved_holes=int(reserved.sum()),
        reserved_rmse_px=float(np.sqrt(np.mean(errors_px[reserved]**2))),
        reserved_p95_px=float(np.percentile(errors_px[reserved], 95)),
        reserved_rmse_cm=float(np.sqrt(np.mean(errors_cm[reserved]**2))),
        grid_column_span=float(np.ptp(grid[:, 0])), grid_row_span=float(np.ptp(grid[:, 1])))
    if metrics['reserved_p95_px'] > 1.5 or min(metrics['grid_column_span'], metrics['grid_row_span']) < 5:
        raise ValueError('Grid consistency or spatial coverage is insufficient.')
    return pixel_to_cm, metrics, grid, pixels, reserved


def calibrate_wall(video, root, observation_path):
    """Calibrate only the annotated wall; keep the original contact coordinates intact."""
    from .hold_mapping import person_mask
    root, observation_path = Path(root), Path(observation_path)
    observation = json.loads(observation_path.read_text())
    video_hash = file_sha256(video)
    reference_number = observation['reference_frame_number']
    image = read_frame(video, reference_number)
    height, width = image.shape[:2]
    if (observation['video_sha256'] != video_hash or observation['image_size'] != [width, height]
            or observation['grid_angle_degrees'] != 90):
        raise ValueError('Grid observations must match the video, image size, and orthogonal grid.')
    map_path, pose_path = root/'state/reviewed_hold_map.npz', root/'automatic/wholebody.npz'
    records, masks, run_id, selected = load_reviewed_holds(map_path, video, reference_frame_number=reference_number)
    if run_id != observation['candidate_run_id']:
        raise ValueError('Recheck the calibration panel after a candidate-map rebuild.')
    inputs = dict(video_sha256=video_hash, observation_sha256=file_sha256(observation_path),
        hold_map_sha256=file_sha256(map_path), pose_sha256=file_sha256(pose_path),
        module_sha256=file_sha256(__file__))
    manifest = root/'state/plane_calibration_manifest.json'
    if valid_cache(manifest, inputs):
        return json.loads((root/'state/white_wall_calibration.json').read_text())
    polygon = np.asarray(observation['plane_polygon_xy'], np.int32)
    if polygon.shape != (4, 2) or np.any(polygon < 0) or np.any(polygon >= [width, height]):
        raise ValueError('The calibration plane needs four image corners inside the reference.')
    eligible = np.zeros((height, width), np.uint8)
    cv2.fillPoly(eligible, [polygon], 1)
    with np.load(pose_path, allow_pickle=False) as pose:
        if json.loads(str(pose['metadata']))['video_sha256'] != video_hash:
            raise ValueError('Pose exclusion belongs to another video.')
        excluded = person_mask(pose['keypoints'][reference_number], pose['scores'][reference_number],
                               image.shape, padding=20).astype(bool)
    for mask in masks.values():
        excluded |= cv2.dilate(mask.astype(np.uint8), np.ones((7,7),np.uint8)).astype(bool)
    eligible[excluded] = 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    binary = ((contrast > 32) & eligible.astype(bool)).astype(np.uint8)
    _, _, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    candidates = np.array([p for s,p in zip(stats[1:],centers[1:])
        if 2 <= s[4] <= 40 and 1 <= s[2] <= 9 and 1 <= s[3] <= 9 and .35 <= s[2]/s[3] <= 2.5])
    seed_pixels, seed_grid = [], []
    for seed in observation['seed_holes']:
        if not len(candidates):
            raise ValueError('No candidate holes detected on the specified wall.')
        distance = np.linalg.norm(candidates-np.array(seed['pixel_xy']),axis=1)
        nearest = int(np.argmin(distance))
        if distance[nearest] > 2:
            raise ValueError('A seed hole has no nearby dark-blob center; inspect the reference image.')
        seed_pixels.append(candidates[nearest]); seed_grid.append(seed['grid_column_row'])
    matrix, metrics, grid, pixels, reserved = fit_grid(candidates, seed_pixels, seed_grid, observation['spacing_cm'])
    hull = cv2.convexHull(pixels.astype(np.float32))
    hold_positions = []
    for record in records:
        if record['hold_id'] not in selected:
            continue
        point = (record['cx_px'], record['cy_px'])
        inside_plane = cv2.pointPolygonTest(polygon, point, False) >= 0
        hold_positions.append(dict(hold_id=record['hold_id'], pixel_xy=list(point),
            wall_xy_cm=project([point],matrix)[0].tolist() if inside_plane else None,
            inside_plane=inside_plane, inside_grid_hull=cv2.pointPolygonTest(hull, point, False) >= 0,
            distance_outside_grid_hull_px=max(0., -cv2.pointPolygonTest(hull, point, True))))
    output = dict(schema_version=1, video_sha256=video_hash, reference_frame_number=reference_number,
        candidate_run_id=run_id, plane_id=observation['plane_id'], spacing_cm=observation['spacing_cm'],
        spacing_source='user_reported', grid_angle_degrees=90, wall_floor_angle_degrees=90,
        coordinate_system='centimeters; x follows grid columns; y points up; origin is the seed (0,0) hole',
        image_to_wall_cm=matrix.tolist(), wall_cm_to_image=np.linalg.inv(matrix).tolist(),
        plane_polygon_xy=polygon.tolist(), metrics=metrics, selected_hold_positions=hold_positions,
        status='plane_homography_fitted', ready_for_metric_ik=True, observation_file=str(observation_path),
        remaining_ik_work=None,
        ik_scope='Only contact schedules on this wall and inside the observed-hole coverage are eligible.',
        limitations=['Conditional on the user-reported 15 cm orthogonal grid and one planar wall.',
            'Grid residuals measure internal geometric consistency, not physical accuracy or contact accuracy.',
            'Holds outside the observed-hole convex hull use extrapolation and need additional local checks.',
            'Hold depth, body depth, camera intrinsics, and the second wall are not reconstructed.',
            'Seed holes and the panel polygon were identified by the assistant; no user review of their image coordinates.'])
    overlay = image.copy()
    cv2.polylines(overlay,[polygon],True,(255,130,0),2)
    for point, check in zip(pixels,reserved):
        cv2.circle(overlay,tuple(np.rint(point).astype(int)),4,(0,150,255) if check else (40,170,40),1)
    for point in seed_pixels:
        cv2.drawMarker(overlay,tuple(np.rint(point).astype(int)),(220,0,180),cv2.MARKER_CROSS,12,2)
    cv2.rectangle(overlay,(0,0),(width,55),(255,255,255),-1)
    cv2.putText(overlay,'White wall grid | green: fit | orange: reserved',(10,22),0,.5,(25,25,25),1)
    cv2.putText(overlay,'15 cm square spacing supplied by user; purple: seed patch',(10,45),0,.46,(25,25,25),1)
    overlay_path = root/'images/white_wall_grid.png'
    cv2.imwrite(str(overlay_path),overlay)
    # Rectify at a display-only scale. The input image has no single px/cm scale.
    corners = project(polygon,matrix)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    display_scale = 2.
    display_transform = np.array([[display_scale,0,-lo[0]*display_scale],
                                  [0,-display_scale,hi[1]*display_scale],[0,0,1.]]) @ matrix
    size = tuple(np.ceil((hi-lo)*display_scale).astype(int))
    with np.load(map_path,allow_pickle=False) as saved:
        wall = saved['reference_frame'].copy()
    pane = np.zeros((height,width),np.uint8);cv2.fillPoly(pane,[polygon],255)
    wall[pane==0]=245
    rectified = cv2.warpPerspective(wall,display_transform,size,borderValue=(245,245,245))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,9))
    ax.imshow(cv2.cvtColor(rectified,cv2.COLOR_BGR2RGB),extent=[lo[0],hi[0],lo[1],hi[1]])
    for hold in hold_positions:
        if hold['inside_plane']:
            x,y=hold['wall_xy_cm'];ax.scatter(x,y,s=28,facecolors='none',edgecolors='yellow')
            ax.text(x+3,y+3,hold['hold_id'],fontsize=8,color='darkred')
    ax.set_xlabel('Horizontal coordinate (cm)');ax.set_ylabel('Height coordinate (cm; arbitrary origin)')
    ax.set_title('White wall: metric plane rectification\n15 cm square grid; hold depth omitted',fontsize=11)
    ax.set_aspect('equal');fig.tight_layout()
    rectified_path=root/'images/white_wall_metric.png'
    fig.savefig(rectified_path,dpi=150);plt.close(fig)
    state_path=root/'state/white_wall_calibration.json';write_json(state_path,output)
    correspondences=root/'state/white_wall_grid_points.json'
    write_json(correspondences,dict(grid_column_row=grid.tolist(),image_xy=pixels.tolist(),
        reserved_from_final_fit=reserved.tolist(),discovery='Dark blobs matched by expanding grid consensus'))
    checkpoint(manifest,inputs,[state_path,correspondences,overlay_path,rectified_path])
    return output
