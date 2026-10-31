"""Convert tracked landmarks to the coordinate and angle conventions used by IK."""

import cv2
import numpy as np

from .inverse_kinematics import LIMBS

# points_px: current-frame pixels -> fixed wall-reference pixels.
# The legacy name to_frame0 is retained for callers; the reference may be any frame.
# Homographies map the configured wall reference into each video frame.
def to_frame0(points_px, homographies, frame_index):
    points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    H = homographies[frame_index][0]
    return cv2.perspectiveTransform(points, np.linalg.inv(H)).reshape(-1, 2)


def image_to_wall_cm(points_px, px_per_cm, image_height_px):
    """Reference-image pixel points (N, 2) -> wall coordinates in cm. Temporary: image-frame scaling."""
    points_px = np.asarray(points_px, dtype=float)
    return np.stack([points_px[:, 0] / px_per_cm,
                     (image_height_px - points_px[:, 1]) / px_per_cm], axis=1)

def wall_cm_to_image(points_cm, px_per_cm, image_height_px):
    """Inverse, for drawing results back onto the reference image."""
    points_cm = np.asarray(points_cm, dtype=float)
    return np.stack([points_cm[:, 0] * px_per_cm,
                     image_height_px - points_cm[:, 1] * px_per_cm], axis=1)

# Convert aligned landmark rows into the solver's pose vector.
def landmark_position_cm(row, landmark_name, px_per_cm, image_height_px):
    x_cm = row[f'{landmark_name}_x_px'] / px_per_cm
    y_cm = (image_height_px - row[f'{landmark_name}_y_px']) / px_per_cm   # flip so y points up
    return np.array([x_cm, y_cm])

# Angle of start->end in the model's convention: 0 = straight down, counterclockwise positive.
def segment_angle(start, end):
    dx, dy = end - start
    return np.arctan2(dx, -dy)

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

LIMB_LANDMARKS = {
    'left_hand':  ('left_shoulder', 'left_elbow', 'left_wrist'),
    'right_hand': ('right_shoulder', 'right_elbow', 'right_wrist'),
    'left_foot':  ('left_hip', 'left_knee', 'left_ankle'),
    'right_foot': ('right_hip', 'right_knee', 'right_ankle')
}

# Converts a row to a pose vector
def pose_row_to_vector(row, px_per_cm, image_height_px):
    position = lambda name: landmark_position_cm(row, name, px_per_cm, image_height_px)
    hip_center = (position('left_hip') + position('right_hip')) / 2
    shoulder_center = (position('left_shoulder') + position('right_shoulder')) / 2
    torso_dx, torso_dy = shoulder_center - hip_center
    torso_angle = np.arctan2(-torso_dx, torso_dy)        # inverse of up = (-sin, cos)
    pose_vector = [hip_center[0], hip_center[1], torso_angle]
    for limb_name in LIMBS:
        root, middle, end = (position(name) for name in LIMB_LANDMARKS[limb_name])
        upper_angle = segment_angle(root, middle)                                   # theta1
        bend_angle = wrap_angle(segment_angle(middle, end) - upper_angle)           # theta2
        pose_vector += [upper_angle, bend_angle]
    return np.array(pose_vector)

# Returns a weight in [0, 1] for each angle (torso + 8 limb angles), low when the landmarks it depends on are hard to see.
def angle_visibility_weights(row):
    visibility = lambda name: row[f'{name}_visibility']
    weights = [min(visibility(n) for n in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'))]
    for limb_name in LIMBS:
        root, middle, end = LIMB_LANDMARKS[limb_name]
        weights.append(min(visibility(root), visibility(middle)))                    # theta1
        weights.append(min(visibility(root), visibility(middle), visibility(end)))   # theta2
    return np.array(weights)
