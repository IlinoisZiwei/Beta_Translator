"""2D body model and IK. Positions are centimeters, y is upward, angles are radians."""

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple
from scipy.optimize import least_squares

# An approximation of human segment lengths as a fraction of standing height H
# Change as needed
RATIOS = {
    "upper_arm": 0.186,   # shoulder joint --- elbow
    "forearm": 0.146,     # elbow --- wrist
    "thigh": 0.245,       # hip joint --- knee
    "shank": 0.246,       # knee --- ankle
    "shoulder_w": 0.259,  # left shoulder --- right shoulder
    "hip_w": 0.191,       # left hip joint --- right hip joint
    "torso": 0.288        # midpoint of the hips --- midpoint of the shoulders
}

def make_body(height_cm, ratios=None):
    """Convert height and segment ratios to lengths in centimeters."""
    ratios = RATIOS if ratios is None else ratios
    if height_cm is None or not np.isfinite(height_cm) or height_cm <= 0:
        raise ValueError("Height must be positive and finite.")
    if set(ratios) != set(RATIOS):
        raise ValueError(f"Body ratios must contain exactly {list(RATIOS)}.")
    if any(not np.isfinite(value) or value <= 0 for value in ratios.values()):
        raise ValueError("Every body ratio must be positive and finite.")
    return {name: float(ratio * height_cm) for name, ratio in ratios.items()}

LIMBS = [
  'left_hand', 'right_hand', 'left_foot', 'right_foot'
]

# x increases to the right, y increase upward, all angles are radians increasing with CCW rotation
# hip_x = x position of hip center on the wall
# hip_y = y position of hip center
# torso_angle = torso tilt in radians, 0 is upright
# left_hand_theta1, left_hand_theta2, ... , right_foot_theta1, right_foot_theta2
# theta1 is the angle of the upper segment closer to torso (shoulder or hip), 0 is downward
# theta2 is the angle of the lower segment (elbow or knee) relative to theta1, 0 is parallel to upper segment
POSE_VECTOR_NAMES = ['hip_x', 'hip_y', 'torso_angle'] + [f'{l}_{j}' for l in LIMBS for j in ('theta1', 'theta2')]
POSE_VECTOR_LENGTH = len(POSE_VECTOR_NAMES)

HIP_POSITION = slice(0, 2)
ALL_ANGLES = slice(2, None)
FIRST_LIMB_INDEX = 3

# Indices of (theta1, theta2) for LIMBS[limb_number] in POSE_VECTOR_NAMES
def limb_angle_indices(limb_number):
    upper_segment_angle_index = FIRST_LIMB_INDEX + 2 * limb_number
    return upper_segment_angle_index, upper_segment_angle_index + 1

# Immutable
class LimbJoints(NamedTuple):
    root: np.ndarray     # shoulder or hip joint
    middle: np.ndarray   # elbow or knee
    end: np.ndarray      # wrist or ankle (the point that touches the hold)

# Mutable
@dataclass
class BodyJoints:
    hip_center: np.ndarray
    shoulder_center: np.ndarray
    limbs: dict          # limb name -> LimbJoints

# Wall coords: x right, y UP, in cm. Image pixels have y DOWN, so flip y when converting.
# "Left" = image-left, which assumes the camera is behind the climber
def forward_kinematics(pose_vector, body_lengths):
    hip_x, hip_y, torso_angle = pose_vector[:3]

    # unit vector from midpoint between hips to midpoint between shoulders
    torso_up_direction = np.array([-np.sin(torso_angle), np.cos(torso_angle)])

    # unit vector from shoulder/hip line, toward +x when body is upright
    torso_right_direction = np.array([np.cos(torso_angle), np.sin(torso_angle)])

    hip_center = np.array([hip_x, hip_y])
    shoulder_center = hip_center + body_lengths['torso'] * torso_up_direction

    half_shoulder_offset = torso_right_direction * body_lengths['shoulder_w'] / 2
    half_hip_offset = torso_right_direction * body_lengths['hip_w'] / 2
    limb_root_positions = {
        'left_hand': shoulder_center - half_shoulder_offset,
        'right_hand': shoulder_center + half_shoulder_offset,
        'left_foot':  hip_center - half_hip_offset,
        'right_foot': hip_center + half_hip_offset
    }

    # (upper segment length, lower segment length) for arms and legs
    segment_lengths = {
        'hand': (body_lengths['upper_arm'], body_lengths['forearm']),
        'foot': (body_lengths['thigh'], body_lengths['shank'])
    }

    limbs = {}
    for limb_number, limb_name in enumerate(LIMBS):
        upper_angle_index, bend_angle_index = limb_angle_indices(limb_number)
        upper_segment_angle = pose_vector[upper_angle_index] # theta1

        bend_angle = pose_vector[bend_angle_index] # theta2
        limb_kind = limb_name.split('_')[1] # 'hand' or 'foot'

        upper_segment_length, lower_segment_length = segment_lengths[limb_kind]
        lower_segment_angle = upper_segment_angle + bend_angle

        root_position = limb_root_positions[limb_name]
        # -np.cos(upper_segment_angle) because 0 is downward but positive Y direction is upward
        middle_position = root_position + upper_segment_length * np.array([np.sin(upper_segment_angle), -np.cos(upper_segment_angle)])
        end_position = middle_position + lower_segment_length * np.array([np.sin(lower_segment_angle), -np.cos(lower_segment_angle)])

        limbs[limb_name] = LimbJoints(root_position, middle_position, end_position)
    return BodyJoints(hip_center, shoulder_center, limbs)

# Find the pose vector that puts each contacted limb's end point on its target
# while staying close to the reference pose.
# contact_targets: {limb name: (x, y) in cm, or None for a free limb}
# Returns (pose_vector, contact_errors_cm), where contact_errors_cm maps each
# contacted limb to the distance between its end point and its target. The units
# are in cm or radians, and the origin is the origin of coordinates in contact_targets.
# pose_adapter.image_to_wall_cm uses the bottom-left of frame 0 as the origin.
def solve_inverse_kinematics(
    body_lengths,
    contact_targets,
    reference_pose_vector,
    contact_weight = 10.0,              # wrist or ankle position minus target, x and y per limb
    angle_weight = 0.3,                     # torso angle and 8 limb angles minus the reference’s
    hip_position_weight = 0.02,     # hip x, y minus the reference’s
    max_bend_angle = np.deg2rad(150)
):
    reference_pose_vector = np.asarray(reference_pose_vector, dtype=float)
    if reference_pose_vector.shape != (POSE_VECTOR_LENGTH,) or not np.isfinite(reference_pose_vector).all():
        raise ValueError("The reference pose must contain 11 finite values.")
    for limb, target in contact_targets.items():
        if limb not in LIMBS:
            raise ValueError(f"Unknown limb: {limb}")
        if target is not None:
            point = np.asarray(target, dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError(f"{limb}: contact targets must be finite 2D points.")

    # Returns all residuals (error in actual vs. expected location) for the pose vector
    # used as argument to scipy.least_squares
    def residuals(pose_vector):
        joints = forward_kinematics(pose_vector, body_lengths)
        residual_terms = []
        for limb_name, target in contact_targets.items():
            if target is not None:
                contact_error = joints.limbs[limb_name].end - np.asarray(target)
                residual_terms.append(contact_weight * contact_error)

        # stay close to the reference joint angles
        residual_terms.append(angle_weight * (pose_vector[ALL_ANGLES] - reference_pose_vector[ALL_ANGLES]))

        # weakly keep the hips near the reference position
        residual_terms.append(hip_position_weight * (pose_vector[HIP_POSITION] - reference_pose_vector[HIP_POSITION]))

        # return flattened list of residual terms
        return np.concatenate(residual_terms)

    lower_bounds = np.full(POSE_VECTOR_LENGTH, -np.inf)
    upper_bounds = np.full(POSE_VECTOR_LENGTH, np.inf)
    for limb_number in range(len(LIMBS)):
        _, bend_angle_index = limb_angle_indices(limb_number)
        lower_bounds[bend_angle_index] = -max_bend_angle
        upper_bounds[bend_angle_index] = max_bend_angle

    # Constrain the starting guess while retaining the original reference in residuals.
    initial_pose = np.clip(reference_pose_vector, lower_bounds, upper_bounds)
    # Scale centimeter positions and radian angles by their effect on the residuals.
    solution = least_squares(
        residuals, initial_pose, bounds=(lower_bounds, upper_bounds), x_scale="jac",
    )
    if not solution.success:
        raise RuntimeError(f"IK did not converge: {solution.message}")

    solved_joints = forward_kinematics(solution.x, body_lengths)
    contact_errors_cm = {
        limb_name: float(np.linalg.norm(solved_joints.limbs[limb_name].end - np.asarray(target)))
        for limb_name, target in contact_targets.items() if target is not None
    }
    return solution.x, contact_errors_cm
