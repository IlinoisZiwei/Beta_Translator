"""
!!! PLEASE READ !!!

This is compatible with data collected in pose_coordinates.csv,
but there are some important instructions to do use this file properly:
1. The .csv file's +y direction is downward while this assumes +y is upward
2. The .csv file's units are in pixels while this is in cm (requires a px_per_cm from the wall calibration)
3. hip_x, hip_y, torso_angle, and theta1/theta2 for all limbs need to be calculated. These can be computed
   from landmark vectors with theta = arctan(dx, -dy) for a segment direction.
4. The angle terms need to be weighed by visibility.
5. Use df[df.frame == i] instead of df.iloc[i] for frame lookup because rows only exist for frames where a
   pose is detected.
6. Landmarks are in frame-i pixels and holds are in frame-0 pixels (see the last cell in Step 9 of Beta_Translator.ipynb)
"""

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

def make_body(height_cm):
  return {k: v * height_cm for k, v in RATIOS.items()}

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
		limb_kind = limb_name.split('_')[1]	# 'hand' or 'foot'

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
# Using `pose_adapter.py`, the origin would be the bottom-left corner of the first frame (frame 0).
def solve_inverse_kinematics(
	body_lengths,
	contact_targets,
	reference_pose_vector,
	contact_weight = 10.0,				# wrist or ankle position minus target, x and y per limb
	angle_weight = 0.3,						# torso angle and 8 limb angles minus the reference’s
	hip_position_weight = 0.02,		# hip x, y minus the reference’s
	max_bend_angle = np.deg2rad(150)
):
	reference_pose_vector = np.asarray(reference_pose_vector, dtype=float)

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

	solution = least_squares(residuals, reference_pose_vector, bounds=(lower_bounds, upper_bounds))

	solved_joints = forward_kinematics(solution.x, body_lengths)
	contact_errors_cm = {
		limb_name: float(np.linalg.norm(solved_joints.limbs[limb_name].end - np.asarray(target)))
		for limb_name, target in contact_targets.items() if target is not None
	}
	return solution.x, contact_errors_cm


# Helper functions


# Testing inverse kinematics
# d = np.deg2rad
# reference_pose_vector = np.array([
#   0, 100, 0.05,        # hip_x, hip_y, torso_angle
#   d(120), d(-25),      # left hand
#   d(140), d(-15),      # right hand
#   d(-15), d(25),       # left foot
#   d(20), d(35)         # right foot
# ])
# reference_joints = forward_kinematics(reference_pose_vector, make_body(185))
# contact_targets = {limb: tuple(reference_joints.limbs[limb].end) for limb in LIMBS}

# for height_cm in (185, 170, 160):
# 	solution, contact_errors_cm = solve_inverse_kinematics(make_body(height_cm), contact_targets, reference_pose_vector)
# 	print(solution)
# 	print(height_cm, {limb: round(err, 2) for limb, err in contact_errors_cm.items()})
