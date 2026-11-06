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


# =============================================================================
# Static balance (gravity) extension
#
# Everything below is additive: no function above this line was changed and none of
# them calls anything below, so existing callers behave exactly as before.
#
# Model: the climber is the planar body above, in static equilibrium in the wall plane
# (x right, y up, cm). Each contacted limb end (wrist/ankle) gets a contact force from
# its hold, and gravity acts straight down through the center of mass (COM):
#     sum(contact forces)                 = (0, m*g)
#     sum((contact point - COM) x force)  = 0
# Hands may pull or push in any in-plane direction. Feet can only push up (fy >= 0) and
# their sideways force is limited by friction, |fx| <= mu * fy. Every force component is
# bounded by max_force_n. Contact forces are what the hold applies TO the climber.
#
# Not modeled: out-of-plane forces and torques (wall angle, hips-to-wall distance,
# barn-dooring), grip strength, joint torques, or muscle effort. Contacts are the
# wrist/ankle points, as in solve_inverse_kinematics.
# =============================================================================

GRAVITY = 9.81  # m/s^2

# Per-side mass fractions, checked against Xu & Zhou (2017), Table 1,
# which attributes its parameters to Dempster. These are population priors,
# not measurements of this climber. See anthropometry_profile() for provenance.
# The skeleton ends at the wrist/ankle: "forearm" includes hand, "shank" foot,
# and "torso" combines thorax/abdomen and pelvis. Combined MASS does not imply
# that the combined COM has the same position as either constituent's COM.
MASS_FRACTIONS = {
    "head": 0.081,  # neck included
    "torso": 0.497,
    "upper_arm": 0.028,
    "forearm": 0.022,   # hand included
    "thigh": 0.100,
    "shank": 0.061  # shank (knees to ankles) + foot
}

# Multiplicities in the current skeleton; limb masses are for ONE side.
SEGMENT_COUNTS = {
    "head": 1,
    "torso": 1,
    "upper_arm": 2,
    "forearm": 2,
    "thigh": 2,
    "shank": 2
}

# Legacy COM defaults retained for reproducibility. Upper-arm/thigh fractions
# match the cited table; the other positions are explicit project proxies.
# All except "head" interpolate between joint centers. "head" is an offset
# ABOVE the shoulder center, measured in torso lengths, not a head-length ratio.
CENTER_OF_MASS_FRACTIONS = {
    "head": 0.40,
    "torso": 0.50,
    "upper_arm": 0.436,
    "forearm": 0.69,
    "thigh": 0.433,
    "shank": 0.57
}

ANTHROPOMETRY_PROFILES = ("legacy_v1", "dempster_endpoint_proxy_v1")
ANTHROPOMETRY_SOURCES = {
    "values_table": "https://journals.sagepub.com/doi/10.1177/1550147717703257",
    "dempster_report_catalog": "https://contrails.library.iit.edu/item/154630",
    "landmark_caution": "https://doi.org/10.1016/0021-9290(95)00178-6",
}
# Uncombined lower-limb/arm entries from the cited Table 1. The coefficient
# refers to the bone segment, not to the bone-plus-hand/foot lump.
DISTAL_COMPONENTS = {
    "forearm": {"bone_mass": .016, "distal_mass": .006, "bone_com_fraction": .430},
    "shank": {"bone_mass": .0465, "distal_mass": .0145, "bone_com_fraction": .433},
}


def weighted_center_of_mass(points_cm, masses):
    """Mass-weighted points (..., segments, 2 or 3); preserves complex derivatives.

    Masses may be kg or relative weights. No automatic equal-weight averaging,
    missing-point substitution, or normalization of an invalid mass table.
    """
    points, weights = np.asarray(points_cm), np.asarray(masses)
    if (points.ndim < 2 or points.shape[-1] not in (2, 3)
            or weights.ndim != 1 or len(weights) == 0
            or points.shape[-2] != len(weights)
            or not np.isfinite(points).all() or not np.isfinite(weights).all()
            or not np.isrealobj(weights) or np.any(weights <= 0)):
        raise ValueError("Finite 2D/3D points and matching positive real masses required.")
    # Scale first so finite large relative weights cannot overflow their sum.
    scaled = weights / weights.max()
    return np.sum(points * (scaled / scaled.sum())[:, None], axis=-2)


def combined_distal_center_of_mass(proximal_cm, endpoint_cm, distal_com_cm, segment):
    """Combine forearm+hand or shank+foot using the published mass split.

    proximal/endpoint are elbow/wrist for 'forearm', knee/ankle for 'shank'.
    distal_com_cm is the HAND/FOOT COM, not a hold/contact point. All positions
    must share the same coordinate frame and cm units, with shape (..., 2 or 3).
    Passing the wrist/ankle as distal COM is an explicit zero-offset proxy;
    measured/estimated hand or foot COM positions can instead reflect orientation.
    This function does not infer anatomy or a hand/foot orientation from contacts.
    """
    if segment not in DISTAL_COMPONENTS:
        raise ValueError("segment must be 'forearm' or 'shank'.")
    proximal, endpoint, distal = map(np.asarray, (proximal_cm, endpoint_cm, distal_com_cm))
    if proximal.shape != endpoint.shape or proximal.shape != distal.shape:
        raise ValueError("All component positions must have the same shape.")
    values = DISTAL_COMPONENTS[segment]
    bone = proximal + values['bone_com_fraction'] * (endpoint - proximal)
    return weighted_center_of_mass(np.stack([bone, distal], axis=-2),
                                   [values['bone_mass'], values['distal_mass']])


def anthropometry_profile(name="legacy_v1"):
    """Return independent, JSON-serializable parameters, endpoints and caveats.

    The endpoint proxy is an alternative for sensitivity experiments, not a
    validated improvement. Pass its com_fractions to the 2D/3D COM APIs or the
    static gravity IK solver; persist this entire record alongside new results.
    """
    if name not in ANTHROPOMETRY_PROFILES:
        raise ValueError(f"Unknown anthropometry profile; choose {ANTHROPOMETRY_PROFILES}.")
    c = dict(CENTER_OF_MASS_FRACTIONS)
    if name == "dempster_endpoint_proxy_v1":
        for segment in DISTAL_COMPONENTS:
            c[segment] = float(combined_distal_center_of_mass(
                [0., 0.], [1., 0.], [1., 0.], segment)[0])
    endpoint = name == "dempster_endpoint_proxy_v1"
    rules = {
        "head": {"mass_rule": "head+neck: 0.081", "com_evidence": "project_proxy",
                 "coordinates": "shoulder_center + 0.40 * (shoulder_center - hip_center)"},
        "torso": {"mass_rule": "thorax/abdomen 0.355 + pelvis 0.142", "com_evidence": "project_proxy",
                  "coordinates": "hip_center + 0.50 * (shoulder_center - hip_center)"},
        "upper_arm": {"mass_rule": "0.028 per side", "com_evidence": "published_fraction_mapped_to_joint_centers",
                      "coordinates": "shoulder -> elbow; proximal fraction 0.436"},
        "forearm": {"mass_rule": "forearm 0.016 + hand 0.006 per side",
                    "com_evidence": "derived_with_endpoint_proxy" if endpoint else "legacy_project_proxy",
                    "coordinates": "elbow -> wrist; combined forearm+hand COM"},
        "thigh": {"mass_rule": "0.100 per side", "com_evidence": "published_fraction_mapped_to_joint_centers",
                  "coordinates": "hip -> knee; proximal fraction 0.433"},
        "shank": {"mass_rule": "shank 0.0465 + foot 0.0145 per side",
                  "com_evidence": "derived_with_endpoint_proxy" if endpoint else "legacy_project_proxy",
                  "coordinates": "knee -> ankle; combined shank+foot COM"},
    }
    return dict(profile=name, schema_version=1, mass_fractions=dict(MASS_FRACTIONS),
                segment_counts=dict(SEGMENT_COUNTS), com_fractions=c, rules=rules,
                distal_components={k: dict(v) for k, v in DISTAL_COMPONENTS.items()},
                sources=dict(ANTHROPOMETRY_SOURCES), validated_for_individual=False,
                assumptions=[
                    "Population mass fractions; no personal body-composition measurement.",
                    "Published segment landmarks are approximated by this skeleton's joint centers.",
                    "Head offset and combined torso COM are unmeasured project proxies.",
                    ("Hand COM = wrist; foot COM = ankle; orientation/length is not observed."
                     if endpoint else "Combined COM fractions 0.69/0.57 have no verified derivation in this project."),
                    "A profile comparison is sensitivity analysis, not a confidence interval or physical validation.",
                ])


def _check_segment_values(values, name):
    # Checks if all required dict keys are present in `name`
    if set(values) != set(MASS_FRACTIONS):
        raise ValueError(f"{name} must contain exactly {list(MASS_FRACTIONS)}.")

    if any(np.ndim(value) != 0 or not np.isrealobj(value)
           or not np.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError(f"Every {name} entry must be positive and finite.")


def validate_com_fractions(values):
    """Joint interpolation fractions lie in [0, 1]; head is a torso-length offset.

    This checks the parameter representation, not the anatomy of an individual.
    No evidence-based head-offset range is available, so only nonnegativity and
    finiteness are imposed on that explicitly separate proxy.
    """
    if set(values) != set(MASS_FRACTIONS):
        raise ValueError(f"com_fractions must contain exactly {list(MASS_FRACTIONS)}.")
    for name, value in values.items():
        if (np.ndim(value) != 0 or not np.isrealobj(value) or not np.isfinite(value)
                or value < 0 or (name != 'head' and value > 1)):
            raise ValueError(f"Invalid COM fraction for {name}; segments require [0, 1], head a nonnegative offset.")

def make_mass_model(mass_kg, mass_fractions=None):
    """Convert total mass and per-segment mass fractions to the mass in kg for one of each segment"""
    fractions = MASS_FRACTIONS if mass_fractions is None else mass_fractions
    if mass_kg is None or not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("Mass must be positive and finite.")

    _check_segment_values(fractions, "mass_fractions")

    fraction_sum = sum(SEGMENT_COUNTS[name] * value for name, value in fractions.items())
    if abs(fraction_sum - 1.0) > 1e-6:
        raise ValueError(f"Mass fractions (counting both arms and legs) must sum to 1, not {fraction_sum:.6f}.")
    
    return {name: float(value * mass_kg) for name, value in fractions.items()}

def center_of_mass(pose_vector, body_lengths, segment_masses_kg, center_of_mass_fractions=None):
    """
    Returns the whole-body center of mass in wall coordinates (cm).
    `segment_masses_kg` comes from make_mass_model.
    """

    if center_of_mass_fractions is None:
        center_of_mass_fractions = CENTER_OF_MASS_FRACTIONS
    
    validate_com_fractions(center_of_mass_fractions)
    _check_segment_values(segment_masses_kg, "segment_masses_kg")

    joints = forward_kinematics(pose_vector, body_lengths)

    torso_vector = joints.shoulder_center - joints.hip_center

    # A list of 2-element tuples representing a list of center of mass points for each segment
    # The first element is the weight of it (based on segment_masses_kg)
    # The second element is the approximate location of this segments center of mass
    weighted_com_points = [
        (segment_masses_kg["head"],
         joints.shoulder_center + center_of_mass_fractions["head"] * torso_vector),
        (segment_masses_kg["torso"], joints.hip_center + center_of_mass_fractions["torso"] * torso_vector)
    ]

    for limb_name, limb in joints.limbs.items():
        upper, lower = ("upper_arm", "forearm") if limb_name.endswith("hand") else ("thigh", "shank")
        weighted_com_points.append((segment_masses_kg[upper],
                                limb.root + center_of_mass_fractions[upper] * (limb.middle - limb.root)))
        weighted_com_points.append((segment_masses_kg[lower],
                                        limb.middle + center_of_mass_fractions[lower] * (limb.end - limb.middle)))

    # Calculate the center of mass for the whole body based on the intermediate weighted center of mass points
    masses = np.array([mass for mass, _ in weighted_com_points])
    points = np.array([point for _, point in weighted_com_points])
    return weighted_center_of_mass(points, masses)

def static_balance_error(center_of_mass_cm, contact_points_cm, contact_forces_n, mass_kg, gravity=GRAVITY):
    """Leftover wrench after gravity and the contact forces (both in newtons, positions in cm).
 
    Returns (net_force_n, net_moment_ncm): the net force (N) on the center of mass and 
    the moment (torque) about the COM (CCW positive) in (N * cm) units.

    Gravity exerts no moment about the COM, so only the contact forces enter the moment.
    Both are ~0 for a body in static equilibrium.

    See GravityIKResults for more details about what this net force and net moment mean.
    """
    com = np.asarray(center_of_mass_cm, dtype=float)
    points = np.asarray(contact_points_cm, dtype=float).reshape(-1, 2)
    forces = np.asarray(contact_forces_n, dtype=float).reshape(-1, 2)
    
    net_force = forces.sum(axis=0) + np.array([0.0, -mass_kg * gravity])
    lever_arms = points - com

    # What is inside the sum is just a cross product
    net_moment = float(np.sum(lever_arms[:, 0] * forces[:, 1] - lever_arms[:, 1] * forces[:, 0]))

    return net_force, net_moment

class GravityIKResult(NamedTuple):
    """
    The solution to a gravity-based inverse kinematics.
    These are all the parameters relating to the climber in the proposed solution.
    """
    pose_vector: np.ndarray
    contact_errors_cm: dict         # limb -> distance from end point to its target (as in solve_inverse_kinematics)
    contact_forces_n: dict          # limb -> [fx, fy] the hold applies to the climber, newtons
    center_of_mass_cm: np.ndarray
    net_force_n: np.ndarray         # leftover force on the COM after gravity: ~0 when balanced (there should be normal force too)
    net_moment_ncm: float           # leftover torque about the COM: ~0 when balanced
    balanced: bool                  # equilibrium and foot friction hold to within balance_tolerance * body weight

def solve_inverse_kinematics_with_gravity(
    body_lengths,
    contact_targets,
    reference_pose_vector,
    mass_kg,
    mass_fractions=None,
    com_fractions=None,
    contact_weight=10.0,        # same as in solve_inverse_kinematics
    angle_weight=0.3,           # same as in solve_inverse_kinematics
    hip_position_weight=0.02,
    balance_weight=30.0,        # how much to consider force balance, moment balance and friction, in body-weight units
    force_weight=0.05,          # how much to consider lower contact forces
    max_force_n=None,           # per-component force limit in newtons; default is one body weight
    foot_friction=0.8,          # |fx| <= foot_friction * fy at feet
    gravity=GRAVITY,
    balance_tolerance=0.01,     # maximum fraction of body weight allowed in order to be considered balanced
    max_bend_angle=np.deg2rad(150)
):
    """
    Static-balance version of solve_inverse_kinematics.
 
    contact_targets, reference_pose_vector and body_lengths are the same as in
    solve_inverse_kinematics. This also chooses a contact force at every contacted limb so the
    climber is in static equilibrium, and the pose is chosen jointly with those forces.
 
    Contact positions have priority: if no balanced pose can reach them within the force limits,
    the result keeps the contact errors small and reports balanced=False with the leftover wrench.
    Contact forces are not unique when three or more limbs touch; force_weight and the foot/hand
    limits decide which solution is returned, so treat the split between limbs as a modeling
    outcome, not a measurement.
    """

    # Validate reference_pose_vector is in the right format
    reference = np.asarray(reference_pose_vector, dtype=float)
    if reference.shape != (POSE_VECTOR_LENGTH,) or not np.isfinite(reference).all():
        raise ValueError("The reference pose must contain 11 finite values.")

    # Validate contact_targets is in the right format
    for limb, target in contact_targets.items():
        if limb not in LIMBS:
            raise ValueError(f"Unknown limb: {limb}")
        if target is not None:
            point = np.asarray(target, dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError(f"{limb}: contact targets must be finite 2D points.")

    # Get the contact points and validate more arguments
    contacted = [limb for limb in LIMBS if contact_targets.get(limb) is not None]
    if not contacted:
        raise ValueError("At least one contacted limb is needed to support the body.")
    if not np.isfinite(gravity) or gravity <= 0:
        raise ValueError("Gravity must be positive and finite.")
    if foot_friction < 0 or not np.isfinite(foot_friction):
        raise ValueError("Foot friction must be non-negative and finite.")
    if max_force_n is not None and (not np.isfinite(max_force_n) or max_force_n <= 0):
        raise ValueError("max_force_n must be positive and finite.")

    # Initialize the relevant variables for the gravity-based IK model
    segment_masses = make_mass_model(mass_kg, mass_fractions)
    weight_n = mass_kg * gravity
    force_limit = 1.0 if max_force_n is None else max_force_n / weight_n    # in body weights
    count = len(contacted)
    targets = np.array([np.asarray(contact_targets[limb], dtype=float) for limb in contacted])
    is_foot = np.array([limb.endswith("foot") for limb in contacted])
    torso_length = body_lengths["torso"]

    # Unknowns: A list that is equivalent to: [a pose] + [(fx, fy) per contacted limb in units of body weight]
    def split(unknowns):
        return unknowns[:POSE_VECTOR_LENGTH], unknowns[POSE_VECTOR_LENGTH:].reshape(count, 2)

    # Helper to get the differences/errors between a desired target for all the different desired parameters
    def residuals(unknowns):
        pose, forces = split(unknowns)
        joints = forward_kinematics(pose, body_lengths)
        ends = np.array([joints.limbs[limb].end for limb in contacted])
        com = center_of_mass(pose, body_lengths, segment_masses, com_fractions)

        net_force = forces.sum(axis=0) - np.array([0.0, 1.0])
        lever_arms = ends - com

        # cross product inside sum()
        net_moment = np.sum(lever_arms[:, 0] * forces[:, 1] - lever_arms[:, 1] * forces[:, 0]) / torso_length
        friction_excess = np.maximum(0.0, np.abs(forces[is_foot, 0]) - foot_friction * forces[is_foot, 1])

        return np.concatenate([
            contact_weight * (ends - targets).ravel(),
            angle_weight * (pose[ALL_ANGLES] - reference[ALL_ANGLES]),
            hip_position_weight * (pose[HIP_POSITION] - reference[HIP_POSITION]),
            balance_weight * np.array([net_force[0], net_force[1], net_moment]),
            balance_weight * friction_excess,
            force_weight * forces.ravel(),
        ])

    # Intitialize min/max bend angles and min/max forces for all poses
    # Rememeber y direction is positive upwards
    # lower is for minimum; upper is for maximum
    lower = np.full(POSE_VECTOR_LENGTH + 2 * count, -np.inf)
    upper = np.full(POSE_VECTOR_LENGTH + 2 * count, np.inf)
    for limb_number in range(len(LIMBS)):
        _, bend_index = limb_angle_indices(limb_number)
        lower[bend_index], upper[bend_index] = -max_bend_angle, max_bend_angle
    force_lower = np.full((count, 2), -force_limit)
    force_upper = np.full((count, 2), force_limit)
    force_lower[is_foot, 1] = 0.0                       # feet cannot pull (negative y)
    lower[POSE_VECTOR_LENGTH:], upper[POSE_VECTOR_LENGTH:] = force_lower.ravel(), force_upper.ravel()

    # Initialize the initial forces
    initial_forces = np.zeros((count, 2))
    initial_forces[:, 1] = min(1.0 / count, force_limit)   # share the weight evenly, straight up
    initial = np.clip(np.concatenate([reference, initial_forces.ravel()]), lower, upper)

    # Attempt to find a solution with least_squares
    solution = least_squares(residuals, initial, bounds=(lower, upper), x_scale="jac")
    if not solution.success:
        raise RuntimeError(f"Gravity IK did not converge: {solution.message}")

    # Build variables/fields for a GravityIKResult object
    pose, forces_bw = split(solution.x)
    joints = forward_kinematics(pose, body_lengths)
    ends = np.array([joints.limbs[limb].end for limb in contacted])
    com = center_of_mass(pose, body_lengths, segment_masses, com_fractions)
    forces_n = forces_bw * weight_n
    net_force, net_moment = static_balance_error(com, ends, forces_n, mass_kg, gravity)
    friction_excess_n = np.maximum(0.0, np.abs(forces_n[is_foot, 0]) - foot_friction * forces_n[is_foot, 1])
    tolerance_n = balance_tolerance * weight_n
    balanced = bool(
        np.linalg.norm(net_force) <= tolerance_n
        and abs(net_moment) <= tolerance_n * torso_length
        and (friction_excess_n.size == 0 or friction_excess_n.max() <= tolerance_n)
    )

    return GravityIKResult(
        pose_vector=pose,
        contact_errors_cm={limb: float(np.linalg.norm(ends[i] - targets[i])) for i, limb in enumerate(contacted)},
        contact_forces_n={limb: forces_n[i].copy() for i, limb in enumerate(contacted)},
        center_of_mass_cm=com,
        net_force_n=net_force,
        net_moment_ncm=net_moment,
        balanced=balanced,
    )
