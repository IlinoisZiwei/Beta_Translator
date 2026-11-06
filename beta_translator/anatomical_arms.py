"""Compatibility imports for historical anatomical experiments.

New trials use BodyModel; the implementation lives in body_kinematics.
"""
from .body_kinematics import (ARM_NAMES, ARM_LIMITS_DEG, ARM_SPEED_DEG_S,
    ARM_ACCEL_DEG_S2, QDIM, AXES, OSIM_TO_BODY, MIRROR, axis_rotation,
    arm_frames, kinematics, bounds_and_rates, initialize, rotation_steps_deg,
    elbow_swivel_rates)
