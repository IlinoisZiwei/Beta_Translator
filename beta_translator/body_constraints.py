"""Shared constraint contract for all trials (body_constraint_specification.md).

This file owns bounds, thresholds and coverage declarations. Numerical checks
live in body_checks/body_support; trial files contain observations and geometry.
Unimplemented or uncalibrated physiology is UNKNOWN, never a silent pass.
"""
from dataclasses import dataclass
import numpy as np

ARM_NAMES = ('elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion',
             'pro_sup', 'deviation', 'flexion')
ARM_LIMITS_DEG = np.array([[-90,130], [0,180], [-90,20], [0,130],
                          [-90,90], [-10,25], [-70,70.]])
ARM_SPEED_DEG_S = np.array([120,150,120,180,180,120,150.])
ARM_ACCEL_DEG_S2 = np.array([600,750,600,900,900,600,750.])

# Assumed screening ranges within the pinned source coordinate domain.
# Source ranges are NOT human physiological limits; see lower_limb_source.json.
LEG_NAMES = ('hip_flexion','hip_adduction','hip_rotation','knee_flexion','ankle_dorsiflexion','subtalar')
LEG_LIMITS_DEG = np.array([[-30,120],[-60,35],[-50,50],[0,120],[-50,30],[-20,25.]])
LEG_SPEED_DEG_S = np.array([150,120,120,180,120,90.])
LEG_ACCEL_DEG_S2 = np.array([750,600,600,900,600,450.])


@dataclass(frozen=True)
class Limits:
    """Project numerical thresholds; not individualized human capabilities."""
    contact_cm: float = .1
    bone_cm: float = 1e-6
    penetration_cm: float = 1.
    foot_friction: float = .8
    force_component_bw: float = 1.
    reserve_cm: float = 2.
    departing_force_bw: float = .01
    trusted_confidence: float = .8
    leg_bend_deg: float = 155.
    bend_tolerance_deg: float = .01
    motion_step_cm: float = 8.
    motion_second_difference_cm: float = 4.
    hip_speed_cm_s: float = 100.
    hip_acceleration_cm_s2: float = 350.
    torso_speed_deg_s: float = 90.
    torso_acceleration_deg_s2: float = 450.
    leg_speed_deg_s: float = 240.
    leg_acceleration_deg_s2: float = 1200.


DEFAULT_LIMITS = Limits()


@dataclass(frozen=True)
class Constraint:
    id: str
    section: str
    enforcement: str
    evidence: str
    missing: str = ''


# A complete coverage inventory, not a claim that every mechanism is implemented.
CONSTRAINTS = (
    Constraint('fixed_geometry','3–4','hard','user-profile fixed dimensions or declared height-scaled proportions'),
    Constraint('arm_chain','4','partial','pinned source rotational chain', 'source translations and shoulder girdle'),
    Constraint('arm_rom','5.1','hard','published model coordinate ranges'),
    Constraint('shoulder_envelope','5.3','profile_soft_and_audit','profile plane/elevation envelope in fitting and audit','person-specific loaded envelope calibration'),
    Constraint('wrist_coupling','5.4','not_implemented','scalar wrist bounds only','calibrated coupled flexion/deviation envelope'),
    Constraint('finger_toe_rom','5.4','not_implemented','no fingers or toes','task-specific articulated digit model'),
    Constraint('lower_limb_chain','4.3','hard','source hip/knee/ankle/subtalar rotations and coupled knee translation'),
    Constraint('lower_limb_rom','4.3,5.4','hard','assumed signed hip/knee/ankle/subtalar screening bounds'),
    Constraint('lower_limb_personal_rom','5.4','not_implemented','fixed screening boxes','person-specific coupled loaded lower-limb envelope'),
    Constraint('spine_neck_rom','5.4','not_implemented','rigid torso/head proxy','segment-relative spine/neck model'),
    Constraint('coordinate_rates','6.1','hard','project speed/acceleration assumptions'),
    Constraint('rotation_rates','6.2','parameterized_audit','rotation-matrix rates','justified segment rate caps'),
    Constraint('continuous_time_limits','6.2','not_implemented','native-frame checks only','substep or analytic derivative certificate'),
    Constraint('shared_contact_regions','7.1','hard','image-derived regions; fixed within episode'),
    Constraint('physical_patch_contact','7.1–7.2','soft_and_audit','same patch in objective, audit and video'),
    Constraint('hand_orientation','7.2','partial','source wrist chain; palm orientation inferred','measured grip/patch orientation'),
    Constraint('foot_orientation','7.2','partial','articulated foot frame with bounded ankle/subtalar coordinates','measured shoe orientation and contact-mode validation'),
    Constraint('contact_pose_slip','7.2','not_implemented','persistent patch position only','surface-normal compatibility, patch overlap and slip law'),
    Constraint('contact_complementarity','7.3','partial','soft gap-force product only','rigid complementarity or declared compliance'),
    Constraint('free_contact_force','7.3','hard','free/unknown contacts exert zero optimized force'),
    Constraint('friction_and_support','7.3','soft_and_audit','assumed foot cone and hand force box'),
    Constraint('grip_wrench','7.3','not_implemented','point-contact force model','hold-specific grip wrench capacity'),
    Constraint('release','7.3','soft_and_audit','remaining-support audit plus departing-force check'),
    Constraint('trusted_support','2.3','audit','source/confidence/ground eligibility'),
    Constraint('balance_reserve','2.3','audit','four declared COM perturbations'),
    Constraint('wall_floor','8','partial','joint penetration and conditional scene','whole-body environment collision'),
    Constraint('self_collision','8','profile_soft_and_audit','capsule clearance','body radii and collision-pair profile'),
    Constraint('swept_collision','8','not_implemented','sample checks are insufficient','continuous collision coverage'),
    Constraint('torque_capacity','9.2','profile_soft_and_audit','directional net joint-moment capacity in fitting and audit','actuator-space validation and personal capacity calibration'),
    Constraint('musculotendon','9.3','not_implemented','no muscle paths or tendon states','verified muscle model and person parameters'),
    Constraint('fatigue','10','profile_soft_and_audit','causal joint-actuator occupancy proxy coupled to capacity','muscle model, calibrated controller/rates and pre-clip state'),
    Constraint('strength_fatigue_coupling','9–10','profile_soft_and_audit','causal 3CC joint-actuator capacity coupling','personal capability calibration'),
    Constraint('parameter_calibration','13','not_implemented','height and population/project proxies','person-specific geometry, mass, inertia and capacity evidence'),
    Constraint('centroidal_dynamics','11','profile_soft_and_audit','segment inertia and external-wrench residuals in fitting and audit','measured mass/inertia and independent dynamic validation'),
    Constraint('joint_dynamics','11','parameterized_audit','floating-base equation residual','mass matrix, bias, contact Jacobian and actuation'),
    Constraint('impact_forward_validation','11.3','not_implemented','no forward dynamic replay','impact law, integration and replay validation'),
    Constraint('candidate_selection','12','gate','feasibility before motion/effort; uncertainty retained'),
    Constraint('provenance','13','audit','input/code/config hashes and independent saved-result checks'),
)


def coverage_table():
    """Human-readable contract; actual run outcomes are recorded separately."""
    return [dict(id=c.id, specification_section=c.section, enforcement=c.enforcement,
                 evidence=c.evidence, missing=c.missing,
                 status='unknown' if c.missing or c.enforcement=='not_implemented' else 'available')
            for c in CONSTRAINTS]


def require_complete(results):
    """Full specification acceptance cannot bypass unknown or untested checks."""
    rows = {r['id']:r for r in results}
    if len(rows)!=len(results) or set(rows) != {c.id for c in CONSTRAINTS}:
        raise ValueError('Every specified constraint needs an explicit result')
    statuses = [r['status'] for r in rows.values()]
    if any(s not in ('pass','fail','unknown','not_applicable') for s in statuses):
        raise ValueError('Invalid constraint status')
    return 'fail' if 'fail' in statuses else ('unknown' if 'unknown' in statuses else 'conditional_pass')
