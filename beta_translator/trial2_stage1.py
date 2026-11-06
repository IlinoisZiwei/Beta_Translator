"""Historical Stage1 API; body/contact logic is now shared across trials."""
from .body_reference import source_reference, source_curve, coupled_arm_frames, stage1_kinematics, verify_chain
from .body_contacts import foot_frames
from .body_articulation import PatchProblem as Stage1Problem, audit_body as audit_stage1
