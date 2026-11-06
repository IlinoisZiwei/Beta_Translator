# Beta Translator Body Model and Constraint Specification

**Personalized-model update (November 5, 2026):** The shared entry now reads an editable 175 cm / 176 cm arm-span user profile and actively fits/audits shoulder envelopes, capsule collision, joint-moment capacity, fatigue coupling and inertial balance. Defaults are uncalibrated assumptions. Earlier descriptions and results below remain historical; see [current implementation and boundaries](personalized_body.md).

**Scope:** existing body constraints and the proposed anatomical and dynamic model for climbing. **Status:** specification only; proposed capabilities below are not implemented by this document. **Code review date:** November 4, 2026, America/New_York, against the local working tree based on commit `addd352`, including uncommitted Trial2 updates.

The current system fits fixed-length skeletons to uncertain climbing observations and checks simplified support. Trial2 now has a reduced, bounded shoulder–elbow–forearm–wrist chain. It does not yet establish that the complete movement is anatomically or dynamically executable. The proposed model adds pose-dependent joint limits, oriented hands and feet, collision geometry, strength, fatigue, and whole-body dynamics without replacing missing measurements with an automatic pass.

The specification is independent of a particular notebook. **Shared implementation update:** Trial1, Trial2 and Trial3 now use the same reduced anatomical chain and implemented constraints through `BodyModel`. See [shared model ownership, coverage and verification](shared_body_model.md). The historical implementation survey below describes the pre-refactor experiments; it is not a claim that the proposed physiological stages are complete. Every new run reports all specification categories, including explicit unknown coverage.

**Stage 1 implementation update:** [Trial2 Stage 1](trial2_stage1_contacts.md) now verifies the reduced rotational chain against a pinned source reference, restores distal wrist rotation coupling, and shares one hand/foot patch definition across fitting and support audits. Its 180 cm short-clip experiment is separate from the historical model described below. Source translations/girdle motion, anatomical ankle ROM and the later physiological stages remain unimplemented.

**Lower-limb update (November 5, 2026):** The shared model now implements bilateral anatomical hip/knee/ankle/subtalar rotations, coupled source knee translation, articulated shoe patches and hard assumed lower-limb coordinate/rate bounds. See [implementation, source conventions and 175 cm verification](anatomical_lower_limbs.md). The historical coverage descriptions below predate this update; personalized coupled lower-limb ROM and the remaining physiological stages are still unknown.

## Contents

1. [Status and evidence conventions](#1-status-and-evidence-conventions)
2. [Current implementation and code locations](#2-current-implementation-and-code-locations)
3. [Body state and coordinate conventions](#3-body-state-and-coordinate-conventions)
4. [Anatomical joint chain](#4-anatomical-joint-chain)
5. [Joint range of motion](#5-joint-range-of-motion)
6. [Velocity acceleration and continuity](#6-velocity-acceleration-and-continuity)
7. [Hands feet and contact constraints](#7-hands-feet-and-contact-constraints)
8. [Body and environment collision](#8-body-and-environment-collision)
9. [Joint torque and muscle capacity](#9-joint-torque-and-muscle-capacity)
10. [Muscle fatigue and recovery](#10-muscle-fatigue-and-recovery)
11. [Dynamic balance with inertia](#11-dynamic-balance-with-inertia)
12. [Optimization and candidate selection](#12-optimization-and-candidate-selection)
13. [Required inputs and validation](#13-required-inputs-and-validation)
14. [Implementation order and acceptance criteria](#14-implementation-order-and-acceptance-criteria)
15. [References and source boundaries](#15-references-and-source-boundaries)

## 1 Status and evidence conventions

Implementation status and scientific validity are separate fields.

| Label | Meaning |
| --- | --- |
| Implemented hard constraint | Enforced by the current parameterization, variable bounds, or linear constraints, subject to numerical tolerance and the stated sampling grid. |
| Implemented soft objective | Penalized during fitting. It can still be violated in the exported trajectory. |
| Implemented audit | Checked after fitting. An audit does not necessarily constrain the optimization or affect ranking. |
| Partial or proxy | Represents a simplified part of the mechanism. Its result must retain that qualification. |
| Proposed | Defined here for future implementation; not available in the current solver. |

Each parameter must also identify its evidence: **measured for this person/scene**, **published model parameter**, **population reference**, **project assumption**, or **unknown**. A number copied from a musculoskeletal model is not automatically the limit of a particular climber.

Future audit results must distinguish `pass`, `fail`, `unknown`, and `not_applicable`. Missing palm observations, strength parameters, or inertia cannot produce a physiological pass. Solver convergence, numerical feasibility under assumptions, and experimental validation are separate outcomes.

## 2 Current implementation and code locations

### 2.1 Model families

| Existing component | Current capability | Boundary |
| --- | --- | --- |
| Planar IK and gravity IK | An 11-coordinate body, fixed lengths, generic signed bend bounds, endpoint fitting, planar force/moment balance. | A 2D model cannot establish anatomical shoulder rotation or depth-dependent collision. The single-pose IK default bend bound is ±150°; sequence configurations must be checked separately. |
| Trial1 refined support and techniques | Conditional ground support, remaining-support checks, temporal optimization, exact singleton COM-x constraints in the refined solver, hand-load and arm-moment proxies. | Ground geometry and contact timing remain uncertain. Simplified arm moments are not strength limits or muscle fatigue. |
| Trial3 and earlier Trial2 spatial fitting | 3D wall hypotheses, fixed-bone fitting, contact regions/offsets, conditional 3D quasi-static support. | Generic limb directions and bend checks are not a complete anatomical joint chain. |
| Latest Trial2 anatomical experiment | 28 pose coordinates, bilateral seven-coordinate arms, hard sampled coordinate speed/acceleration bounds, shared contact regions and global trajectory fitting. | Lower limbs remain direction-based; shoulder limits are independent scalar bounds; wrist orientation is inferred. |

### 2.2 Implementation map

Links below are repository-relative so the specification remains usable after cloning.

| File and symbol | Responsibility |
| --- | --- |
| [inverse_kinematics.py](../beta_translator/inverse_kinematics.py), `RATIOS`, `make_body`, `solve_inverse_kinematics`, `solve_inverse_kinematics_with_gravity` | Length scaling, planar kinematics, generic bend bounds, gravity and COM assumptions. |
| [anatomical_arms.py](../beta_translator/anatomical_arms.py), `arm_frames`, `kinematics`, `bounds_and_rates` | Current arm hierarchy, local axes, fixed segment geometry, ROM and temporal caps. |
| [trial2_anatomy_optimizer.py](../beta_translator/trial2_anatomy_optimizer.py), `AnatomicalProblem`, `audit`, `selection_row`, `rank_candidates` | Hard constraints, soft residuals, audits and current anatomical candidate policy. |
| [temporal_control_basis.py](../beta_translator/temporal_control_basis.py), `ControlledProblem` | Whole-sequence cubic joint-coordinate parameterization. |
| [restored_sequence_solver.py](../beta_translator/restored_sequence_solver.py), `solve` | Constrained optimization, feasibility restoration and convergence checks. |
| [trial2_spatial_optimizer.py](../beta_translator/trial2_spatial_optimizer.py), `force_audit`, `audit_plan` | Independent quasi-static force LP, matched-contact gating, reserve and pre-release checks. |
| [trial2_spatial_data.py](../beta_translator/trial2_spatial_data.py), `episode_regions`; [trial2_scene3d.py](../beta_translator/trial2_scene3d.py), `Trial2Scene` | Contact rectangles, offsets, conditional ground targets, camera and joined wall panels. |
| [trial2_reach_diagnostics.py](../beta_translator/trial2_reach_diagnostics.py), `target_reach_diagnostics` | Necessary span bound for the selected contact targets. |
| [support_refinement.py](../beta_translator/support_refinement.py); [technique_planner.py](../beta_translator/technique_planner.py) | Trial1 refined support, technique preferences and simplified arm-load comparisons. |
| [anthropometry.md](anthropometry.md); [trial2_anatomical_constraints.md](trial2_anatomical_constraints.md) | Existing parameter provenance and Trial2 experiment details. |

### 2.3 Current Trial2 constraints and checks

| Item | Current rule | Enforcement and limitation |
| --- | --- | --- |
| Segment geometry | Fixed lengths; rigid shoulder and hip widths. | Hard through forward kinematics. No independent elbow-position variable. |
| Arm ROM | Seven bounds per arm in Section 5. | Hard; no pose-dependent shoulder envelope. |
| Coordinate rates | First and second differences bounded at native 30 fps. | Hard at sampled frames; Section 6 defines scope. |
| Persistent contact location | One normalized `(u,v)` pair per contact episode, within `[0,1]²`; degenerate regions stay fixed. | Hard shared location in an assumed static scene. It is not a measured no-slip hand/foot pose. |
| Contact evidence | Hold regions are rectangles contained in the selected hold mask/panel; sparse evidence may collapse them to a point. | Inferred region, not a measured 3D grip surface. |
| Wrist/ankle target fitting | Endpoint-to-target residual; independent failure threshold **0.1 cm**. | Soft during fitting, strict audit afterwards. The threshold is numerical agreement, not video accuracy. |
| Hand contact proxy | Articulated hand connector, with a separate **2 cm** palm-proxy gap check. | Soft plus audit; not finger grip or measured palm orientation. |
| Force bounds | Each active Cartesian force component lies in **[−1,+1] BW**. Unknown/free force components equal zero. | Hard. A component bound is not a one-BW bound on the vector norm or personalized strength. |
| Force/moment balance | Sum of normalized forces equals `[0,1,0]`; net moment about COM is zero. | Soft in fitting; independent quasi-static LP on reached contacts. |
| Foot friction | `Fn >= 0`; each tangent component obeys `abs(Ft) <= 0.8/sqrt(2) * Fn`. | Soft in fitting; hard in the audit LP. An assumed inscribed square friction cone. Hands retain broad component bounds without a grip-specific wrench model. |
| Contact-force consistency | Penalize the product of endpoint gap and force. | Soft; not exact contact complementarity. The audit excludes contacts farther than 0.1 cm. |
| Balance reserve | Repeat the force audit after COM shifts of ±2 cm along X and Z. | Audit only. This samples four perturbations; it does not certify all perturbations or dynamic balance. |
| Pre-release support | Penalize departing-limb force; audit whether current remaining matched contacts can support the pose. | Partial. This Trial2 gate does not independently impose a hard near-zero departing-force threshold. Trial1 has additional release-load checks. |
| Wall and floor | Penalize sampled joint penetration; audit fails beyond **1 cm** penetration. | Soft plus audit. No segment-volume or whole-body collision coverage. |
| Generic leg bend | Soft penalty beyond 155°; inherited geometric audit at **155.01°**. | Unsigned segment-angle check, not an anatomical knee hinge or hip/ankle ROM model. |
| Motion warnings | Maximum joint step >8 cm/frame or second difference >4 cm/frame². | Audit, separate from the new coordinate speed/acceleration limits. |
| Source/geometry priors | Visibility-weighted reprojection, weak pose prior, 25 cm hip-depth prior; camera-front penalty begins below 10 cm camera depth. | Soft fitting aids; not human constraints or measurements. |
| Evidence eligibility | Confidence ≥0.8, source observation available, reached contact; inferred ground excluded from the trusted set. | Separate audit. Conditional support can pass using less reliable evidence. |
| Reach conflict | Selected target distance cannot exceed two limb lengths plus their rigid root separation. | Necessary-condition diagnostic. Passing does not prove reachability; failing is conditional on the chosen targets and geometry. |

The present candidate gate includes anatomical/temporal, geometric, conditional support, pre-release, palm-proxy and motion failures, plus convergence. **`eligible_balance` and `robust_2cm` are reported but are not required by that predicate.** Self-collision, strength, fatigue and dynamics are not in that predicate either. Thus `numerically_feasible_under_assumptions` must not be interpreted as complete human feasibility.

Trial-specific details remain in [Trial1 ground support](trial1_ground_support.md), [Trial1 refinement](trial1_refinement.md), [Trial1 techniques](trial1_techniques.md), [Trial3 support optimization](trial3_support_optimization.md), and [Trial3 contact regions](trial3_contact_regions.md). Their thresholds and ranking policies must not be silently replaced by Trial2 defaults.

## 3 Body state and coordinate conventions

### 3.1 Current representation

Trial2 uses `q = [hip_xyz, torso_ZYX, left_arm_7, right_arm_7, leg_directions_8]`. Positions are in **cm**, angles in **radians**, and Y points upward. Torso ZYX coordinates describe the body's world orientation; they are not lumbar joint angles.

Current length ratios multiply the selected height `H`:

| Dimension | Ratio | Length at 180 cm |
| --- | ---: | ---: |
| Upper arm | 0.186 | 33.48 cm |
| Forearm | 0.146 | 26.28 cm |
| Thigh | 0.245 | 44.10 cm |
| Shank | 0.246 | 44.28 cm |
| Shoulder width | 0.259 | 46.62 cm |
| Hip width | 0.191 | 34.38 cm |
| Hip-to-shoulder torso | 0.288 | 51.84 cm |

These are project proportions, not measurements of the user. Height alone does not identify segment proportions, mass, inertia, flexibility or strength. The head is currently a geometric/COM proxy rather than a complete anatomical segment.

The default `legacy_v1` COM calculation uses the following mass fractions. Limb entries are **per side**, and all masses including both sides sum to one.

| Segment or aggregate | Fraction of total mass | Current COM location |
| --- | ---: | --- |
| Head | 0.081 | Shoulder center plus 0.40 of the hip-to-shoulder vector; project proxy. |
| Torso and pelvis | 0.497 | Midpoint of hip and shoulder centers; project proxy. |
| Upper arm | 0.028 each | 0.436 along shoulder to elbow. |
| Forearm plus hand | 0.022 each | 0.69 along elbow to wrist; legacy proxy. |
| Thigh | 0.100 each | 0.433 along hip to knee. |
| Shank plus foot | 0.061 each | 0.57 along knee to ankle; legacy proxy. |

An alternative endpoint-COM profile exists, but existing sequence pipelines still use the legacy defaults. Switching profiles must update optimization, force audits and ranking consistently. These COM coefficients do not define segment inertia tensors; their detailed evidence and approximations are recorded in [anthropometry.md](anthropometry.md).

The broad hip search box is X `[−1000,1000]`, Y `[−30,1600]`, Z `[−1500,1000]` cm. Torso-coordinate boxes are `[−110,110]`, `[−120,120]`, `[−90,90]` degrees. Leg-direction coordinates alternate `[−30,30]` radians and `[−π/2,π/2]`. These are numerical search bounds, **not physiological ROM**.

### 3.2 Proposed common representation

Use a floating pelvis/root, articulated torso/shoulder girdle, bilateral anatomical arms and legs, explicit hand/foot frames, and optional finger/toe joints. Every joint needs a parent, child, anatomical frame, rest transform, axis or joint manifold, sign convention, and parameter source.

A dynamic state contains configuration `q`, generalized velocity `v`, muscle/actuator states, fatigue states and contact mode. With quaternion orientations, `q_dot = N(q) v`; quaternion component differences are not angular velocities. Use SI units internally for dynamics: **m, kg, s, rad, N, N·m, kg·m²**. Convert the current cm and BW quantities at a single documented boundary.

Forces are environment-on-body forces. Define the world gravity vector explicitly, normally `[0, −9.81, 0] m/s²`. Define each contact frame and its outward surface normal. A model imported from OpenSim, MuJoCo or another source must be transformed into these conventions, not merely relabeled. Anatomical frame reporting should follow an explicit convention such as [ISB Part II](https://pubmed.ncbi.nlm.nih.gov/15844264/?otool=icznmelib); this recommendation defines coordinate systems, not universal angle limits.

## 4 Anatomical joint chain

### 4.1 Required hierarchy

```text
World → floating root/pelvis → torso
  → shoulder girdle / scapular frame
  → glenohumeral shoulder rotation
  → humerus
  → elbow flexion/extension about its specified local hinge axis
  → ulna / forearm base
  → radioulnar pronation/supination about its specified local axis
  → radius / distal forearm
  → wrist flexion/extension and radial/ulnar deviation
  → hand frame → selected finger/palm contact patch
```

The elbow axis moves with its parent segment. It is not a fixed world axis. Forearm rotation changes the distal forearm/hand orientation; it is not permission to orbit the elbow around a fixed shoulder–wrist line.

For a parent-to-child joint, use homogeneous transforms of the form

$$
T_{W,C}=T_{W,P}\,T_{P,J}\,T_J(q_J)\,T_{J,C}^{rest}.
$$

`T_P,J` and `T_J,C^rest` retain the joint-center offsets and anatomical rest frames. Chain order is essential because rotations do not commute. A full model must retain the source model's coupled translations and moving axes where present; it cannot assume every real joint is a perfect fixed-axis hinge.

### 4.2 Current reduced implementation

In `anatomical_arms.arm_frames`, `p,e,r,b,s,d,f` denote shoulder plane, elevation, axial coordinate, elbow flexion, pronation/supination, wrist deviation and wrist flexion:

```python
A = R(axis0, p) @ R(axis1, e) @ R(axis0, r - p)
F = A @ R(axis2, b)
H = F @ R(axis3, s) @ R(axis4, d) @ R(axis5, f)
```

This is explanatory notation for the existing `axis_rotation` calls, not a new implementation. The code converts source frames with `OSIM_TO_BODY`, composes torso rotation, and mirrors the left-side frames as `MIRROR @ R @ MIRROR`.

| Axis | Current normalized source direction, approximately | Use |
| --- | --- | --- |
| 0 | `(0.0048, 0.99908918, 0.04240001)` | Shoulder plane and third shoulder rotation |
| 1 | `(−0.99826136, 0.0023, 0.05889802)` | Shoulder elevation |
| 2 | `(0.04940001, 0.03660001, 0.99810825)` | Elbow flexion |
| 3 | `(−0.01716099, 0.99266564, −0.11966796)` | Forearm pronation/supination |
| 4 | `(−0.819064, −0.135611, −0.557444)` | Wrist deviation |
| 5 | `(0.95642673, −0.25220693, 0.14710396)` | Wrist flexion |

The source frame is forward/up/right. `OSIM_TO_BODY` maps forward toward body −Z, up to +Y and right to +X. The code normalizes each axis before using it; these values must not be interpreted as world-space axes for every frame. Source provenance is recorded in [trial2_arm_model_sources.json](../data/trial2_arm_model_sources.json).

The elbow and wrist centers are generated from segment frames:

$$
E=S-L_{upper}A\,e_y,\qquad W=E-L_{forearm}F\,e_y.
$$

Pronation and wrist coordinates affect `H`, not `E` or `W`, in this reduced model. The shoulder coordinate `r` also enters through `r-p`; it must not be equated without calibration to a clinical internal/external-rotation reading.

**Remaining anatomical omissions:** scapulothoracic rhythm, clavicle/scapula motion, source shoulder translations, non-collinear bone offsets, explicit radius/ulna geometry, ligaments and detailed carpal coupling. The current chain is MoBL-inspired, not a complete MoBL-ARMS import. The [MoBL benchmarking publication](https://pubmed.ncbi.nlm.nih.gov/24995410/) also demonstrates why muscle-model and actuator-path differences matter when transferring models between simulators.

### 4.3 Proposed lower-limb chain

```text
Pelvis → hip rotation → femur → anatomical knee
  → tibia → talocrural ankle → subtalar coupling
  → foot frame → heel / forefoot / shoe-edge patch
  → optional metatarsophalangeal toe motion
```

Define signed knee flexion and an appropriate coupled knee/ankle model rather than two independent world-direction vectors. Tibial rotation, knee translation and subtalar motion must follow the chosen anatomical model. Adding arbitrary independent rotations can recreate the same degeneracy that caused the old elbow swivel.

## 5 Joint range of motion

### 5.1 Existing arm limits

The ranges come from the [pinned MoBL-derived OpenSim snapshot](https://github.com/aikkala/O2MConverter/blob/ad2599d5ecb2c18b44570b4d0b53e062ac9b82ae/models/opensim/MoBL_ARMS_OpenSim_tutorial_33/MoBL_ARMS_model_for_mujoco.osim). They are bounds of that model, not a universal human ROM table. The speed and acceleration columns are **project assumptions**, not values established by MoBL.

| Coordinate | Hard range, degrees | Speed cap, deg/s | Acceleration cap, deg/s² |
| --- | ---: | ---: | ---: |
| Shoulder elevation plane `elv_angle` | −90 to 130 | 120 | 600 |
| Shoulder elevation `shoulder_elv` | 0 to 180 | 150 | 750 |
| Shoulder axial coordinate `shoulder_rot` | −90 to 20 | 120 | 600 |
| Elbow flexion | 0 to 130 | 180 | 900 |
| Forearm pronation/supination | −90 to 90 | 180 | 900 |
| Wrist deviation | −10 to 25 | 120 | 600 |
| Wrist flexion/extension | −70 to 70 | 150 | 750 |

Do not replace the anatomical elbow bound with the generic 155° spatial bend check. Do not infer wrist radial/ulnar or flexion/extension sign labels from the numeric signs alone without verifying the rest-frame mapping for each side.

### 5.2 Population ROM is reference evidence

The [CDC reference table](https://archive.cdc.gov/www_cdc_gov/ncbddd/jointrom/index_1704899060.html) reports age/sex-specific values. Selected **ages 20–44** values in degrees are shown for context; they are not installed solver limits.

| Measured motion | Female reference value | Male reference value |
| --- | ---: | ---: |
| Shoulder flexion | 172.0 | 168.8 |
| Elbow flexion | 150.0 | 144.6 |
| Forearm pronation | 82.0 | 76.9 |
| Forearm supination | 90.6 | 85.0 |
| Hip extension | 18.1 | 17.4 |
| Hip flexion | 133.8 | 130.4 |
| Knee flexion | 141.9 | 137.7 |
| Ankle dorsiflexion | 13.8 | 12.7 |
| Ankle plantar flexion | 62.1 | 54.6 |

These reference values and the source's confidence intervals do not define an individual's admissible interval. Measurement posture, passive versus active motion, loading and coordinate convention matter. Shoulder flexion in this table is not automatically the same variable as `shoulder_elv`. Neither this table nor the current source snapshot supplies a complete individualized, loaded climbing ROM model.

### 5.3 Pose-dependent shoulder internal/external rotation

**Proposed constraint:** replace independent shoulder boxes with a calibrated admissible set. One implementation can use

$$
r_{min}(p,e,q_{sc},\theta)\le r\le r_{max}(p,e,q_{sc},\theta),
$$

where `q_sc` describes the shoulder girdle and `θ` identifies the anatomical/subject profile and measurement conditions. If an angular chart becomes singular, evaluate an equivalent validity function on relative rotations and the relevant pose state, rather than trusting unstable Euler coordinates.

A cadaveric study measured different axial-rotation envelopes across abduction levels and planes. Its loading protocol is evidence for pose dependence, not a loaded climbing limit for this user. [Southgate et al., 2009](https://pubmed.ncbi.nlm.nih.gov/19375080/).

Required implementation properties:

- Map measured humerus-relative-to-scapula orientation into the model's coordinates before fitting bounds.
- Store the sampled posture domain and uncertainty. Interpolate only within supported regions; unsupported posture combinations return `unknown`.
- Preserve left/right conventions and report distance to the admissible-set boundary.
- Use a smooth validity function for optimization, followed by an independent boundary check.
- Treat learned pose priors as evidence about sampled poses, not proof of personal strength or an exhaustive anatomical envelope. [Akhter and Black's pose-conditioned limits](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Akhter_Pose-Conditioned_Joint_Angle_2015_CVPR_paper.pdf) provide a relevant methodology, not a drop-in shoulder axial-limit table.

There is currently no validated `r_min/r_max` surface in the repository. It must remain unspecified rather than being filled with an invented universal curve. Shoulder-girdle coupling also needs a chosen model; do not impose a universal fixed scapulohumeral ratio.

### 5.4 Elbow forearm wrist and lower-body coupling

The elbow must have signed flexion about the declared local anatomical axis. Hyperextension should require an explicit profile, not arise from an elbow branch flip. Pronation/supination belongs to the radioulnar chain and must preserve the intended elbow mechanism.

Wrist flexion/deviation may need a coupled admissible region, rather than allowing every corner of two independent intervals. Hip ROM can depend on knee position; knee/tibial and ankle/subtalar coupling require their own model definitions. Spinal articulation needs segment-relative limits, separate from the body's global lean.

| Anatomical region | Required range/coupling profile beyond the current arm table | Current coverage |
| --- | --- | --- |
| Shoulder girdle | Clavicle/scapula orientation and any coupled translation relative to thorax | Absent; shoulder roots currently follow a rigid torso. |
| Wrist | Coupled flexion–deviation envelope, with forearm/contact context where supported | Independent scalar bounds only. |
| Hip | Flexion/extension, abduction/adduction and axial rotation, including posture dependence | No anatomical hip-coordinate limits. |
| Knee | Signed flexion/extension, declared hyperextension policy and any tibial coupling | Generic unsigned segment-bend audit only. |
| Ankle and subtalar complex | Dorsiflexion/plantarflexion, inversion/eversion and model-specific coupling | No articulated ankle/foot chain. |
| Toe and fingers | Task-relevant joint limits and contact mechanics when the selected mode requires them | Not represented; detailed hooks/pinches cannot be validated. |
| Spine and neck | Segment-relative ROM and coupling if articulated; otherwise explicitly fixed/proxy | Rigid torso and head proxy, not measured spinal/neck limits. |

No loaded wrist envelope, anatomical hip/knee/ankle envelope, or spinal ROM model has been validated in this project. Their numerical profiles are required inputs for the corresponding future checks; absence must remain visible.

## 6 Velocity acceleration and continuity

### 6.1 Current sampled constraints

For frame spacing `Δt = 1/30 s`, the implemented constraints are

$$
\frac{|q_{k+1,j}-q_{k,j}|}{\Delta t}\le v_j^{max},\qquad
\frac{|q_{k+2,j}-2q_{k+1,j}+q_{k,j}|}{\Delta t^2}\le a_j^{max}.
$$

| Coordinates | Speed cap | Acceleration cap |
| --- | ---: | ---: |
| Hip translation, each component | 100 cm/s | 350 cm/s² |
| Torso orientation, each coordinate | 90°/s | 450°/s² |
| Each arm | Section 5.1 | Section 5.1 |
| Legacy leg-direction coordinates | 240°/s | 1200°/s² |

These componentwise limits do not bound the norm of global segment angular velocity. The audit uses a relative tolerance of `1e-4` for rate comparisons and `1e-7` for coordinate-bound excess. Initial frames without a first/second-difference history are padded with zero in the current report; that does not establish pre-video motion.

Additional soft objectives smooth joint coordinates, Cartesian joint positions and physical rotation matrices. The full Trial2 model uses 125 clamped cubic controls per coordinate, with eight-frame knot spacing, for all 976 observations. Bounded controls preserve scalar ROM through the nonnegative basis. Sampled rate/acceleration checks do **not** establish continuous-time derivative bounds between frames.

### 6.2 Proposed temporal constraints

Use actual timestamps. For segment rotation matrices, an interval angular-velocity diagnostic can be computed from

$$
\omega_k=\operatorname{vee}\!\left(\log(R_k^T R_{k+1})\right)/\Delta t.
$$

Successive angular velocities must be expressed in a common frame before differencing to estimate angular acceleration. Also audit joint-relative velocities, since global segment motion contains torso/parent motion. Quaternion sign changes and Euler branch wrapping must not become artificial motion spikes.

Validate trajectories at substeps or by derivative bounds on the chosen curve representation. Add jerk regularization only as a declared motion preference unless a justified jerk limit is available. Separate stance, release, swing and arrival; an impact model may allow a velocity jump and impulse. Global smoothness must not erase that event or pretend a finite acceleration bound applies across an ideal instantaneous impact.

Elbow swivel remains a diagnostic, not a joint coordinate. Its axis becomes ill-conditioned near a straight arm. Report masked intervals and compare common observable intervals; `NaN` is not zero rotation.

## 7 Hands feet and contact constraints

### 7.1 Current offsets versus oriented contact geometry

Trial2 maps a surface contact `c` to a wrist/ankle target `c + offset`. When evidence is sufficient, nominal hand offsets use `0.035H` tangential and `0.012H` normal components; foot offsets use `0.045H` and `0.025H`. Ground proposals use an upward `0.035H` offset. Sparse non-ground evidence can produce zero offsets. These are inferred/project quantities, not measured hand or shoe geometry.

The new hand proxy is instead generated through the wrist frame:

$$
p_{tip}=W-0.035H\,R_{hand}e_y,\qquad
p_{palm}=p_{tip}-0.012H\,R_{hand}e_z.
$$

The inherited target offset and articulated proxy are not guaranteed to agree. The two residuals are reported separately. Neither represents detailed finger closure.

### 7.2 Proposed hand and foot pose constraints

Each contact needs a position **and** a local orientation, a selected patch, a contact mode, and confidence. For a local patch point `r_patch`,

$$
p_{contact}(q)=p_{segment}(q)+R_{segment}(q)r_{patch}.
$$

An orientation residual may use `log(R_contact,target^T R_contact(q))`, but constrain only observable, mechanically justified components. A flat sole patch can align its outward normal opposite the floor normal. A fingertip, edge, pinch, undercling or heel hook requires its own geometry and admissible wrench set; full palm-to-wall alignment is not a universal climbing constraint.

For feet, define heel, forefoot, lateral shoe edge and sole-normal landmarks in a rigid foot frame. The ankle position alone cannot identify them. A heel–toe line alone still leaves rotation about that line ambiguous. Ankle/subtalar ROM must constrain how the foot frame relates to the shank.

Required checks include patch position, normal/direction compatibility, tangential slip, local patch containment, shoe/hold overlap and permitted contact moment. Store these separately from ankle/wrist target error.

### 7.3 Contact forces and transitions

For an ordinary unilateral compressive contact, a proposed rigid-contact formulation is

$$
\phi\ge0,\quad f_n\ge0,\quad \phi f_n=0,\quad
\|f_t\|_2\le\mu f_n,
$$

where `φ` is separation. A compliant formulation may be used instead, with its deformation, damping and tolerances declared. Do not mix a compliant simulation with claims of exact rigid complementarity.

Sticking requires compatible contact velocity; sliding needs a declared friction law. A finite patch may support moments, constrained by patch geometry and pressure/friction. A point contact has no arbitrary free moment. Pulling hand contacts require a grip-specific mechanical model or explicit conditional force envelope; they must not obtain unlimited tensile force from an ordinary foot-style cone.

Before release, transfer load to currently available support and reduce the departing contact's force below a declared threshold. At release, that support is removed. Do not use the future target hold before arrival. Trial2's current remaining-support LP alone is not enough to establish this dynamic transfer.

Ground contact must be supported by floor geometry, foot proximity/orientation and temporal evidence. **Not on a wall hold does not imply on the ground.** Uncertain ground proposals remain conditional. Preserve raw observations separately from the effective contact schedule.

## 8 Body and environment collision

**Current status:** sampled joint-to-wall/floor penalties; no complete body self-collision check.

**Proposed representation:** capsules or other convex volumes for limb segments, a torso/pelvis volume, head geometry, and oriented hand/foot geometry. For non-exempt capsule pairs,

$$
d_{segments}(q)-(r_i+r_j)\ge d_{clearance}.
$$

`d_clearance`, radii and shape uncertainty must be explicit. Check arms against torso/head, left against right limbs, thighs/shanks, and moving hands/feet against the body and wall. Exempt only documented adjacent-joint overlaps or allowed contact regions. Globally disabling arm–torso or left–right pairs would defeat the check.

Use a soft barrier to guide fitting and an independent signed-distance audit for acceptance. Validate swept motion between samples to catch tunneling. Wall/hold meshes need suitable collision geometry; a rendered mesh is not necessarily the geometry tested by the engine. [MuJoCo's collision documentation](https://mujoco.readthedocs.io/en/3.3.3/computation/#collision-detection) describes collision filtering and supported geometric representations.

Allowing hand-to-hold contact does not permit the forearm to pass through the wall. Matching or crossing limbs also requires volume and timing checks, not just two successful endpoint fits.

## 9 Joint torque and muscle capacity

### 9.1 Existing proxies

Trial1 can compare simplified shoulder/elbow moments from contact forces and modeled segment weights. The model lacks muscle moment arms, co-contraction, personalized capacity and inertia. Its hand-force and moment integrals are effort proxies, not endurance, metabolic energy or measured muscle forces. Trial2's ±1 BW component bounds likewise do not establish joint capacity.

### 9.2 Proposed torque-driven model

Compute required generalized actuation from consistent whole-body inverse dynamics and contact forces. Apply direction-, posture- and velocity-dependent capacity:

$$
\tau_j^{-}(q,v,z,\theta)\le\tau_j^{act}\le\tau_j^{+}(q,v,z,\theta),
$$

where `z` includes fatigue/history and `θ` is the capacity profile. Positive and negative capacities need not be symmetric. Record utilization and remaining reserve, with separate treatment of passive torque and active capacity.

A torque-capacity profile needs strength data or a documented source model, joint-coordinate mapping, subject scaling assumptions and uncertainty. **No universal N·m limits are assigned in this specification.** A static maximum cannot be applied unchanged at every posture, speed and fatigue state. Independent torque boxes are a useful approximation but can overestimate simultaneous capacity across joints sharing muscles.

### 9.3 Proposed muscle-driven model

For a chosen muscle/tendon model,

$$
\tau^{MT}=\mathcal{R}(q)F_{MT},\qquad
F_{fiber}=F_{max}\,[a_{eff}f_L(\ell)f_V(\dot\ell)+f_{passive}(\ell)].
$$

`ℛ(q)` is the signed moment-arm matrix. Tendon compliance, pennation and fiber–tendon equilibrium determine transmission to `F_MT`; the displayed equation is a structural description, not a replacement for the selected model's equations. Here `τ_MT` includes the selected musculotendon model's active and passive contributions. When substituting it into whole-body dynamics, count each passive contribution once: do not include the same passive fiber force again in a separate passive-torque term. Activation, excitation, normalized length/velocity and force conventions must be consistent. [Millard et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3705831/) provides benchmarked musculotendon formulations.

Required inputs include muscle paths, maximum isometric force, optimal fiber length, tendon slack length, pennation and activation parameters. Preserve multiarticular coupling and avoid double-counting muscle-generated torque as a separate unrestricted actuator. Net joint torque does not determine co-contraction or joint reaction force uniquely. Mechanical power `τ·v` is not, by itself, metabolic expenditure.

## 10 Muscle fatigue and recovery

**Current status:** no fatigue state or history-dependent strength reduction. Hand-load accumulation is not a fatigue model.

A proposed three-compartment formulation tracks active `M_A`, resting `M_R` and fatigued `M_F` motor-unit fractions:

$$
M_A+M_R+M_F=1,\quad M_A,M_R,M_F\ge0,
$$

$$
\dot M_A=C(t)-k_fM_A,\quad
\dot M_R=-C(t)+k_r^{eff}M_F,\quad
\dot M_F=k_fM_A-k_r^{eff}M_F.
$$

Recruitment controller `C(t)`, fatigue/recovery rates and any rest multiplier must come from one explicitly selected formulation. [MyoSuite's fatigue documentation](https://myosuite.readthedocs.io/en/latest/fatigue_validation.html) describes a 3CC-r implementation and its validation context. Pin the implementation/version when adopting it; this specification does not copy its numerical defaults into the project.

Required integration rules:

- Carry fatigue continuously across phases and candidate segments; do not reset it at each hold or frame.
- Compare candidates from identical initial fatigue states and with their actual durations.
- Couple fatigue to available active force/torque. A fatigue score that never changes capacity is only an annotation.
- Use the chosen formulation's activation semantics. Do not multiply activation and an active-motor-unit fraction twice if they already represent the same recruitment effect.
- Model recovery explicitly; an arm being visually straight or temporarily unloaded does not automatically imply complete recovery.
- Save state trajectories, initial conditions, parameters and time step. Check conservation, non-negativity and integration stability.

OpenSim also provides a [custom fatigable-muscle example](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53088956/Custom%2BMuscle%2BModel%2BPart%2BOne). It has its own state/activation definition and must not be mixed with 3CC equations without deriving the correspondence. Neither example validates personal climbing endurance. Finger-flexor/grip fatigue needs representation if claims concern sustained gripping.

## 11 Dynamic balance with inertia

### 11.1 Current model

Current balance is quasi-static: contact forces must balance gravity and net moment. Temporal smoothing or acceleration limits do not add inertial forces. The ±2 cm COM audit is a local quasi-static sensitivity check, not a dynamic stability criterion.

### 11.2 Proposed whole-body equations

Use floating-base dynamics with contact wrenches. The standard multibody structure is described in [Tedrake’s multibody dynamics notes](https://underactuated.csail.mit.edu/multibody.html); the following is the proposed Beta Translator formulation:

$$
M(q)\dot v+h(q,v)=S^T\tau^{act}+\tau^{passive}+J_c(q)^T\lambda+\tau^{other}.
$$

`M` is generalized inertia; `h` includes gravity, Coriolis and centrifugal terms; `S` selects actuated coordinates; `J_c` maps generalized velocities to the selected contact coordinates. The floating root is unactuated. A free pelvis/root force must not compensate for missing support. Other external forces, such as rope forces, require explicit inputs if relevant.

The corresponding centroidal checks are

$$
m\ddot r_{COM}=m g+\sum_i f_i+f_{other},
$$

$$
\dot L_{COM}=\sum_i[(c_i-r_{COM})\times f_i+m_i]+m_{other},
$$

where `m_i` denotes a contact moment, not segment mass. Force and momentum abstractions are also discussed in [Tedrake’s articulated-body notes](https://underactuated.csail.mit.edu/humanoids.html). Centroidal consistency is necessary but does not replace joint-level dynamics, capacity or kinematic consistency. The general dynamics and quaternion-velocity distinction are documented in [MuJoCo's computation reference](https://mujoco.readthedocs.io/en/3.3.3/computation/).

### 11.3 Inertial and contact requirements

Provide body mass, segment masses/COMs, and positive physically admissible inertia tensors in declared body frames. When explicit hands/feet are added, split their masses out of the existing forearm-plus-hand and shank-plus-foot aggregates to avoid double counting. Current COM priors in [anthropometry.md](anthropometry.md) do not supply full inertias.

During a sticking phase, enforce compatible contact motion, including `J_c v = 0` and acceleration consistency `J_c v_dot + J_c_dot v = 0` for stationary contact coordinates. Rolling or sliding has different constraints. Arrival may require an impulse model, for example `M(v+ − v−) = J_c^T Λ`, with appropriate unilateral/frictional impulse conditions and an explicit impact law. Do not infer a huge muscle torque merely by finite-differencing across an ideal impact.

Dynamic balance requires feasible contact forces/wrenches and actuation throughout the trajectory, including releases and arrivals. COM projection inside a hand/foot polygon is insufficient for general vertical, inclined or non-coplanar climbing contacts. Validate linear and angular momentum, joint equations, contact behavior and capacity together.

A proposed audit must report dynamics residuals in consistent units, actuation utilization, contact-cone margins, slip/penetration, and perturbation recovery. Time-step refinement and forward simulation are needed in addition to inverse-dynamics residuals. Kinematics alone does not uniquely identify real contact and muscle forces; an optimization solution remains conditional on its force and actuation assumptions.

## 12 Optimization and candidate selection

### 12.1 Current policy

The anatomical experiment first checks its implemented numerical gate. When every candidate fails, diagnostic ordering is:

1. Anatomical/temporal failure-frame count.
2. Worst physical failure fraction among geometric, support, unloading and palm-proxy flags.
3. Union of failed frames.
4. Palm-proxy failure count.
5. Maximum contact gap.
6. Convergence as the final tie-breaker.

The current fractions use all native frames as the denominator, including when unloading is inapplicable. Future reports should additionally show failures divided by applicable samples/events. These are declared project comparisons, not a universal ranking of human techniques.

Current solver success requires fresh normalized stationarity ≤`1e-5`, normalized complementarity ≤`1e-6`, relative cost change ≤`1e-9`, linear feasibility excess ≤`1e-7`, and a successful QP subproblem. An iteration-limit exit is not convergence; convergence is not a global optimum or evidence of biological validity.

### 12.2 Proposed complete policy

Use a constrained objective over the entire trajectory, with explicit state, contacts and actuation. Bone geometry, adopted joint admissibility, capacity, collision exclusion, contact-mode equations and dynamics are acceptance conditions. Reprojection, effort, fatigue reserve and smoothness are preferences within the accepted model. Soft penalties may help solve the problem, but final acceptance must independently check each required condition.

Require both a result and a coverage record:

- `fail`: a checked required constraint is violated.
- `unknown`: necessary parameters, evidence or coverage are missing.
- `conditional_pass`: implemented checks pass under declared assumptions, with unresolved validation limits.
- `validated_in_scope`: reserved for a specified experimental validation domain; never inferred solely from low objective value.

Only compare effort or fatigue benefits among candidates with comparable feasibility and evidence. No smoothing benefit, coaching label or lower hand load can cancel a required contact/collision/dynamics failure. If none qualifies, retain a diagnostic candidate with its explicit selection reason.

Shared contact episodes, fatigue history and end-state compatibility must connect phases. Local winners must not be concatenated without a joint transition optimization and complete re-audit. Preserve failed frames and historical baselines.

### 12.3 Current Trial2 example

The saved 180 cm run covers frames 0–975. Its selected `balanced` candidate has zero anatomical/temporal failure frames, but **534 contact-failure frames, 872 support-failure frames and 156 pre-release failure frames**. It did not converge. Its status is `diagnostic_only_not_feasible`, not a validated human motion.

This illustrates why imposing anatomical bounds can expose incompatibilities with contact targets, geometry, timing or the optimizer. A failure does not identify which cause is responsible and does not prove the filmed climber could not perform the movement. See the [saved numerical summary](../outputs/trial2/anatomy_plans/target_180cm_controls_db7b5a5de762/summary.json).

## 13 Required inputs and validation

### 13.1 Parameter contract

Every future constraint record should contain:

```text
id, implementation_status, model_version
coordinate_frame, units, applicable_modes
parameter_values, parameter_sources, uncertainty
hard_or_soft, optimization_residual, independent_audit
threshold, threshold_origin, observed_value, violation
coverage_status, failure_or_unknown_reason
```

New orientation, collision, torque, fatigue and dynamics thresholds remain unset until their model, units and validation domain are selected. An unknown value remains `null` with a reason; it is not silently converted to zero, infinity, a generic person's value or a pass. A deliberately assumed scenario may be evaluated, but its label must survive ranking, notebook display and video export.

| Input group | Required information | Present situation |
| --- | --- | --- |
| Person | Segment lengths, mass, COM/inertia profile, personal or population ROM/strength assumptions | Height-scaled lengths and approximate COM priors; full individual data missing. |
| Shoulder/elbow | Rest frames, axes, offsets, shoulder envelope and any coupling | Reduced source axes/scalar limits available; full coupling missing. |
| Hands/feet | Orientation evidence, dimensions, patch geometry, wrist/ankle/subtalar models | Hand proxy only; no complete foot orientation. |
| Scene | Camera model, wall/floor dimensions, panel geometry and contact surface normals | Hypotheses and image-based estimates, not complete measurement. |
| Contacts | Timing, mode, patch, friction, admissible wrench, confidence | Reviewed/predicted labels and assumed contact mechanics. |
| Actuation | Torque surfaces or muscle/tendon parameters and paths | No calibrated capacity model. |
| Fatigue | Initial state, recruitment definition, muscle-specific fatigue/recovery parameters | Not implemented. |
| Dynamics | Mass/inertia, timestamps, contact/impact law and integration settings | Quasi-static checks only. |

### 13.2 Required validation tests for future implementations

| Test | Expected result |
| --- | --- |
| Forward-kinematic invariants | Fixed segment lengths, proper rotations, consistent joint centers and bilateral conventions. |
| Anatomical elbow mechanism | Pure elbow flexion follows the specified parent-local axis; no independent elbow orbit. |
| Forearm and wrist isolation | Rotations affect the intended distal frames without introducing an unintended elbow displacement. For a full source model, verify against that model's actual coupled translations. |
| Pose-dependent ROM | Tested valid/invalid boundary cases; unsupported domain returns `unknown`; no interpolation through invalid regions. |
| Orientation observability | Missing palm/sole orientation cannot become a measured orientation pass. |
| Temporal invariance | Consistent results after equivalent timestamp/frame-rate changes; no quaternion-sign or Euler-wrap spikes; substep checks. |
| Collision | Known crossing and tunneling cases fail; documented adjacent-joint exceptions do not mask non-adjacent collisions. |
| Contact consistency | No force on separated/free supports; valid stick/slide behavior; no support borrowed before arrival. |
| Strength | Required torque above directional capacity fails; shared-muscle capacity cannot be spent independently at multiple joints. |
| Fatigue | State fractions remain valid; sustained loading reduces modeled availability; specified rest produces recovery; splitting a trajectory does not reset history. |
| Static dynamics limit | With zero acceleration/velocity, the dynamic model reproduces a consistent static case. |
| Free flight | COM accelerates with gravity in the absence of external support; no virtual root actuation holds it stationary. |
| Angular momentum | Contact wrench and angular-momentum change agree; no unexplained whole-body rotation. |
| Contact transitions | Release/impact tests satisfy the chosen continuous and impulsive laws. |
| Independent dynamics validation | Forward replay and time-step refinement agree within declared tolerances; solver residuals alone are insufficient. |
| Selection semantics | A smooth but infeasible candidate cannot outrank a feasible comparable candidate; unknown coverage remains explicit. |

For empirical validation, compare against suitable 3D motion, contact-force and, where relevant, activation/strength measurements. A single monocular video cannot independently validate every internal quantity listed here.

## 14 Implementation order and acceptance criteria

| Stage | Deliverable | Acceptance condition |
| --- | --- | --- |
| 1 Anatomical frames and contact consistency | Verify the existing chain against the pinned model; reconcile joint offsets, palm contacts, foot frames and uncertain contact timing/geometry. | Correct axis/rest-frame tests and explicit target incompatibilities; no hidden bone stretching or contact deletion. |
| 2 Coupled ROM and collision | Pose-dependent shoulder envelope, anatomical lower limbs, wrist/foot orientation checks, body collision volumes. | Boundary, observability and swept-collision tests pass within a declared parameter domain. |
| 3 Torque-driven whole-body dynamics | SI mass/inertia model, unactuated floating base, contact wrenches and parameterized torque capacity. | Static, flight, momentum, contact-transition and forward-replay checks pass. Unknown capacities remain conditional. |
| 4 Muscle and fatigue model | Verified muscle paths/force laws, consistent activation and fatigue states, continuous history. | Force/activation/fatigue benchmarks and person/task calibration support the intended claims. |
| 5 Shared trial integration | One versioned specification used by all trials; complete-trajectory comparison and explanation. | Same assumptions, units, gates and provenance reach optimization, audits, rankings and exported videos. |

The immediate priority is not to assign arbitrary strength or fatigue constants. It is to make the current geometry/contact/anatomical model consistent, then add dynamics with explicit capacity inputs. More physiological detail cannot correct an inconsistent wall scale, an impossible target span or incorrect contact timing by itself.

## 15 References and source boundaries

| Source | What it supports here |
| --- | --- |
| [MoBL-ARMS project](https://simtk.org/home/upexdyn) and [Saul et al., published 2015, online 2014](https://pubmed.ncbi.nlm.nih.gov/24995410/) | Upper-limb musculoskeletal-model context and cross-platform benchmarking; not validation of this reduced implementation. |
| [Pinned O2MConverter MoBL snapshot](https://github.com/aikkala/O2MConverter/blob/ad2599d5ecb2c18b44570b4d0b53e062ac9b82ae/models/opensim/MoBL_ARMS_OpenSim_tutorial_33/MoBL_ARMS_model_for_mujoco.osim) | Current source axes and coordinate ranges; not individualized physiological limits. |
| [ISB joint-coordinate recommendations, Wu et al.](https://pubmed.ncbi.nlm.nih.gov/15844264/?otool=icznmelib) | Anatomical coordinate conventions for shoulder, elbow, wrist and hand. |
| [Southgate et al., glenohumeral axial ROM](https://pubmed.ncbi.nlm.nih.gov/19375080/) | Dependence on test posture/loading in a cadaveric experiment; not a personal active climbing envelope. |
| [Akhter and Black, pose-conditioned limits](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Akhter_Pose-Conditioned_Joint_Angle_2015_CVPR_paper.pdf) | Pose-dependent validity methodology; not complete axial-twist or strength validation. |
| [CDC joint ROM reference](https://archive.cdc.gov/www_cdc_gov/ncbddd/jointrom/index_1704899060.html) | Age/sex-specific reference measurements; not individual hard bounds. |
| [Millard et al., musculotendon dynamics](https://pmc.ncbi.nlm.nih.gov/articles/PMC3705831/) | Muscle force/activation/tendon modeling and benchmark methodology. |
| [OpenSim custom fatigable muscle](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53088956/Custom%2BMuscle%2BModel%2BPart%2BOne) | An example with explicit fatigue states and activation semantics. |
| [MyoSuite fatigue validation](https://myosuite.readthedocs.io/en/latest/fatigue_validation.html) | A documented three-compartment fatigue/recovery implementation; version and parameter choice must be pinned before adoption. |
| [Tedrake, multibody dynamics](https://underactuated.csail.mit.edu/multibody.html) and [articulated-body dynamics](https://underactuated.csail.mit.edu/humanoids.html) | General dynamics, contact and momentum foundations; the climbing-specific acceptance rules here are project design choices. |
| [MuJoCo 3.3.3 computation](https://mujoco.readthedocs.io/en/3.3.3/computation/) | Dynamics, contact and collision concepts; installing a physics engine does not supply a validated human model. |

All proposed constraints, staging and acceptance rules are Beta Translator design choices unless attributed otherwise.
