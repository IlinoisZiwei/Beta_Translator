# Anny adapter validation — 2026-11-05

The saved user profile specifies height 175 cm and arm span 176 cm. Native Anny
surface calibration measures 175.00 cm height and 176.01 cm extended-arm span,
within the 0.1 cm display tolerance. Three existing
personalized verification trajectories and a new Trial1 motion interval were
adapted. All adapters passed the 0.1 cm mesh-to-solver patch mapping limit.
**This validates appearance transfer, not physical feasibility.**

| Run | Frames | Count | Maximum mapping drift (cm) | Frames with source failure | Frames with mesh/hold gap > 0.1 cm | Maximum assumed-wall penetration (cm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| trial1 verification | 113–136 | 24 | 1.14e-13 | 24 | 21 | 8.65 |
| trial1 demonstration | 103–180 | 78 | 1.27e-13 | 78 | 45 | 11.73 |
| trial2 verification | 550–595 | 46 | 1.17e-13 | 46 | 43 | 21.58 |
| trial3 verification | 296–331 | 36 | 1.14e-13 | 36 | 28 | 13.19 |
| trial1 full diagnostic | 0–416 | 417 | 0.0412 | 417 | 400 | 22.37 |

Trial1 frames 103–180 contain unchanged-contact phases, left-hand release/arrival
at 113/136, and right-foot release/arrival at 153/173. The new source solve used
35 constrained iterations with the shared personalized model. Its convergence
criterion was not met. None of the four source motions has a physical-pass claim.
Existing failures and uncertain geometry remain visible in every rendered video.

The adapter transfers complete segment rotations, including twist. A dedicated
regression rotates a wrist with unchanged endpoints and checks that the mesh
follows the saved rotation. It also verifies that changing hold targets does
not silently move the mesh. Ground/hold sole anchors are tested separately.

Validation: 41 targeted tests passed, covering avatar calibration/transfer,
source integrity, constraints, generalized temporal controls, anatomical legs,
shared-model behavior and personalized mechanics. The new notebook cells in all
three trials executed against saved results with two externally referenced
videos each. All previous notebook cells were retained. The main
`Beta_Translator.ipynb` was not modified. The single Python environment passed
`pip check` and `scripts/check_environment.py`, including compiled Detectron2/NMS.

Known limits: fixed template fingers, inferred flat sole and body thickness,
uncalibrated wall/camera, remaining source contact/support failures, mesh-wall
penetration, and no original-person removal or wall/person occlusion. Rendering
adds no mesh collision avoidance, new grasp mechanics or trained motion model.

## Full Trial1 diagnostic

The same personalized model fitted all 417 consecutive native frames (0–416),
with 15 constrained iterations and 8-frame cubic joint controls. Every contact,
force and final audit remains at native 30 FPS. Solver converged: **False**;
linear-bound violation: 0. Coordinate-rate failures: 0; angular-acceleration failures: 0.
Mesh-to-solver contact mapping passed: **True**; maximum drift
0.04119 cm. Source-failure frames: **417/417**.
This is a complete diagnostic trajectory, not a feasible-route recommendation.

Select `AVATAR_CASE = "full"` in Trial1 Section 22 to view the full comparison
and overlay; `"demonstration"` retains the short stretch. Both runs, their source
trajectories and all previous results are preserved separately.

## Full Trial2 and Trial3

The same 175 cm / 176 cm profile now has complete native-frame Anny outputs for
both remaining trials. Each source run uses one continuous interval, 15 constrained
iterations, the existing joint-curve controls and independent native-frame forces.
All original contact, anatomical and personalized checks remain active.

| Trial | Frames | Seconds | Maximum mapping drift (cm) | Source-failure frames | Mesh/hold gap failures | Maximum assumed-wall penetration (cm) | Source converged |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| trial2 | 0–975 | 32.53 | 0.04119 | 976/976 | 914 | 22.24 | False |
| trial3 | 0–502 | 16.77 | 0.00000 | 503/503 | 464 | 14.49 | False |

These are appearance-transfer checks and diagnostic trajectories, not verified
feasible climbs. Wall/camera and contact orientation remain inferred; fingers
retain their template pose and the original climber is retained in overlays.

For longer intervals, native-frame mechanical derivatives now apply the same
finite-difference stencil directly instead of materializing and squaring a dense
time matrix. Checks at 46, 503 and 976 frames differed from the previous mechanics
by less than 5e-11. Sparse fatigue assembly preserves exactly the same nonzero
causal Jacobian; mixed active/clipped recruitment is checked by finite differences.
Twelve personalized-mechanics tests and fourteen avatar/solver/control tests passed.
Local quadratic subproblems above 9,000 variables use a 1,000-iteration budget
(Trial2); smaller problems retain 3,000 (Trial3). These remain inexact SQP steps: a
hard-bound projection and nonlinear objective decrease are required for acceptance.
The same convergence and physical-failure gates remain in effect. Source bounds
and failure gates were not relaxed. Solver logging reports every
iteration so longer runs show progress.

Both notebook cells were executed against the saved results; videos remain external
references. Video frame counts, FPS, beginning/middle/end decoding, source manifests
and every saved output hash were checked. Earlier notebook workflow cells, short
Anny results, source videos and historical results were preserved.
