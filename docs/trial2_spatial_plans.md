# Trial2: complete 3D plans across a wall-angle transition

Open **Trial2_Analysis.ipynb**, select **Python (Beta Translator)**, set the target
height in Section 1, then use Sections 11–15. The saved experiment uses **180 cm**,
all **976 native frames (0–975)**, and 30 FPS. The previously reported **173 cm**
source climber is a separate input; the target skeleton does not inherit that height.

```sh
.venv/bin/python scripts/run_trial2_spatial.py --height-cm 180
```

The existing environment and repository assets are sufficient. An unchanged run
reuses a cache only after checking its input/code provenance and every exported
artifact hash. Changed settings create a separate result directory. Old Trial1,
Trial2 and Trial3 results are retained. A fresh full comparison is substantially
slower than opening the saved videos or verified cache.

## Joined spatial panels, not a rotating flat picture

The experiment has an upper wall, an overhanging middle wall and a lower vertical
panel. Each face has a distinct 3D plane and outward normal. Adjacent planes share
the reconstructed hinge exactly. The wall is static throughout a candidate; the
body moves from middle-panel contacts through mixed support at the hinge to the
upper panel. Gravity is always world-down. Surface-normal and tangential foot
force constraints change with the contacted panel.

The principal angle hypotheses, in degrees from vertical, are:

| Middle panel | Upper panel | Purpose |
| ---: | ---: | --- |
| 15 | 0 | Mild overhang followed by a vertical upper face |
| 30 | 0 | Larger spatial change at the same upper face |
| 45 | 0 | Stronger overhang stress test |
| 30 | 15 | Both faces overhang, with a smaller difference at the hinge |

These are **hypotheses**, not inferred true wall angles. The middle face is rotated
about the actual reconstructed hinge; its realized gravity-relative tilt is
reported separately when the fitted edge is not exactly horizontal. Selecting a
low-failure scenario does not identify the real angle.

`data/trial2_spatial_hypotheses.json` contains assistant-selected image boundaries
on reference frame 450. It adds an explicitly hypothetical rectangular grid to
the earlier **unmeasured 15 cm hole-spacing assumption**. A perspective camera is
fitted to the upper grid with square pixels, a centered principal point and zero
distortion. One local held-out hole measures consistency only; it does not
validate the whole wall or the camera intrinsics. Source joints use the existing
global stabilization homographies. Those transforms are approximate for a moving
camera and multiple planes, especially away from their dominant fitted surface.

The lower panel is vertical through its hinge. A horizontal floor uses the mean
reconstructed height of the selected wall/floor boundary; disagreement between
the boundary endpoints is saved. Floor depth, height and normal remain conditional.
The reported source height does not make this calibration unique and is not used
to force a crouched silhouette to have a particular scale.

## Contacts, ground and missing observations

Original model hold IDs, confidence scores and native timestamps are preserved.
The contact network is not retrained. Its outputs remain proposals, not confirmed
technique labels. The source pose table has rejected/missing observations; the
loader restores the native time axis and gives those observations zero weight.
Interpolation initializes the optimizer only. Every output video keeps all frames.

Each continuous limb/hold episode gets one source-evidence contact patch inside
the observed mask and on one panel. Concave holes and gaps are not bridged. A
single bounded contact coordinate is shared across the episode: a hand cannot
slide around the hold independently on every frame. Missing landmark evidence
freezes the proposed point instead of granting extra search space.

The actual surface contact is distinct from its wrist/ankle target. Hand offsets
use 0.035h along the panel and 0.012h outward; foot offsets use 0.045h along the panel
and 0.025h outward. These are the illustrative Trial3 defaults, not measurements.
The source provides a planar direction, not a target-body segment length. A
proposed ground contact instead uses an illustrative ankle height of 0.035h.
Independent support forces act at the **surface point**.

Ground is proposed only for visible feet within 18 reference pixels of the
selected floor edge, with at most 4 pixels of motion per frame for at least 10
consecutive frames, and without a known hold. These numerical rules are recorded
in `ground_evidence.csv`; raw labels remain separate. A foot missing a hold does
**not** automatically become a ground support. Unlocated smears and unresolved
free-foot support receive zero force. Ground proposals are excluded from the
more conservative eligible-evidence support count because they are unreviewed.

## Whole trajectories and technique candidates

Each angle scenario is solved across the entire video. The numerical model uses
fixed bone lengths, perspective reprojection, contact regions, explicit offsets,
wall/floor penetration penalties and temporal joint-position regularization.
All 976 frames contribute even when a shared pose-control basis uses every fourth
frame plus contact-transition neighbors. This is one global optimization, not
independent clips joined together.

Contacted limbs use exact two-bone geometry with an elbow/knee swivel and a
155-degree bend limit. Targets outside the reachable range remain misses against
the unchanged original target. Near instantaneous known-hold-to-different-hold
switches, free angular coordinates permit a visible miss instead of forced
teleportation. Raw contact schedules are not silently rewritten.

After comparing angles, the lowest-failure tested geometry is also used for:

| Candidate | Change |
| --- | --- |
| `center_control` | Fixed hold centers without joint/contact offsets |
| `support` | Joint posture/force optimization with gravity, friction and pre-release unloading terms |
| `leg_drive` | Support terms plus conditional COM-to-feet and phased hip-height preferences |
| `flag_assist` | Leg-drive preferences plus a lateral path preference for already-free feet |
| `arm_economy` | Leg-drive preferences plus a hand-force penalty |

These names describe requested preferences, not proof of correct human technique.
No new foothold order, foot swap, hook or dynamic jump is generated. A non-contact
flag has zero contact force. A larger geometric gap cannot be excused by a lower
hand-load score. The trials use the same frame schedule, body, audit tolerances,
friction and capacity assumptions. Technique candidates start from the same full
geometric candidate; they have additional optimization compute, explicitly distinct
from the first angle sweep.

A tenth candidate refines the selected complete plan with independent pose variables at all 976 native frames, jointly coupled in time. It keeps the same contacts, body and audit limits. Its additional freedom and compute are labeled explicitly; it is not an equal-budget technique ablation. Both the original comparison and the refinement remain available.

## Independent checks and selection

The independent support LP enforces three force and three moment equations.
Hand forces are component-bounded and may pull; feet can push only along their
own panel/floor normal and use a conservative friction pyramid with coefficient
0.8. Each force component is limited to one body weight. A missed contact beyond
0.1 cm is excluded from support even if the optimizer assigned a force to it.
Unknown/free contacts have exactly zero force.

The audit also checks fixed bone lengths, bends, wall/floor penetration,
8 cm/frame motion and 4 cm/frame² acceleration warnings, remaining supports after
removing departing limbs, and ±2 cm COM perturbations. These are numerical model
thresholds, not clinical or human capability limits. Quasi-static support cannot
validate swinging, jumping, inertia or impact.

Candidates rank by geometric failures, continuity warnings, nonconvergence,
support failures, pre-release failures, missing reserve and finally hand-load
proxies. Hand load is evaluated using the same minimum-L1-force LP; comparison
uses the same feasible frames shared by the candidate set. A lower bound is not
measured grip effort, energy savings or fatigue. When there are no common feasible
frames, the load criterion is explicitly uninformative.

Strict convergence requires a freshly checked normalized projected gradient no
greater than 1e-5 and the solver's stabilization criterion. Stalled or
iteration-limited solves remain marked as nonconverged. Failure labels and
geometry uncertainty are visible even in the selected animation.

## Deliverables

- `selected_plan_3d.mp4`: one selected continuous plan inside a single 3D coordinate
  box, showing joined translucent wall faces, contact points, joint/contact
  connectors, COM, source inset, phases and unresolved constraints.
- `angle_comparison_3d.mp4`: four complete 3D animations with shared axis limits.
- `viewer.html`: offline interactive 3D orbit, zoom, native-frame slider and all
  candidate choices; no hosted service or external JavaScript library is needed.
- `candidate_comparison.csv`, `angle_comparison.csv`, `phase_explanations.csv`:
  full-plan metrics and first-differing ranking reasons.
- `contact_span_review.csv` bounds reach conflicts across each entire conservative contact patch; `timing_conflicts.csv` identifies instantaneous known-contact changes incompatible with the declared motion limit. These are conditional model certificates, not human impossibility claims.
- Each candidate retains its 3D trajectory, raw/derived contacts, geometry,
  episode evidence, independent audits and solver history.

## Limits that remain visible

Geometry, surface normals, offsets and ground proposals need physical review.
Collision checks are limited to wall/floor penetration; inter-limb collision,
fingers, articulated feet, hold thickness, grip direction, fatigue and dynamic
forces are not resolved. Matching image projections or selecting a candidate
does not validate a real climb or identify a true wall angle.

## Saved 180 cm experiment

[Selected full 3D video](../outputs/trial2/spatial_plans/target_180cm_5f9e7fcd448b_native/selected_plan_3d.mp4) ·
[Four-angle video](../outputs/trial2/spatial_plans/target_180cm_5f9e7fcd448b_native/presentation_60434267bc36/angle_comparison_3d.mp4) ·
[Offline 3D viewer](../outputs/trial2/spatial_plans/target_180cm_5f9e7fcd448b_native/presentation_60434267bc36/viewer.html) ·
[Full comparison and reasons](../outputs/trial2/spatial_plans/target_180cm_5f9e7fcd448b_native/candidate_comparison.csv)

The selected complete candidate is **`native_refinement`**, using the
**15° middle / 0° upper** hypothesis.
The ranking compares the entire 976-frame plans, with geometry failures before
support/load proxies. It does not establish the actual wall angle or a physically
successful climb. The table reports all candidates in ranked order.

| Candidate | Contact failures | Geometry failures | Motion warnings | Conditional balance passes | Unload failures / checks | Converged |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| native_refinement | 91 | 101 | 17 | 228 / 976 | 156 / 156 | False |
| region_m15_u0 | 107 | 125 | 38 | 224 / 976 | 156 / 156 | False |
| support | 111 | 126 | 37 | 212 / 976 | 156 / 156 | False |
| flag_assist | 112 | 128 | 37 | 212 / 976 | 156 / 156 | False |
| leg_drive | 114 | 130 | 38 | 212 / 976 | 156 / 156 | False |
| arm_economy | 114 | 130 | 38 | 212 / 976 | 156 / 156 | False |
| region_m30_u15 | 113 | 130 | 45 | 119 / 976 | 156 / 156 | False |
| center_control | 150 | 163 | 53 | 188 / 976 | 156 / 156 | False |
| region_m30_u0 | 170 | 198 | 91 | 184 / 976 | 156 / 156 | False |
| region_m45_u0 | 270 | 291 | 159 | 144 / 976 | 156 / 156 | False |

The final audit retains **53 frames** with
conditional reach contradictions even across the full allowed contact patches,
and **2 instantaneous known-contact
switches**. These diagnoses concern the assumed camera/scale/panels/offsets and
unreviewed contact timing; they do not prove that a real person cannot climb the route.

**14 native frames** lack usable source pose
observations, including **10 identity rejections**.
All are retained with missing-source labels. There are
**78 proposed ground limb-frames**, each requiring
positive image evidence. They remain unreviewed.

Hand-load comparisons use **49 common feasible
frames** across all ten cases. This limited overlap must not be presented as an
energy-saving result over the whole video. Own-feasible integrals are separately
retained for diagnostics and are not compared as if they used the same frames.

Validation: **200 tests passed**; independent reconstruction checked every saved
candidate against its parameters and original inputs, including exact bone lengths,
contact episode bounds, fresh convergence gradients, force/moment balance,
foot friction/capacity, missing observations and unchanged raw contact labels.
Both exported videos decode to exactly **976 frames, 30 FPS, 1600 × 1000**.
Every input and exported artifact hash was checked. New notebook sections are
also executed in a fresh kernel before installation.
