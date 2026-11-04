# Trial1: conditional ground support

Open [Trial1_Analysis.ipynb](../Trial1_Analysis.ipynb) with **Python (Beta Translator)**.
Configure target height, assumed mass, body proportions and foot friction in Section 1,
then run through Section 7 in order. The default is **180 cm, 70 kg, friction 0.8**.
Alternatively, run from the repository:

```sh
.venv/bin/python scripts/run_trial1_ground_optimization.py --height-cm 180 --mass-kg 70
```

All **417 native frames, 0–416**, enter each global optimization round together.
A fresh gravity IK fit uses the new ground evidence from initialization onward. No
frames are assembled from previous single-move clips. New runs go to
`outputs/trial1/ground_support/target_<height>cm_<hash>/`; old results remain intact.

## Evidence and geometry

`data/trial1_ground_support.json` binds the visual interval assessment to the
source video hash and native frame numbers. The assessment is by the assistant,
**not user-confirmed**. It deliberately leaves the ambiguous rolling, occluded
and lift-off boundaries unreviewed. A visible heel/toe landmark is also required
to locate a point; a visually assessed interval alone cannot supply a trustworthy
location when the tracker is uncertain.

The first implementation uses manually estimated floor image polygons. It does
**not** automatically segment the wall or infer ground contact from `free`.
Inside-polygon + visible + three-frame stability produces `ground_candidate`,
which receives **no ground force**. `off_ground` means off the floor; it does not
mean the foot is necessarily airborne rather than on a hold. After frame 74,
ground observations are unreviewed and no new ground support is inferred.

Only a visually observed, sufficiently visible, located ground contact may be
added to the **conditional** support set. A conflicting decoded hold is preserved
and flagged. Raw model labels and model probabilities are never overwritten.
`support_ids` and `contact_type` describe the separate effective contact model;
the ground fitting weight is an explicit objective weight, not a probability.

The source sole location is the lower of the heel/toe image landmarks, an
approximate shoe-floor point. The target sole is a world-down offset from the
ankle of `0.025 * target_height_cm` (4.5 cm at 180 cm). This is an assumption,
not a measured foot model; there is no ankle rotation, sole mesh or foot mass.
The fixed limb lengths remain unchanged. Contact errors, force moments, coupling
penalties, independent audits and rendered arrows all use the same sole point.

Wall alignment only co-registers source image points for the existing renderer.
It **does not calibrate the floor plane**. The old 1.798 px/cm uniform scale,
image-up gravity and image-up floor normal are still approximations. Ground
points can drift or roll; no-slip, global floor collision and floor depth are not enforced. Therefore
ground contacts never become geometrically eligible/confirmed in this run.

## Forces and unloading

In the assumed model coordinates, the floor normal is `[0, 1]`: `Fn = Fy >= 0`
and `|Fx| <= mu * Fy`. The same configurable foot friction is used for ground
and foot-on-hold contacts. These are assumed force models, not estimated surface
friction or wall normals. Friction is a soft optimization objective and an
independent reported check; an infeasible candidate is not silently certified.

A ground support ID ending while the raw label remains `free` now creates a
departure event. Up to six preceding frames include unloading and remaining-
support objectives. With uncertain visual timing, the event is explicitly
`ground_observation_ends_boundary_uncertain`: the model stops granting support
at the last reliable observation, without claiming that is the true liftoff.
After the active ground interval, that limb's **ground force is exactly zero**.
A later located hold may provide a separate hold force. No future support is
borrowed for a pre-release check.

## Inspecting the outputs

- `review.html`: synchronized original video, new result and preserved previous
  **180 cm** no-ground video. The diagnostics table always describes the new run.
- `ground_evidence.jpg` and `ground_evidence.csv`: native-image points, interval
  evidence, confidence gating, support activation and ground force.
- `limb_diagnostics.csv`: raw labels/probabilities alongside effective support
  IDs/types, fitting weights, sole coordinates, gaps and force checks.
- `trajectory.npz`: all poses, unchanged raw labels, effective contact geometry
  and forces, with no pickled objects.
- `comparison.csv`: fresh gravity IK with ground evidence versus support-refined
  poses. Both trajectories use the same new contact schedule and audit thresholds.
- `published_comparison.csv`: previous published no-ground counts versus the
  new ground-model counts. Because support evidence changes, this is not an
  isolated measure of optimizer quality.
- `release_events.csv`, `preunload_checks.csv`, `issue_intervals.csv` and
  `summary.json`: uncertain boundaries, failures and solver termination/history.

Changing height refits all frames with the requested fixed body. The previous
video remains a labeled 180 cm reference; it is not a same-height baseline when
the new height differs. Annotation or code changes invalidate the cache.

Walking and liftoff are dynamic; this model is still quasi-static. Failed
contacts, uncertain support and failure to converge remain visible. Improvement
in a conditional 2D check is not proof that a person can execute the movement.

## Saved 180 cm run (2026-11-03)

The saved video has 417 decoded frames at 30 FPS (13.9 seconds). There are 48
active ground limb-frames across 45 video frames; **zero** ground contacts are
classified as geometrically eligible. The original 1,668 decoded limb labels,
probabilities and hold targets are unchanged.

Compared with the previous published no-ground model, balance failures change
from 49 to 18, while uncertain-support frames change from 65 to 96. Thus **31
frames move from failure to conditional uncertainty, not to validated success**;
model-check passes remain 303. Residual balance failures are frames **18–35**,
where neither foot has reliable located ground evidence. These flags describe
the available support model, not proof that the observed person could not stand.

Contact errors above the unchanged 0.1 cm threshold change from 1 to 5 frames
(54, 57, 58, 61, 241); the maximum is 0.1755 cm. Pre-unload failures change from
0 to 6 frames (12–17), associated with the uncertain right-foot support cutoff
at frame 18. New ground departure events are 3, 18 and 63, all timing-uncertain.
No final pre-departure candidate load exceeds 1% body weight. Force, moment and
friction residual checks pass their tolerances throughout; these tolerances are
separate from the strict fixed-contact reserve audit.

For a fixed NEW ground schedule, fresh gravity IK -> support refinement reduces
contact-error frames 105 -> 5, balance failures 65 -> 18 and pre-unload failures
26 -> 6. The gravity initialization converges at 580 evaluations. Each of the
three subsequent IRLS fits reaches its 800-evaluation limit, so the final result
is explicitly **unconverged** and combined numerical-pass frames remain zero.
No physical feasibility claim is made.

Validation: 67 relevant tests; independent recomputation of contact gaps,
segment lengths, force/moment/friction residuals and 702 feasible audit force
solutions; zero force on unknown/departed supports; complete video decoding.
The review page's controls were tested with a mock DOM, not a browser rendering.
