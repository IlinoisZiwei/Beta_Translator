# Anny appearance adapter

The same shared anatomical trajectory drives a stick figure and an Anny human
surface. This adapter changes appearance, not selected motions, contact timing,
forces, joint limits or the meaning of a physical failure. It uses the existing
Python environment on CPU; no retraining, Blender, CUDA or second kernel is needed.

## Run and view

Run `bash setup.sh` to update the existing `.venv`, then continue using the
**Python (Beta Translator)** notebook kernel. The final section of each
`Trial*_Analysis.ipynb` displays the saved comparisons and source-video overlays.
Video files are referenced, not embedded into the notebooks.

```bash
# Adapt the three existing personalized verification runs.
.venv/bin/python scripts/run_avatar.py --trial all

# Adapt a specific completed shared-body run (short or full sequence).
.venv/bin/python scripts/run_avatar.py --trial trial1 --source-run outputs/trial1/shared_body/YOUR_RUN
```

Python entry point:

```python
from beta_translator.body_model import BodyModel
from beta_translator.avatar_pipeline import run_avatar, show_avatar

# Read a saved trajectory when it already exists. Fit only for a new profile,
# different interval, or different scene. All temporal constraints remain active.
motion = BodyModel(user_profile=ROOT / "configs/users/trial_user_175.json").fit(
    "trial1", project=ROOT, frames=[103, 180], iterations=35,
    candidates=("steady",), render=False
)
avatar = run_avatar(ROOT, motion["output_dir"])
show_avatar(ROOT, avatar["output_dir"])
# frames="all" uses the same motion entry point for a complete native sequence.
```

New motion fitting can take tens of minutes or longer on CPU; the 35-iteration
example is a diagnostic budget, not a convergence guarantee. Viewing the saved
notebook outputs does not run this fitting step.

The adapter reads the profile embedded in the saved motion, not a subsequently
edited user JSON. Generate a new motion for a new body. Outputs are content-keyed
under `outputs/<trial>/avatars/`; old outputs are never replaced. Anny's derived
asset cache is under `.cache/anny/` (ignored by Git); `ANNY_CACHE_DIR` can override it.

For shared-body intervals longer than 120 frames, the solver now reuses the
existing clamped cubic joint-coordinate controls at an eight-frame stride,
generalized from 28 to 32 anatomical coordinates. Supported contact forces stay
independent at every native frame; free/unknown-contact forces remain exactly
zero. Every residual and final audit still evaluates every native frame. This
restricts the trajectory search space to reduce matrix factorization cost; it
does not establish convergence or interpolate rendered Cartesian poses. The
solver manifest records the parameterization and actual iteration budget.

## Calibration and rotation transfer

- Native Anny 0.6.2 topology and weights are used: 13,718 vertices, 27,420 triangles.
  No optional SMPL-family assets are requested.
- Convert metres and Anny's X-left/Y-back/Z-up frame into project centimetres and
  X-right/Y-up/Z-out using a proper rotation, not a reflection.
- Calibrate segment lengths and shoulder/hip anchors to the user's shared body.
  Head/neck extent completes stature and remains inferred. Rest surface height
  and extended-arm surface span are measured after calibration.
- Bind Anny's helper bones to 14 existing segment frames. Source shoulder, elbow,
  forearm, wrist, hip, knee, ankle/calcaneus orientations drive the surface.
  Ankle and subtalar motion enter through the saved shank and foot frames.
  The shared model's knee translation remains in the saved joint origins.
- Each posed segment uses `R(t) @ R(bind).T` and its saved origin. There is no
  endpoint-only swivel fit, new Euler-angle solve, per-frame smoothing or hold
  snapping. The torso is as rigid as the current source model; extra Anny spine
  and finger degrees of freedom are not independently optimized.
- Fingers retain their template pose and move with the wrist. This is visible
  hand anatomy, **not observed finger tracking or a verified climbing grasp**.

## Contact surfaces and audits

Palm anchors are barycentric points on actual mesh triangles, calibrated once
in the bind pose to the profile's palm offset. A local palm-pad correction is
recorded in centimetres. It does not move a surface to a hold during playback.
The foot surface uses an explicitly inferred flat sole. Persistent ground/hold
sole anchors correspond to the same 0.45/0.65 foot-length locations as the solver.

`contact_audit.npz` preserves mesh anchors, solver anchors, hold targets, known
contacts and ground modes for every native frame. `surface_metrics.csv` separates:

1. **Mapping drift:** mesh anchor versus solver anchor (display limit 0.1 cm).
2. **Hold gap:** mesh anchor versus hold target (existing limit 0.1 cm).
3. **Source failures:** all existing failed constraints, without changing status.
4. **Mesh-wall penetration:** signed surface clearance under the assumed scene;
   more than 0.5 cm is flagged. This is a diagnostic, not collision avoidance.

A display mapping pass does not imply that the motion is feasible. Mesh-based
self-collision, muscle shape, inertia calibration, finger/hold collision and
grasp forces are not added by this adapter. Existing mechanics still use their
documented geometric proxies. Arbitrary extreme profiles may distort the mesh.

## Scene and video

`avatar_comparison_3d.mp4` shows both representations with the same fixed 3D
camera, wall geometry, contacts, frame count and native FPS. `avatar_source_overlay.mp4`
projects the surface through the existing trial camera and the reference-to-frame
homography. The original climber stays visible; person segmentation, removal,
background completion and correct wall/person occlusion are not implemented.
Trial1 remains an image-plane/depth proxy; Trial2 and Trial3 retain their
unmeasured wall/camera assumptions. CPU triangle painting is for presentation,
not a replacement for physical visibility/collision reasoning.

The small `avatar_bind.npz` and `avatar_motion.npz` files reconstruct the complete
surface animation without storing a large mesh for every frame. `bone_mapping.csv`,
profile measurements, surface calibration and SHA-256 manifests document the
conversion. Neither old meshes nor old solver results are overwritten.

## Sources and licenses

- [NAVER Anny](https://github.com/naver/anny), version 0.6.2: Apache-2.0 code;
  included MakeHuman/MPFB assets use CC0. Preserve the package's license notices.
- [Anny shape parameterization](https://naver.github.io/anny/build/shape_parameterization.html)
  and [pose transfer](https://naver.github.io/anny/build/pose_transfer.html).
- [MPFB asset/source license split](https://github.com/makehumancommunity/mpfb2/blob/master/LICENSE.md).

Use the native `anny` rig/topology. Optional SMPL/SMPL-X assets have different
terms and are not needed or downloaded by this integration.

## Saved full-sequence displays

All three analysis notebooks now default to `AVATAR_CASE = "full"` in their
Anny section. Trial1 covers 0–416, Trial2 0–975 and Trial3 0–502, all at 30 FPS
with the 175 cm / 176 cm profile. Each shows the 3D comparison and source overlay.
Use `"verification"` for the preserved short checks (`"demonstration"` is also
available for Trial1). The full diagnostics retain source failures and should not
be interpreted as feasible climbing recommendations. See the
[recorded checks](validation/anny_avatar_20261105.md).
