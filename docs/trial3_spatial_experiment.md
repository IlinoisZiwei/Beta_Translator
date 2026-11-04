# Trial3 conditional 3D experiment

This workflow constructs a metric planar wall, per-frame perspective cameras and
an articulated 3D target skeleton, then exports native-frame trajectories,
comparison videos, an offline orbit viewer and actual glTF/GLB meshes. It is a
complete executable **hypothesis experiment**, not a measured 3D reconstruction
of the gym or a scanned human surface. It preserves the earlier 2D results.

## Run in the existing environment

Use Section 12 of `Trial3_Analysis.ipynb`, or run from the repository:

```sh
.venv/bin/python scripts/run_trial3_3d_experiment.py
```

Defaults: 180 cm, eight isolated keyframes (92, 147, 250, 296, 314, 331, 428,
486), native clips 296–331 and 474–497, tilts 0/15/30 degrees from vertical and
soft hip-depth priors 10/25/40 cm. Positive tilt means the top overhangs toward
the climber. **Actual solved hip depth is reported separately from its prior.**

The default `geometry` mode solves fixed-length spatial kinematics and performs
independent 3D force, contact, balance-reserve and pre-unload audits. Its objective
does not depend on an unobserved global gravity orientation. Therefore each
depth/clip trajectory is solved once and shared across the tilt scenarios;
world rotation and force audits are recomputed. This is an explicit gauge
experiment, not three independent recoveries of the true angle, and it does not
concatenate prior video fragments.

The optional `support` mode also optimizes candidate contact forces, COM support
reserve and pre-release unloading under assumed normals and capacity limits:

```sh
.venv/bin/python scripts/run_trial3_3d_experiment.py --mode support \
  --angles 15 --depths 25 --keyframes 314 486 --clip reach:296:331
```

For a different short interval:

```sh
.venv/bin/python scripts/run_trial3_3d_experiment.py --height-cm 180 \
  --angles 0 15 30 --depths 15 30 --keyframes 314 --clip review:300:329
```

CLI parameters select this verified Trial3 analysis video, not arbitrary new
videos. A new video needs its own tracking, calibration and contact evidence.
Notebook body-ratio overrides are forwarded to the 3D solver. Changing the body,
settings, inputs or relevant code creates a new output directory. Cache reuse
checks hashes of every exported artifact. Incomplete runs are not accepted as
completed results, and old runs are never overwritten.

## Geometry and what is identifiable

1. Read the existing 373 wall-grid correspondences, with 301 fitting and 72
   reserved points. Fifteen-centimeter square spacing and a single plane remain
   assumptions. Reserved points test consistency: correspondence discovery used
   shared grid consensus, so this is not independent physical accuracy.
2. Fit a perspective camera with square pixels, zero skew, centered principal
   point and zero distortion. Focal length and camera pose are optimized against
   fitting points. None of these intrinsics were measured.
3. Fit per-frame poses to the tracked wall correspondences using that camera.
   Body keypoints are compared in native image coordinates. Warping an off-wall
   body with a planar homography is not used as a 3D observation.
4. Fit two alternative principal-point hypotheses. They can explain the plane
   similarly while implying different off-plane geometry. Save all hypotheses.
5. Rotate wall, body and camera together for each hypothesized gravity angle.
   `tilt_identifiability.json` verifies that wall reprojection is unchanged.
   No image-only score selects a "true" wall-to-floor angle.

Wall coordinates are centimeters, X along grid columns, Y up the wall, and Z
outward. The origin is a seed hole, not the floor. GLB exports convert to meters
and gravity-aligned world Y-up coordinates. Only the white wall is modeled.
Hold markers show calibrated locations, not measured hold surfaces or thickness.

The saved 90-degree wall/floor assumption is retained in historical files. This
experiment treats it as the 0-degree-from-vertical scenario, **not a measurement**.

## Skeleton, temporal solve and numerical checks

The target uses the project's fixed segment proportions and a rigid torso with
shoulder/hip width. Torso rotation has three dimensions. A contacted limb uses
the analytic intersection of two bone-length spheres, with an optimized elbow
or knee swivel. Unreachable targets remain missed; endpoints are not stretched
or silently moved onto holds. Free limbs retain spatial direction parameters.
The 0.02 cm reach reserve avoids a singular fully extended elbow circle and is
smaller than the unchanged 0.1 cm contact audit threshold.

The swivel reference axes are fixed for each optimization. Earlier prototypes
exposed a discontinuity when an axis was selected again during fitting; the
final implementation fixes this and tests its derivative across that boundary.
Temporal penalties act on actual XYZ joints, not on mixed swivel/free-limb
coordinates. Every short clip is optimized jointly at its native frame rate.

The solver combines bounded sparse-direct damped Gauss–Newton with an L-BFGS-B
fallback. Both report a freshly evaluated, column-normalized projected gradient.
Nonconvergence is retained even when the animation looks smooth or contacts fit.
First-order stationarity is not a global-optimum or physical-feasibility proof.

Exact bone lengths, contact gaps, joint-center wall penetration, bend limits,
camera visibility and native frame coverage are checked independently. Cylinders
are a display proxy; skin thickness, full anatomical joint limits, self-collision,
grip shape, joint torque limits and dynamics are not modeled.

## Failure and force interpretation

`pair_span_certificates.csv` provides a conservative geometric bound: two fixed
contacts cannot be farther apart than the connecting limb lengths and rigid
torso-root separation. Rigid wall tilt cannot change that distance. Adding body
depth cannot defeat the bound. A positive excess greater than 0.2 cm proves that
both endpoints cannot simultaneously satisfy the 0.1 cm tolerance **under the
specified body, hold IDs and grid scale**. It does not prove that a human cannot
climb the route, or identify which input assumption is wrong.

The primary force audit assumes wall-outward foot normals, a friction
coefficient of 0.8, and a per-component capacity of one body weight. Hands may
pull or push. A second audit uses world-up foot normals to expose sensitivity to
unmeasured foothold surfaces. Neither force model is a measured grip model.
The same idealized force limits must not be treated as user strength measurements.

Force feasibility uses an independent six-equation wrench LP. Missed contacts
are excluded. Eligible contacts additionally require checked calibration coverage
and sufficient classifier confidence. Conditional audits retain weaker evidence
and are labeled accordingly. COM perturbations of ±2 cm in wall X and Z probe
reserve; they are not a full robustness certificate. Pre-unload audits remove
the upcoming departing support, but automatic timing remains uncertain.

Optional support optimization uses a sampled inner COM-support polygon, with
0.65 optimization friction and 0.9 body-weight force limits to leave reserve
against the unchanged 0.8/1.0 independent audit. It does not add contacts, change
hold IDs or turn an extrapolated/low-confidence support into confirmed evidence.

## Optional open-weight depth comparison

The main experiment needs no new packages or model downloads. A separate probe
uses the official Apache-2.0 **Depth Anything V2 Small** model. Its source is pinned
to `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`; its checkpoint SHA-256 is checked.
The approximately 99 MB checkpoint and source stay outside the repository.

```sh
.venv/bin/python scripts/probe_trial3_depth.py --download --input-size 518 \
  --output outputs/trial3/spatial_experiments/my_depth_probe
```

The default assets directory is `~/.cache/beta-translator/depth-anything-v2-small`.
Use `--assets-dir` to choose another cache. This uses the same `.venv`, explicitly
runs inference on CPU, and refuses to overwrite a populated output directory.
Run with `--input-size 280` and a different output directory for the smaller probe.

The probe fits relative inverse-depth scale/shift to the hypothetical wall camera,
excluding the observed person's box. It exports depth images and a genuine XYZ
surface point cloud (`conditional_neural_surface.ply`). Its centimeter scale is
conditional on the camera/grid assumptions. Predicted visible surfaces are not
joint centers. These depths are **not injected into IK as ground truth**.

## Open-source options investigated

| Method | Role | This experiment |
| --- | --- | --- |
| [OpenCV pose fitting](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html) + SciPy | Calibrated-plane cameras, articulated IK and force LP | Implemented and executed using existing packages |
| [Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2) | Lightweight relative-depth surface prior | Executed at two input resolutions on four native frames; not metric ground truth |
| [Depth Anything 3 Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main/da3_streaming) | Chunked reconstruction of long image/video sequences | Documentation reviewed; not locally benchmarked or installed |
| [VGGT](https://github.com/facebookresearch/vggt) | Joint cameras, depth and point maps | Documentation reviewed; original and commercial checkpoints have different terms; not locally executed |
| [COLMAP](https://colmap.github.io/tutorial.html) | Multi-view camera/scene reconstruction | Candidate when future footage contains useful viewpoint translation and static scene correspondences; not executed here |
| [GVHMR](https://github.com/zju3dv/GVHMR) | Learned articulated human mesh/motion prior | Not executed; official setup requires separately obtained SMPL/SMPL-X body assets ([instructions](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md)) |

The inspected DA3 Streaming workflow uses overlapping chunks and reports GPU
experiments. Its published memory/runtime results must not be applied to this Mac
or to climbing footage without a separate benchmark. Model outputs cannot replace
contact/geometry validation just because they look like a complete 3D scene.

For larger input resolution, keep analysis coordinates and camera intrinsics
consistent when resizing; render separately at the desired resolution. For longer
videos, prefer bounded windows with overlap and explicit seam checks, then a
global continuity refinement. The current executable jointly solves each selected
clip; it does not yet implement an arbitrary-length streaming optimizer. Benchmark
short/medium windows before claiming large-video performance. A practical next
measurement is wall tilt plus a second viewpoint or calibrated camera information.

## Published results

The [results report](trial3_spatial_results.md) contains measured runtimes, the
iteration comparison, remaining failures, and links to the model/viewer/videos.
It also distinguishes fixed-center impossibility bounds from whole-region
bounds: the former must not be treated as proof that the route is impossible.
All existing videos, notebooks and historical outputs remain preserved.
