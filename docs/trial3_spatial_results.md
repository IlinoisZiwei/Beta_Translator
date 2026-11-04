# Trial3: failure review and conditional 3D results

The implemented experiment constructs a planar wall, perspective cameras and a
fixed-length **180 cm articulated 3D skeleton**. It exports meshes, an offline
orbit viewer, native-frame trajectories, videos and independently audited metrics.
It does **not** recover measured wall tilt, camera intrinsics, hold surfaces or
the climber's true 3D body. All earlier 2D results are retained.

Start with [Trial3 notebook Section 12](../Trial3_Analysis.ipynb), the
[converged short-clip angle video](../outputs/trial3/spatial_experiments/refined_angles_180cm/refined_angles.mp4),
or its [offline viewer](../outputs/trial3/spatial_experiments/refined_angles_180cm/viewer.html).
The [method guide](trial3_spatial_experiment.md) describes assumptions and setup.

## What was run

| Study | Scope | Result and interpretation |
| --- | --- | --- |
| Earlier failure audit | All 503 frames, original support-optimized 180 cm result | 98 contact-failure frames; 83 also violate a fixed hold-center pair-span bound. The remaining 15 cannot be assigned a unique cause from this audit. |
| Main 3D matrix | Eight keyframes: 92, 147, 250, 296, 314, 331, 428, 486; clips 296–331 and 474–497; 0/15/30° tilt × 10/25/40 cm hip-depth priors | Nine scenarios, 90 reported cases. Geometry is shared across angles and force audits are recomputed. This is 24 distinct static fits and six distinct clip fits, not 90 independent recoveries. |
| Continued trajectory | Joint refinement of all 36 frames, 296–331, 25 cm depth prior | Strict numerical convergence; maximum contact gap **0.010562 cm**; maximum bone-length error about **5 × 10⁻¹⁴ cm**. Physical feasibility remains conditional. |
| Contact-region review | Same 503 contact schedules, whole observed hold regions | **Zero** frames violate the corresponding pairwise region-distance lower bound. This removes that particular impossibility certificate; it does not prove simultaneous whole-body feasibility. |
| Source-contact proxy ablation | Keyframes 314/486 and 24 native frames 474–497 | Frame 486 gap falls from **11.018 cm** with hold centers to **3.284 cm** with source-derived proxies; still fails the unchanged 0.1 cm tolerance. Contact targets changed, so this is not a same-constraint solver improvement. |
| Neural depth probe | Official Depth Anything V2 Small, four frames, input sizes 280 and 518, CPU | Real surface-depth predictions and a conditional XYZ point cloud were produced. Estimates vary materially with resolution and are not used as joint-depth truth. |
| Runtime probe | Joint 12/36/72-frame fits | 24.9/32.8/51.2 seconds at the initial iteration budget. All three retain nonconvergence. This is a bounded-window benchmark, not a long-video or real-time claim. |

The primary matrix took 273.9 seconds in this local run. Its 23/24 distinct static
fits converged; the frame-92, 10 cm case did not. None of its six initial clip fits
met the strict convergence check. The selected 36-frame continuation used a
further 46.8 seconds and reached a freshly evaluated normalized projected
gradient of 8.72 × 10⁻⁶. The original solve time is separate from that continuation.
Nonconverged original results remain available rather than being relabeled.

## The main failure finding: contact centers are a strong assumption

At frame 486, the H001–H032 center distance is 214.509 cm, while the configured
180 cm body's same-side connecting chain permits at most 200.340 cm. Under those
fixed centers, at least one endpoint must miss by 7.085 cm or more. Rotating the
wall or adding body depth cannot remove this Euclidean-distance contradiction.

However, the minimum distance between the **whole observed regions** is only
185.653 cm. The center approximation discards a large part of H032's usable
image region. Thus the 83 center-span conflicts must not be presented as evidence
that the target person is too short, or that the human route is impossible.
Mask errors, scale assumptions, joint-to-contact offsets and different body
proportions remain possible causes.

The proxy experiment selects episode-constant candidates from source RTMW
fingertips and shoe landmarks near the observed masks, without using the target
body to choose favorable locations. Original hold IDs and confidences stay
unchanged. These are **unreviewed contact proxies**, not manually confirmed grip
points and not anatomical wrist/ankle centers. The late clip has 12/24 contact
failures with those proxies, versus 24/24 in the center-target clip, and remains
unconverged. This motivates reviewing contact geometry before stronger fitting.

Inspect [frame 486 contact evidence](../outputs/trial3/spatial_experiments/contact_regions_180cm/contact_evidence_0486.jpg),
[the proxy viewer](../outputs/trial3/spatial_experiments/contact_regions_180cm/viewer.html)
and [pairwise region bounds](../outputs/trial3/spatial_experiments/contact_regions_180cm/region_span_audit.csv).

## Angle, depth and force sensitivity

The main camera has a fitted focal length of 566.74 px and reserved-grid RMS
reprojection error of 0.371 px. Other plausible principal-point hypotheses fit
nearly as well (about 0.361 px) but imply focal lengths around 303 and 724 px and
substantially different camera depth. These are internal consistency results,
not independent physical calibration accuracy.

Rotating wall, camera and skeleton together preserves image projections to
floating-point precision. An image fit therefore cannot select a true gravity
angle here. The angle comparison changes the relationship to gravity, not the
observed pixels. The preserved historical 90° wall/floor assumption is treated
as the unmeasured 0°-from-vertical scenario.

For the **same converged 36-frame trajectory**, the independent force audit gives:

| Assumed tilt from vertical | Conditional balance, wall-normal feet | Conditional balance, upward-normal feet | Eligible-evidence balance, wall-normal feet |
| --- | ---: | ---: | ---: |
| 0° | 16/36 | 16/36 | 7/36 |
| 15° | 8/36 | 16/36 | 7/36 |
| 30° | 8/36 | 8/36 | 7/36 |

These counts depend on assumed friction, surface normals and force capacities.
All frames still have geometry uncertainty, and source height is unknown.
Numerical convergence and matching contact points do not establish a balanced
or executable climbing movement. Failed contacts are excluded from support.

The 10/25/40 cm soft priors yielded median hip depths of approximately
12.82/24.80/38.04 cm in the initial reachable clip. The late center-target clip
instead hit the 3 cm lower bound for each prior. It must not be described as
three successfully maintained fixed-depth trajectories.

The optional force/support objective was also executed. Its frame-486 fit
converged but retained an approximately 11.021 cm contact gap; frame 314 and the
36-frame fit did not converge. Increasing physical penalties did not remove
incompatible contacts. It is retained as an experimental comparison, not the
preferred default for larger runs.

## Iterations and the selected implementation

Early free-angle spatial fits were slow and sensitive to force weighting.
The selected implementation uses exact two-bone geometry for contacted limbs,
optimized elbow/knee swivel, a rigid torso and fixed bone lengths. A discovered
swivel-axis switching discontinuity was removed by freezing the reference basis
within each solve. Temporal penalties operate on XYZ joints. A tiny reach
reserve avoids the fully extended two-sphere singularity without changing the
0.1 cm audit tolerance.

Sparse damped Gauss–Newton plus a bounded L-BFGS-B fallback is followed by a
fresh stationarity check and independent force LPs. Continued joint fitting
resolved the selected reachable clip's numerical nonconvergence. The optional
support-optimized variant remains less reliable in this small experiment.
Development prototypes remain outside the repository in the working archive.

The practical next backend is **camera/plane geometry + fixed-length spatial
IK + independent contact/force audits**, with neural depth as an uncertain prior
for investigation. Contact evidence needs improvement before scaling the same
constraints to a much longer sequence.

## Open-weight depth and larger videos

[Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2) was
actually executed, using pinned source and SHA-256-verified official weights.
No additional environment was installed. The model and source cache are outside
the repository; the existing environment can download them with the documented
optional command. A representative surface point cloud is available in the
[518-size probe](../outputs/trial3/spatial_experiments/depth_probe_518/conditional_neural_surface.ply).

| Native frame | Median source-keypoint surface depth, input 280 | Same quantity, input 518 |
| --- | ---: | ---: |
| 296 | 13.90 cm | 20.69 cm |
| 314 | 13.03 cm | 23.33 cm |
| 334 | 18.16 cm | 20.43 cm |
| 486 | 16.34 cm | 32.78 cm |

These depths use an affine inverse-depth fit to the assumed wall camera and
refer to visible surfaces, not joint centers. Warm inference times were roughly
0.09 s and 0.49 s per image, respectively, in this small CPU probe. Resolution
sensitivity and camera ambiguity prevent a metric-accuracy claim.

[Depth Anything 3 Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main/da3_streaming),
[VGGT](https://github.com/facebookresearch/vggt),
[COLMAP](https://colmap.github.io/tutorial.html) and
[GVHMR](https://github.com/zju3dv/GVHMR) were investigated through their official
documentation, but not locally run. Their capabilities, licensing and setup
constraints are listed in the method guide. DA3's chunked workflow is a candidate
for a future multi-view scene experiment; its GPU benchmarks do not establish
performance on this Mac or accuracy on climbing footage.

The 12/36/72-frame IK benchmark reported about 232 MB process peak RSS, measured
cumulatively rather than independently for each window. A long-video streaming
solver with overlap, seam checks and global refinement is **not implemented**.
For larger-resolution footage, resize analysis consistently with intrinsics,
sample neural depth at informative frames, and render separately. For longer
sequences, first benchmark bounded windows with convergence and continuity checks.

Recommended next order:

1. Review source contact positions and hand/foot offsets, then implement region
   contacts with evidence priors rather than silently moving targets to make IK pass.
2. Obtain wall tilt and a calibrated camera or second useful viewpoint; preserve
   uncertainty when these are unavailable.
3. Improve occluded joint guidance and repeat the same short-clip ablations.
4. Add overlapping-window spatial optimization and measure seam error, solve
   convergence and audit failures before expanding to whole long videos.

## Reproduce the additional studies

The main matrix is cached by input, body, code and artifact hashes. The additional
scripts refuse to overwrite existing output directories. From the repository:

```sh
# Main 180 cm matrix; existing verified output is reused.
.venv/bin/python scripts/run_trial3_3d_experiment.py

# Optional support-objective comparison.
.venv/bin/python scripts/run_trial3_3d_experiment.py --mode support \
  --angles 15 --depths 25 --keyframes 314 486 --clip reach:296:331

# Runtime probe; choose a fresh directory for every rerun.
.venv/bin/python scripts/benchmark_trial3_3d.py \
  --output outputs/trial3/spatial_experiments/my_scaling

# Continue the whole existing 36-frame solve, then compare gravity angles.
.venv/bin/python scripts/benchmark_trial3_3d.py --lengths --max-iterations 8000 \
  --refine-from outputs/trial3/spatial_experiments/target_180cm_8aa2d71aa739/reachable_tilt15_depth25 \
  --output outputs/trial3/spatial_experiments/my_refinement
.venv/bin/python scripts/compare_refined_trial3_angles.py \
  --source-case outputs/trial3/spatial_experiments/my_refinement/continued_trajectory \
  --output outputs/trial3/spatial_experiments/my_refined_angles

# Review region bounds and fit unreviewed source-derived contact proxies.
.venv/bin/python scripts/review_trial3_contact_regions.py \
  --output outputs/trial3/spatial_experiments/my_contact_regions
```

## Saved outputs and validation

- [Main matrix, meshes, videos and metrics](../outputs/trial3/spatial_experiments/target_180cm_8aa2d71aa739/)
- [Converged angle comparison](../outputs/trial3/spatial_experiments/refined_angles_180cm/)
- [Continuation solver history](../outputs/trial3/spatial_experiments/refinement_180cm/)
- [Support-objective comparison](../outputs/trial3/spatial_experiments/target_180cm_eb287ac0a4f4/)
- [Contact proxies and region bounds](../outputs/trial3/spatial_experiments/contact_regions_180cm/)
- [Runtime measurements](../outputs/trial3/spatial_experiments/scaling_180cm/benchmark.csv)
- [Depth probe 280](../outputs/trial3/spatial_experiments/depth_probe_280/) and [518](../outputs/trial3/spatial_experiments/depth_probe_518/)

The targeted suite passes 23 tests, including exact bone geometry, force feasibility,
camera fitting, cache corruption, fixed-basis/temporal derivatives, native-frame
validation and real GLB mesh structure. Video exports contain one rendered image
per native frame, shown at 10 FPS for inspection; the source is 30 FPS. They do
not concatenate earlier 2D clips. Meshes use meters and world Y-up, and contain
an actual wall plane and skeleton cylinders; flat hold markers are not measured
hold surfaces. New notebook cells are verified separately so historical outputs
are not rerun or overwritten.
