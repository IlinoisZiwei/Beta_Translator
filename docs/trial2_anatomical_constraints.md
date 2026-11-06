# Trial2: MoBL-inspired arms and constrained trajectories

The 180 cm experiment adds an articulated shoulder–elbow–forearm–wrist chain to Trial2. It retains the previous source video, frame numbers (0–975, 30 fps), body segment lengths, reviewed hold regions, and assumed middle/upper wall angles (15° / 0°). Old trajectories and videos remain available. This experiment is not an OpenSim simulation or a certification that a climber can execute the motion.

## Sources and model scope

- Original MoBL-ARMS project: <https://simtk.org/home/upexdyn>.
- Saul et al., *Benchmarking of dynamic simulation predictions in two software platforms using an upper limb musculoskeletal model*, <https://doi.org/10.1080/10255842.2014.916698>.
- Parameter source: [MoBL OpenSim snapshot distributed with O2MConverter](https://github.com/aikkala/O2MConverter/blob/ad2599d5ecb2c18b44570b4d0b53e062ac9b82ae/models/opensim/MoBL_ARMS_OpenSim_tutorial_33/MoBL_ARMS_model_for_mujoco.osim), pinned commit `ad2599d5ecb2c18b44570b4d0b53e062ac9b82ae`. `data/trial2_arm_model_sources.json` records its coordinate ranges and provenance. These are the ranges of this particular model snapshot, not universal clinical limits or the latest personalized MoBL release.
- [OSQP Python interface](https://osqp.org/docs/interfaces/python.html) is used for sparse constrained quadratic subproblems. The notebook still uses the repository's single Python environment; `bash setup.sh` installs the added `osqp==1.0.5` requirement.

The reduced model uses the source's rotation axes and shoulder plane/elevation/axial-angle composition, a signed elbow hinge, forearm pronation/supination, and two wrist coordinates. Its left arm is a sagittal reflection of the right. Existing Beta Translator segment lengths and simplified mass fractions are retained. Scapulothoracic rhythm, shoulder-girdle translations, source bone offsets, the source's extra coupled wrist coordinates, muscle/ligament dynamics, and full self-collision are omitted. The reduced chain must therefore be called **MoBL-inspired**, not a full import or reproduction of MoBL-ARMS.

## Declared arm bounds

Ranges below come from that snapshot, rounded to degrees. Speed and acceleration limits are **project assumptions for conservative trajectory regularization**, not limits measured by MoBL, clinical claims, or personalized capacity estimates.

| Coordinate | Range (degrees) | Speed cap (deg/s) | Acceleration cap (deg/s²) |
|---|---:|---:|---:|
| Shoulder elevation plane (`elv_angle`) | −90 to 130 | 120 | 600 |
| Shoulder elevation (`shoulder_elv`) | 0 to 180 | 150 | 750 |
| Shoulder axial rotation (`shoulder_rot`) | −90 to 20 | 120 | 600 |
| Elbow flexion | 0 to 130 | 180 | 900 |
| Forearm pronation/supination | −90 to 90 | 180 | 900 |
| Wrist deviation | −10 to 25 | 120 | 600 |
| Wrist flexion/extension | −70 to 70 | 150 | 750 |

The old model's generic 155° bend threshold is not substituted for the new 130° elbow range. More restrictive anatomy may increase contact error. That cost remains in the comparison.

Initialization prioritizes the old wrist endpoints and uses old elbows only as weak hints. It fits against the actual bounded torso roots and tries alternative shoulder branches when wrist error remains above 0.5 cm. This avoids copying an impossible legacy elbow configuration into the new anatomical chain. The prior initialization and its short-clip failures are retained as an earlier experiment.

The elbow is no longer an independent world-direction/swivel variable. It follows the shoulder orientation and signed hinge; pronation and wrist motion change hand orientation without moving the elbow or wrist point. Wrist orientation is inferred: the video inputs do not establish a measured palm frame. A small articulated hand connector (0.035 × height along the hand, 0.012 × height normal offset) provides a palm/contact proxy, with a separately reported 2 cm gap threshold. Passing this proxy does not validate finger grip or hold surface orientation.

## Hard constraints, soft objectives, and independent checks

`beta_translator/anatomical_arms.py` defines the chain, fixed lengths, bounds and coordinate-rate scales. `beta_translator/trial2_anatomy_optimizer.py` builds one sparse objective over consecutive native frames. `beta_translator/constrained_sequence_solver.py` provides OSQP and feasibility-projection primitives; `restored_sequence_solver.py` solves the revised short experiments and full run. The full run uses `temporal_control_basis.py` and `restored_sequence_solver.py`: approximate quadratic steps are restored to the feasible set inside the optimizer before nonlinear line search. Fresh stationarity and complementarity are checked; failure to converge remains explicit.

Hard constraints:

- Fixed segment lengths through forward kinematics; a rigid torso preserves shoulder and hip widths.
- Bilateral shoulder/elbow/forearm/wrist ranges above.
- Signed first differences: `abs(q[t+1] - q[t]) * fps <= speed`.
- Signed second differences: `abs(q[t+2] - 2*q[t+1] + q[t]) * fps**2 <= acceleration`.
- Hip translation: each coordinate ≤100 cm/s and ≤350 cm/s²; torso coordinates ≤90 deg/s and ≤450 deg/s²; leg direction coordinates ≤240 deg/s and ≤1200 deg/s². These are project assumptions. They are componentwise caps, not caps on resultant Cartesian speed or geodesic segment angular speed.
- Persistent hold-contact region coordinates remain shared within each contact episode and inside its rectangle.
- Unknown/free contacts have exactly zero optimized force. Other force components retain the existing ±1 body-weight bounds.

Soft objective terms:

- Source 2D reprojection with visibility weights; weak pose/depth priors.
- Wrist/ankle contact target errors using the same legacy contact-region offsets; a separate articulated palm proxy.
- Wall/floor penetration and camera-front residuals; the previous 155° leg-bend penalty. Legs are not a new anatomical hip/knee/ankle model.
- Quasi-static net force/moment, foot friction, pre-release unloading, and force/contact-gap coupling.
- Coordinate, physical rotation-matrix, and Cartesian continuity. The physical rotation-matrix term supplements coordinate smoothing near angular-coordinate ambiguities.

The optimizer jointly fits contact, pose and force variables. It does not smooth exported XYZ afterwards, snap endpoints to holds, interpolate missing frames into a final plan, or concatenate winners from separate clips. Independent audits retain the old 0.1 cm endpoint-contact tolerance, force feasibility, ±2 cm COM reserve and pre-release checks, and add ROM/rate/acceleration/palm-proxy checks. Rotation rates of actual segment matrices and elbow-swivel diagnostics are exported separately. The elbow-swivel diagnostic is undefined near straight arms (elbow-circle radius ≤5 cm) and is masked there; it is not a shoulder joint coordinate.

## Experiment order and selection

1. Compare three predetermined anomaly clips: hinge transition 475–515, right elbow 550–595, left elbow 615–650.
2. Compare two smoothness profiles (`balanced=1`, `steady=2.5`) using identical constraints, input data, and initialization.
3. Fit each full 976-frame candidate independently from the same full-video initialization. For scalability and continuity, each joint coordinate uses 125 clamped cubic controls at 8-frame knot spacing. Nonnegative basis weights preserve the bounded control-coordinate range; native-frame first/second differences enforce rate/acceleration caps. All 976 frame residuals and contacts participate in the same solve. This restricts high-frequency motion and can increase contact mismatch; it is an explicit model choice. Each frame is regenerated by forward kinematics, not Cartesian interpolation or post-fit smoothing. Short-clip solutions are not pasted into the final trajectory.
4. Apply a feasibility gate first: anatomy, rates, acceleration, geometry/contact, support, unloading, palm proxy, motion checks, and solver convergence must all pass. Geometry and hand orientation remain uncertain even if the numerical gate passes.
5. If none passes, choose a **diagnostic-only, infeasible** trajectory. The declared ordering is anatomical/temporal failure count, worst physical failure fraction, union of failed frames, palm-proxy failures, maximum contact error, then convergence. This is an explicit diagnostic policy, not proof of an optimal human route. Every candidate and the first differing selection criterion remain in the output table.

`rank_candidates` in the anatomical optimizer supersedes the geometry-first legacy ranking **for this new experiment**. Historical angle experiments and their saved results remain unchanged. This is a controlled comparison at one assumed wall geometry, not a new search over wall angles.

## Run and inspect

```sh
bash setup.sh
.venv/bin/python scripts/run_trial2_anatomy.py --stage clips
.venv/bin/python scripts/run_trial2_anatomy.py --stage all
```

Open `Trial2_Analysis.ipynb`, select **Python (Beta Translator)**, run Section 1, then Sections 16–18 to inspect the saved experiment. Recomputing is opt-in because fitting full trajectories takes time. Videos use file references, fixed cameras, identical before/after box limits, native frame numbers, and 30 fps. All frames remain visible, including failed and uncertain ones.

Short native-frame results live under `outputs/trial2/anatomy_plans/target_180cm_<hash>/`; the recommended full controlled trajectory lives under `target_180cm_controls_<hash>/`. The full run copies the verified short comparison artifacts with their provenance so the result folder remains portable. Each numerical run records input/code hashes, configurations, trajectories, per-frame audits, solver history, full-candidate rankings, before/after metrics, and per-frame explanations. Rendered artifacts have their own manifest. Previous Trial2 result folders are not replaced.

## Necessary reach diagnostic

`trial2_reach_diagnostics.py` applies a triangle-inequality bound to each pair of selected wrist/ankle targets: their distance cannot exceed the two limb lengths plus the fixed separation of their torso/hip roots. If the excess is `e > 0`, at least one of the two endpoint errors must be at least `e/2`. An excess above 0.2 cm therefore prevents both endpoint targets from passing the unchanged 0.1 cm contact tolerance, even before anatomical ranges are imposed.

This is a necessary condition for the **chosen targets**. It does not search the entire hold regions, identify the correct camera/wall/floor geometry, or judge the filmed human. The early hand/ground targets already conflict in the old baseline; they remain explicitly flagged rather than being repaired through bone stretching or ignored contacts. Videos and derived CSVs label these conflicts. An additional swivel table compares only the limb/time intervals observable in both models, avoiding changes in the near-straight-arm mask.
