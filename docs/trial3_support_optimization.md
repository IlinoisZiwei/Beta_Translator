# Trial3: support-aware pose optimization

Open `Trial3_Analysis.ipynb` using **Python (Beta Translator)**, set
`TARGET_HEIGHT_CM` in Section 1 (saved demo: 180 cm), and run through Section 9
in order. Or use:

```bash
.venv/bin/python scripts/run_trial3_support_optimization.py --height-cm 180 --mass-kg 70
```

The target mass defaults to an explicit 70 kg assumption. Source height/mass,
contact labels/timing, calibration, hold map and audit thresholds are unchanged.
The gravity and geometric workflows are in Sections 8 and 7 of the merged notebook;
all previously saved results remain available.

## What now changes the pose

The previous whole-video gravity trajectory is a verified initialization. Every
native pose and contact force then participates in new **global** optimizations.
All 503 frames remain coupled by velocity/acceleration terms; no clip or per-frame
result selection is performed. Fixed body lengths and bend limits remain in force.

- **Contact validity:** stronger endpoint residuals attract limbs to their decoded
  holds. Iteratively reweighted contact residuals prioritize achievable contacts
  when the full decoded set is incompatible with the target body. Every original
  contact is still audited at the same 0.1 cm threshold. Errors on incompatible
  contacts can grow; the before/after table includes maximum gap, not just counts.
  A force-times-gap penalty discourages assigning force to a missed target.
- **Reserve:** two linear programs compute an admissible horizontal COM interval
  from desired support forces, the same simplified foot-force cone, and assumed
  component capacity. The interval uses the audit's reserve threshold plus a small
  optimization buffer, and is narrowed for the same horizontal COM perturbation.
  A pose-dependent penalty moves the COM into the interval. If eligible desired
  contacts cannot supply an interval, a conditional interval is explicitly labeled.
  Empty intervals remain in the exported constraint table. Desired contacts are
  optimization objectives, never evidence that a limb has reached a hold.
- **Unloading:** intervals are also computed after removing all simultaneously
  departing limbs, using only supports present in each pre-frame. Candidate force
  on departing limbs is penalized with an increasing weight toward release.
  Fixed-pose readiness and actual departing candidate load are checked separately.
  A last-pre-release load above 1% body weight is flagged, even if remaining
  contacts could theoretically support the pose. Direct hold changes retain
  uncertain release timing.
- **Continuity:** the original temporal penalties remain, plus soft limits starting
  at 95% of the unchanged 5 cm/frame hip and 15 degree/frame angle warning limits.
  These are numerical motion controls, not human acceleration or dynamic limits.

For planar quasi-static balance, `sum fx = 0`, `sum fy = 1` in body-weight units.
Moment balance gives `COM.x = sum(hold.x * fy - hold.y * fx)`. Bounded admissible
forces therefore project to a COM-x interval. This derivation uses the simplified
planar contact model; it is not the polygon formed by hand/foot image locations.

## Independent evaluation and remaining limits

After optimization, unchanged audits recompute contact validity and feasible
forces at **reached hold centres**. Missed contacts are excluded. Low confidence
and extrapolated contacts remain conditional, and reference visibility uncertainty
remains visible. The video's arrows are candidate optimization forces; independent
LP forces are stored separately. A small candidate equilibrium residual cannot
turn a missed hold into usable support.

The objective is nonconvex and uses soft constraints. Reweighting runs a fixed number
of rounds; final solver termination applies to the last frozen-weight problem,
not proof that the outer reweighting process converged. Finite results can still
fail contacts, reserve, unloading or continuity; iteration-limit exits are marked
unconverged. A termination condition is not proof of a global optimum. This remains
2D quasi-static modeling without inertia, wall depth, measured grip normals,
strength, joint torque or collision. It does not predict real climbing success
or automatically choose new holds/movement order.

## Results and preservation

Each configuration, baseline and input/code hash gets a new directory under
`outputs/trial3/support_optimized/`. Intact matching results are reused; changed
artifacts are preserved and a separate directory is created. The baseline summary
and all output hashes are recorded. No earlier videos or notebooks are overwritten.

- `trial3_support_optimized_<height>cm_full_sequence.mp4`: native frames 0–502 at 30 FPS.
- `comparison.csv`: old/new counts and maximum gap/step metrics with unchanged audits.
- `frame_diagnostics.csv`, `limb_diagnostics.csv`, `issue_intervals.csv`: all failures and uncertainty.
- `preunload_checks.csv`, `release_events.csv`: readiness and predicted timing.
- `optimization_constraints.csv`: desired support sets, intervals and unavailable cases.
- `trajectory.npz`: optimized and initial poses, forces, joint positions and support masks.
- `summary.json`: parameters, each optimization round, comparisons and provenance.

The new combined numerical pass also requires continuity and actual last-pre-release
unloading. The comparison recomputes this same definition for both trajectories;
it should not be compared directly to the older notebook's narrower pass count.
