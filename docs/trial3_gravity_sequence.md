# Trial3: continuous quasi-static gravity IK

Open `Trial3_Analysis.ipynb`, configure the target body in Section 1, and run through
Section 8 in order with the **Python (Beta Translator)** kernel, or:

```bash
.venv/bin/python scripts/run_trial3_gravity_sequence.py --height-cm 180 --mass-kg 70
```

Height, body ratios, assumed mass, foot friction and force-component bounds are
configurable in Section 1 of the merged notebook. The source climber's height/mass
remain unknown. Sections 7–9 share the configured target body; the saved
demonstrations use 180 cm.

## Joint trajectory model

All native frames 0–502 enter a single sparse least-squares problem. Each frame
contains 11 pose variables and 8 potential contact-force components. Sequential
initialization is only a starting point: every pose and force is subsequently
optimized together, with first/second pose differences and force differences
for the same limb on the same decoded hold. No independent clips are concatenated.
Fixed segment lengths follow the existing body model. Bend bounds remain ±150°.

Gravity acts downward through the existing segment-mass center of mass in the
calibrated white-wall coordinate system. Force and moment residuals target zero
under a quasi-static assumption. Feet have nonnegative upward force with the
simplified bound `abs(fx) <= mu * fy`; hands allow any in-plane direction within
component bounds. The default component limit is one body weight. It is not a
measured strength limit. With those defaults mass alone scales forces, not poses.
Contact forces are inferred and are not unique or measured.

Only known decoded hold labels activate force variables. Free, unlocated wall
and unknown support receive exactly zero assigned force. In particular, a free
foot is not silently treated as a ground contact. H053's old extrapolated target
retains its reduced geometry weight and explicit uncertainty. Contact labels,
timing, reviewed hold IDs, pose inputs and calibration remain unchanged.

## Diagnostics and interpretation

Contact gaps are checked separately from balance. Numerical thresholds are
0.1 cm for contact, 1% body weight for net force and foot-constraint excess, and
1% body weight × torso length for moment. A converged optimizer is not proof of
global optimality or of meeting those tests. Finite unconverged output can be
exported, but all frames then carry an optimizer warning.

Forces are applied at fitted limb endpoints. If an endpoint misses its hold and
carries appreciable force, `force_at_missed_contact` is flagged and contact tests
fail even if the force/moment residual is small. Low-confidence and extrapolated
force-bearing contacts receive a separate flag. Reference visibility, support
ambiguity, span conflicts, out-of-coverage geometry and rapid changes stay visible.
Numerical-pass counts and warning counts can overlap through uncertainty; issue
counts themselves also overlap.

This is not dynamic simulation: acceleration-induced inertial forces, body depth,
out-of-plane moments, measured hold normals, grasp mechanics, grip strength, joint
torques and collision are absent. No climbing success probability or alternate
movement order is predicted. The animation must be presented as a diagnostic
target-body candidate, not a physically validated route.

## Outputs and preservation

Every run gets a separate directory keyed by configuration and verified input/code
hashes under `outputs/trial3/gravity_sequence/`. Matching intact results are reused.
Changed or damaged results are preserved; a new directory is created instead.

- `trial3_gravity_<height>cm_full_sequence.mp4`: all 503 frames at 30 FPS (~16.77 s).
- `frame_diagnostics.csv`: pose, COM, net force/moment, checks and flags per frame.
- `limb_diagnostics.csv`: 2,012 decoded limb states, targets, gaps and force vectors.
- `issue_intervals.csv`: inclusive source-frame ranges for each issue.
- `trajectory.npz`: every fitted/reference pose, joint, contact, force and residual.
- `poster.png`, `storyboard.png`, `frame_*.png`: labeled diagnostic stills.
- `summary.json`: parameters, solver status, counts, limitations and provenance.

Orange arrows are candidate forces on eligible contacts; gray arrows are candidate
forces on excluded or conditional contacts; cyan marks COM and gravity. Arrows are
projected from wall-coordinate vectors using one body weight = 50 cm display length.
Red denotes constraint warnings, amber uncertainty, purple extrapolated contacts.
Green never means proven physical feasibility. Earlier videos, notebooks, human
reviews and endpoint gravity comparisons are retained.

## Reached-contact support audits

These audits run after the global pose/force fit. They identify failures without
changing the trajectory, predicted contact labels, calibration or original results.
They do not enforce a new motion plan or simulate a load-transfer maneuver.

1. **Contact validity:** a decoded located hold is usable only if its endpoint gap
   is at most 0.1 cm. Missing or missed contacts are excluded. The `eligible` set
   additionally requires the existing prediction confidence threshold (0.8) and
   calibrated coverage; the `conditional` set also permits low confidence and
   extrapolated geometry. Neither set is human-confirmed ground truth. A contact
   can have multiple limitations; the state label prioritizes missing support,
   gap, extrapolation, then confidence. Original per-limb flags remain available.
2. **Balance reserve:** with pose/COM fixed, a linear program reallocates forces at
   the usable **hold centres**, independently of the candidate endpoint forces.
   It enforces zero net force/moment and the same simplified foot-force cone.
   It minimizes peak force-component utilization `rho`; reserve is `1-rho`.
   Positive reserve is unused assumed component capacity, zero is the limit, and
   negative reserve is a capacity deficit. Structural infeasibility and absent
   support produce N/A, not a fake zero margin. This reserve is not a probability,
   joint-strength measurement, full friction reserve or dynamic stability margin.
   The same check repeats with COM shifted horizontally by +/-2 cm. The worst
   reserve is reported only if all three LPs have a finite solution. This is a
   chosen sensitivity scenario, not a measured uncertainty interval. A reserve
   below 10% or failed perturbation produces a warning; 10% is a configurable
   diagnostic threshold, not a biomechanical standard.
3. **Pre-release unloading:** decoded hold departures define events. For up to
   six preceding source frames (0.2 seconds), remove **all simultaneously departing
   limbs** and repeat the fixed-pose force/reserve checks. Each window stops when a
   departing limb no longer has its event's old hold label. Only contacts available
   in that particular pre-frame can support the body; future arrivals cannot help.
   Direct hold-to-hold transitions and unlocated destinations carry timing
   uncertainty. These are counterfactual support-readiness checks, not proof that
   the limb was already unloaded or that a feasible load-transfer trajectory exists.
   Candidate departing load is informational. Outside the window the check is
   `not_applicable`, never an automatic pass.

Audit statuses:

| Status | Meaning |
| --- | --- |
| `passes_model_checks` | Eligible support balances the fixed pose and passes the selected reserve/sensitivity tests. Other input uncertainties can remain. |
| `fails_located_support_model` | Even the conditional set cannot balance the nominal pose within assumed limits. Unlocated ground/wall support is still unknown. |
| `uncertain_support` | Nominal balance requires the conditional contact assumptions. |
| `low_margin` | Eligible nominal balance exists, but reserve or COM sensitivity fails the chosen threshold. |
| `uncertain_timing` | Support check succeeds or has low margin, but departure timing is ambiguous. Timing uncertainty is also retained separately when support fails. |
| `not_evaluated` | An LP/numerical validation could not establish a result. |
| `not_applicable` | No upcoming release to check at this frame (unloading only). |

`balanced` remains the old candidate-force residual check. The new
`numerical_checks_pass` also requires eligible support reserve/sensitivity and any
applicable unloading check to pass; `legacy_numerical_checks_pass` preserves the
old count. Neither is physical validation, and input-uncertainty flags can overlap.

Additional outputs: `release_events.csv` (one row per departure),
`preunload_checks.csv` (one row per event/pre-frame, including both support sets,
solver statuses, margin and timing), and `support_checks.npz` (contact masks,
classifications and independent audit forces). Frame/limb CSVs include the same
support eligibility and reserve information. Audit forces for capacity-exceeded
cases can exceed the assumed limit; their negative reserve and status must be read
alongside the vectors. N/A forces are zero placeholders with an explicit status.

Configure `AUDIT_SETTINGS` in the notebook, or for example:

```bash
.venv/bin/python scripts/run_trial3_gravity_sequence.py --height-cm 180 --mass-kg 70 --com-perturbation-cm 2 --unload-lookahead-frames 6 --margin-warning 0.1
```

The audit source hash and settings participate in the cache identity. New runs
preserve the earlier gravity-only and geometric videos at their original paths.
