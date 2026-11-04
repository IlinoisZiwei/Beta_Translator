# Trial1: full-sequence support optimization

Open [Trial1_Analysis.ipynb](../Trial1_Analysis.ipynb) with **Python (Beta Translator)**.
Set `TARGET_HEIGHT_CM` in Section 1 and run through Section 5 in order.
The saved demonstration uses 180 cm and an explicitly assumed 70 kg mass.
Alternatively, run from the repository:

```bash
.venv/bin/python scripts/run_trial1_support_optimization.py --height-cm 180 --mass-kg 70
```

All **417 native frames, 0–416**, are retained at 30 FPS (13.9 seconds). The baseline
and support-aware optimizer are the same global methods used for Trial3. No clip
stitching, contact-label rewriting, or new hold/movement-order planning occurs.
The optimizer includes contact-force gap coupling, COM reserve objectives,
pre-release unloading and temporal motion limits, followed by independent audits.
See [optimization method](trial3_support_optimization.md) for the common objectives.

## Trial1 geometry and evidence

Trial1 retains its existing uniform reference-image scale: `hypot(80, -21) / 46`,
about 1.798 pixels/cm, tied to the supplied video's SHA256 and reference frame 0.
The original 46 cm distance was measured edge-to-edge between H44/H40 in the
reviewed map. It is independent of the target height and is not inferred from the
source person's body. Up in the reference image is assumed to oppose gravity.

This does **not** recover perspective, depth, wall inclination or camera roll.
There is no Trial1 calibrated plane/hull comparable to Trial3's white-wall grid.
All frames therefore retain `uniform_scale_approximation` and
`image_up_gravity_assumed`. Eligible support means reached, confident and present
in the reference map **under this approximation**, not geometrically validated
physical contact. Contact tolerance 0.1 cm is a numerical threshold, not a claim
of that measurement accuracy. Do not pool Trial1/Trial3 metrics as though geometry
quality were equivalent.

The loader verifies video/map/prediction hashes, all native pose frames, the pose
manifest and model, alignment validity, selected holds, contact tables and model
checkpoints. Invalid or missing input frames are rejected instead of silently
interpolated. Ground/wall/air support is not invented. The deployment movement
acceptance gate remains disabled; this full-sequence run is an experimental
model-based diagnostic, not automatic human-confirmed beta.

The immutable `data/H44_H28_confirmed_reference.json` supplies a separate check
for frames 107–129. Its release/arrival times and three maintained supports define
92 reviewed limb/frame labels. Model labels are compared with those labels but
are not replaced or used as fresh optimization targets. The recorded 165 cm in
that old annotation describes its target-body example, not the source climber's
height; this new run independently uses 180 cm.

## Inspect results

Each configuration and input/code hash gets a separate directory under
`outputs/trial1/support_optimized/`. Existing main/Trial2/Trial3 notebooks, videos,
annotations and data remain intact. The directory includes:

- `trial1_support_optimized_<height>cm_full_sequence.mp4`: full optimized video.
- `trial1_gravity_<height>cm_baseline.mp4`: full gravity baseline for comparison.
- `comparison.csv`, frame/limb diagnostics and issue intervals: unchanged thresholds.
- `trajectory.npz`, `baseline_trajectory.npz`: both pose and force trajectories.
- `release_events.csv`, `preunload_checks.csv`: predicted timing and remaining-support readiness.
- `optimization_constraints.csv`: desired support intervals and unavailable cases.
- `reviewed_reference_comparison.csv`: all 92 saved-reference comparisons.
- `review.html`: original video alongside either result, synchronized seeking,
  per-frame labels/confidence/gaps, saved reviewed labels, and jump buttons for failures.

The skeleton videos use a fixed reference-frame background, including the person
visible in frame 0. That person is not a current-frame replay; use the synchronized
review page to see the actual moving source video beside the target skeleton.

Open `review.html` from its existing directory, or use its link in the notebook.
Keep the repository's `videos/trial1.mp4` in place; videos are linked rather than
embedded or copied. The HTML review does not save or confirm annotations. Its
numeric table always describes the optimized run, even while viewing the baseline.

All failed or uncertain model checks remain visible. Solver iteration limits are
recorded; final termination is not a proof of optimality. Reserve and unloading
are quasi-static checks with assumed capacity/friction, not real climbing safety.
