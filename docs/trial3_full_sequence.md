# Trial 3: full-sequence 180 cm adaptation

Open `Trial3_Analysis.ipynb`, configure the target body in Section 1, and run through
Section 7 in order with the **Python (Beta Translator)** kernel, or:

```bash
.venv/bin/python scripts/run_trial3_full_sequence.py
```

This workflow reads the existing Trial 3 video, reviewed hold map,
pose/alignment caches, decoded contacts, and white-wall calibration. It does not
retrain models, rebuild hold IDs, or replace the
single-move IK outputs. Results live in separate, content-addressed directories
under `outputs/trial3/full_sequence/`. Identical verified runs are reused;
changed inputs or code create a new directory.

## What the video means

The target skeleton uses fixed segment lengths for 180 cm with the existing
default ratios. Every native frame from 0 through 502 is included at 30 FPS,
for 16.7667 seconds. All 503 pose vectors are jointly optimized with velocity
and acceleration penalties. Sequential initialization is only a starting guess;
the output is not a concatenation of the 13 movement proposals or old clips.

Each limb can change its decoded contact independently. Known hold centers are
soft targets. When contacts conflict with body reach, the solver keeps the body
lengths and reports the remaining contact gap. A finite result can be exported
even if its optimizer has not converged, but every frame is then marked with an
optimizer warning. These diagnostic exports are separate from the strict
single-move solver, whose acceptance rule is unchanged.

The video displays per-limb model confidence, contact gap, coverage status, and
frame issues. Dashed lines connect missed targets to fitted endpoints. Red marks
contact mismatch, purple marks extrapolated hold geometry, amber marks free or
uncertain contacts, and green means only that a numerical gap is at most 0.1 cm.
The timeline shows constraint warnings, uncertainty, or numeric tolerance status.
These colors do not indicate physical climbing feasibility.

## Uncertainty and modeling choices

- All contact labels are unreviewed model predictions, even above the model
  confidence threshold. Confidence coverage is not Trial 3 accuracy.
- `free` means no decoded fixed hold target. It does not distinguish standing on
  the floor, pressing the wall, or moving through the air. Free feet are explicitly
  marked as unresolved support. `wall_contact` has no measured contact location.
- H053 retains the existing white-wall homography extrapolation. It receives
  0.15 of the in-coverage contact penalty weight before confidence scaling and is
  always flagged. The observed-hole coverage has not been extended or validated.
- Free endpoints follow a soft guide from the observed projected wrist/ankle
  positions. This is not recovery of the new climber's true trajectory.
- Reference landmarks are projected onto the wall. Body depth, hold depth, the
  second wall, and true grasp locations remain unknown. Reference points outside
  the observed-hole hull or white-wall polygon receive separate flags.
- The global objective trades contact fit against reference agreement and
  temporal smoothness. It does not guarantee that every geometrically reachable
  frame will meet the strict 0.1 cm diagnostic tolerance.
- Warning thresholds for rapid changes are 15 degrees or 5 cm per source-frame
  interval. They are diagnostic settings, not validated biomechanical limits.
- No force, friction, balance, collision, grasp, or alternative-beta planning is
  modeled. The source climber's height remains unknown.

## Saved evidence

- `trial3_target_180cm_full_sequence.mp4`: complete target-body diagnostic video.
- `trajectory.npz`: 503 pose vectors, fixed-length joints, references, targets,
  decoded labels, confidence, coverage masks, and contact residuals.
- `frame_diagnostics.csv`: per-frame issues, span conflicts, pose changes, and
  numerical contact gaps.
- `limb_diagnostics.csv`: 2,012 limb-frame contact records with original labels.
- `issue_intervals.csv`: inclusive frame ranges for each issue; issues may overlap.
- `summary.json`: input/code/output hashes, solver status, body parameters,
  thresholds, and diagnostic counts.
- `poster.png` and `storyboard.png`: labeled previews.

The original analysis notebook still describes its single-move result. Use this
companion notebook and its separate summary for the full-sequence experiment.
