# Contact matching and review

Run the notebook locally using the repository's `.venv` kernel. Section 6 generates drafts; Section 7 is the shared prediction viewer and manual endpoint editor.

## Rules

`contact_matching.py` warps selected hold masks into each frame, checks landmark visibility and distance, and requires a unique candidate for at least 0.5 consecutive seconds. Default margins are 20 pixels for wrists and 10 for foot-index landmarks, scaled from 720px video width. `group_contact_intervals()` handles persistence separately from frame matching.

Ambiguity, low visibility, missing poses, and unreliable alignment break intervals. `no_candidate` does not imply `free`: a smear, missed detection, or unselected hold may explain it. Match scores describe proximity, not contact probabilities or grip.

## Review and persistence

The Section 7 slider shows the raw frame, projected holds, and predictions. Set BEFORE and AFTER and explicitly select four contacts for each. Predictions never automatically populate labels. Changing a frame resets that endpoint's labels. Draft matching uses foot-index points; IK uses ankles, so review these approximations carefully.

**Save annotations** writes frames, labels, moving limb, notes, and body/wall settings to `outputs/state/endpoint_annotations.json`. **Load annotations** restores only matching video/map/route annotations. `unknown`, `free`, and `wall_contact` remain distinct; only hold IDs and `free` currently work in IK.

## Stable inputs and export

The example map has 65 candidates and 21 selected route holds. Its H44/H40 calibration endpoints were H45/H41 in the earlier 66-hold map. Saved maps retain IDs and verify the video's hash. Fresh detection requires a new route review.

Pose CSV and skeleton video now share one inference pass. Alignment is saved after computation and reused only when video, pose CSV, and settings match. Restoring the example hold map does not replace freshly computed poses.

Section 10 samples valid frames for the clip length and exports current poses, alignment, holds, draft outputs, and saved annotations. Outdated inputs or unsaved endpoint edits stop export with an explanation. For evaluation, compare independent manual labels against predicted hold IDs and release/arrival times, retaining unknown cases. Draft intervals are not counts of moves, and visual review is not an accuracy benchmark.

## Output locations

Contact CSV files are saved in `outputs/tables/`, review images in `outputs/images/`,
and settings, validation summaries, annotations, and caches in `outputs/state/`.
Section 10 writes `outputs/exports/reviewed_contact_data.zip`, keeping those subfolders
inside the archive. Reviewed-map restoration also accepts older flat archives.
