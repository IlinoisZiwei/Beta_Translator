# CIS 5810 Final Project — Beta Translator

Ziwei Wang · Justin Zou · Pengfan Zhang

Beta Translator analyzes a climbing video and fits a target body to reviewed contact positions. The outputs include static 2D IK fits, a contact-aware continuous 2D candidate for a reviewed single-limb move, and a separate replay of the original tracked movement. Personalized movement recommendations remain a project goal.

## Run locally on macOS

1. Retrieve the actual Git LFS video/model assets with `git lfs install` and `git lfs pull`. Install Git LFS with `brew install git-lfs` if needed.
2. For first-time setup, run `bash setup.sh` from the repository. An already working `.venv` can be reused.
3. Open `Beta_Translator.ipynb` in VS Code/Jupyter with the repository as the working directory. Select `.venv/bin/python` as its kernel.
4. Run the notebook in order. Section 1 checks dependencies without installing packages or transferring files.
5. Run Section 6B for learned proposals, then use **Prefill automatic draft** in Section 7. Review BEFORE/AFTER, contact IDs and release/arrival. Confirm support continuity and click **Save annotations**. Section 7B optionally replays the previously confirmed H44 → H28 example.
6. Run Section 8 for a static target-body fit. To fit a sequence, enter release/arrival frames in Section 7, confirm the other three contacts remain fixed, save, and run Section 8B. Section 9 replays the original motion; Section 10 exports current results.

Use **Load annotations** after restarting. Changing an endpoint resets its contact labels. Automatic drafts never replace manual choices. After editing a helper `.py` file, restart the kernel to import the changes.

## Files

| File | Responsibility |
| --- | --- |
| `Beta_Translator.ipynb` | Settings, pipeline, review controls, and visual results |
| `beta_translator/inverse_kinematics.py` | `RATIOS`, body construction, forward kinematics, and IK |
| `beta_translator/pose_adapter.py` | Coordinate conversion and pose-vector construction |
| `beta_translator/contact_matching.py` | Hold-mask proximity and persistent contact intervals |
| `beta_translator/project_io.py` | Video I/O, reviewed maps, caches, and endpoint persistence |
| `beta_translator/motion_ik.py` | Contact schedule, whole-interval IK optimization, and residual checks |
| `beta_translator/motion_pipeline.py` | Saved-input validation, source/fit videos, tables, and export provenance |
| `data/reviewed_contact_data.zip` | Original reviewed example data; only its hold map is restored automatically |

The notebook imports shared functions instead of keeping second copies. The IK solver clips the initial guess, preserves bend bounds, uses Jacobian variable scaling, and reports nonconvergence.

## Settings and new videos

Section 1 contains the video path, output folder, pose thresholds, target height, wall scale, and `REFERENCE_FRAME`. `RATIOS` is defined once in the IK file. Override individual entries with measured segment length divided by height:

```python
RATIOS = {**RATIOS, 'forearm': measured_forearm_cm / TARGET_HEIGHT_CM}
```

For a new video, choose a separate output folder and set `USE_SAVED_HOLD_MAP = False`. Review the new detector output and fill in `NEW_ROUTE_HOLD_IDS` in Section 4. Fresh detection needs `models/experiment_config.yml` and `models/model_final.pth`. IDs belong to one map; never reuse an old selection without review.

All wall geometry uses `REFERENCE_FRAME` (default 0). It selects a wall image without trimming or renumbering the video. Changing it rebuilds the hold map and invalidates alignment/contact caches; select the route again because hold IDs change. The supplied clip's scale is approximately 1.798 pixels/cm, based on a measured 46 cm edge-to-edge gap. The physical endpoints are H44/H40 in the reviewed 65-hold map, formerly H45/H41. The default scale is enabled only for the original video's hash and reference frame 0. A new video or reference image requires its own measured scale, independent of body height.

## Saved outputs

Generated files are grouped by type under the configured output folder:

```text
outputs/
├── videos/   # pose_output.mp4, pose_output_h264.mp4, wall_motion_h264.mp4
├── tables/   # pose_coordinates.csv, limb_contacts.csv, contact_events.csv
├── images/   # wall_reference.jpg and contact review images
├── state/    # hold map, alignment cache, manifests, settings, saved annotations
└── exports/  # reviewed_contact_data.zip
```

`state/pose_manifest.json` validates pose reuse. `state/alignment_cache.npz` stores
the transforms and matching input hashes. `state/reviewed_hold_map.npz` preserves
reviewed hold IDs. `state/endpoint_annotations.json` contains manually saved frames,
contacts, moving limb, notes, body ratios, height, and wall scale.
Contact settings and the run summary are `state/contact_settings.json` and
`state/validation_summary.json`.

The notebook creates these folders automatically. Existing results were moved
without changing their contents. An older `images/contact_detail_samples.png` is
retained as a previous review result; current runs create `images/contact_samples.png`.
Old or mismatched caches are recomputed using the existing validation rules.

New ZIP exports preserve the `tables/`, `images/`, and `state/` layout inside the
bundle. Reviewed-map restoration accepts both this layout and the original flat
bundle in `data/`. Previously saved ZIPs remain unchanged.

The original video remains in `videos/`, model assets in `models/`, and the supplied
reference bundle in `data/`. The unused Colab example datasets in `sample_data/`
have been removed.

## Checks and limitations

Run `.venv/bin/python -m unittest discover -s tests -v` for focused helper checks.

Visibility is not measured joint accuracy. Draft contacts use wrists and foot-index points; IK approximates contact using wrists/ankles and hold centers. Unknown contacts and smears need further annotation before fitting. Uniform scale does not rectify perspective or recover body depth. Small geometric error does not establish grip, balance, strength, or physical feasibility.

See `docs/Contact_Matching.md` for review details. The older Codespaces guide is historical; this README describes the current local workflow.

## Sources

- [MediaPipe](https://github.com/google-ai-edge/mediapipe): pose tracking and adapted visualization code.
- [Indoor Climbing Hold and Route Segmentation](https://github.com/xiaoxiae/Indoor-Climbing-Hold-and-Route-Segmentation): pretrained detector workflow (GPL-3.0).
- [Model dataset](https://www.kaggle.com/datasets/tomasslama/indoor-climbing-gym-hold-segmentation): matching `Basic_Augmented_P2P3_DoubleTopK_NMS_2KIT` configuration and weights.
- [Detectron2](https://github.com/facebookresearch/detectron2): hold inference.
- [The Way Up](https://arxiv.org/html/2505.12854v1): proximity and temporal-filtering idea; this implementation uses masks and does not reproduce the full method.


## Continuous candidate in Section 8B

The current reviewed example uses frames 107–129, release 117 (first free frame),
arrival 128 (first contact on H28), and 165 cm with default body ratios.
Left hand: H44 on 107–116, free on 117–127, H28 on 128–129.
The user confirmed right hand H33, left foot H62, and right foot H65 remain in contact.

The optimizer fits all frames together, using the existing fixed-length body geometry,
visibility-weighted reference angles, contact penalties, and velocity/acceleration penalties.
It checks final contact errors separately; the tolerance is numerical, not a statement about
video accuracy. The free hand has a soft minimum-jerk straight-line guide between targets.
This guide is inferred, especially when hand tracking is occluded. Endpoints are jointly
refitted, so they can differ from the independently fitted static endpoints.

Outputs: `videos/motion_ik_h264.mp4`, `videos/motion_ik_slow_h264.mp4` (one-third speed),
`images/ik_motion_storyboard.png`, `tables/motion_ik_poses.csv`,
`state/motion_ik_trajectory.npz`, and `state/motion_ik_summary.json`.
The summary records input/code/output hashes and timing, residual, and continuity diagnostics.
Section 10 rejects stale motion artifacts. Rerun 8B after changing body, contact, timing,
pose, alignment, hold-map, or motion-code inputs.

This first model requires one moving limb, known endpoint holds, and three maintained supports.
Missing poses or unreliable alignment stop the fit. No forces, friction, balance, collision,
depth, or human success probability are modeled. Numerical convergence is not physical validation.
## Trained automatic contacts

The optional [automatic contact pipeline](docs/automatic_contacts.md) adds frozen
whole-body pose tracking and a trained temporal model for hold association and
release/arrival proposals. Open `Automatic_Contacts.ipynb` for inference and review.
The same model is integrated into Sections 6B–8B of the main `Beta_Translator.ipynb`.


## Main notebook: automatic prefill to reviewed continuous IK

Keep the notebook on its existing `.venv` kernel. Create a separate inference environment
once (already installed on the current machine):

```sh
.venv/bin/python -m venv .venv-contact
.venv-contact/bin/python -m pip install -r requirements-contact.txt
```

Section 6B checks video, hold map, alignment, pose-cache, model and prediction-table hashes.
It reuses matching predictions or calls `.venv-contact` to refresh them. The bundled
whole-body feature cache avoids downloading RTMW weights for the current video.
A new video needs pose extraction and its own output folder. Raw model downloads stay
under `outputs/automatic/model_cache/` and are ignored by Git.

1. **Section 6B:** load learned contact and movement proposals.
2. **Section 7:** choose a move and click **Prefill automatic draft**. This fills contacts,
   endpoints and timing, but always clears the support-confirmation checkbox.
3. Review the source frames, edit endpoint/contact/time fields, confirm the three supports,
   and save. Any contact or timing edit invalidates the previous support confirmation.
4. **Section 7B:** optional replay of the user's saved H44 → H28 reference. Its default
   `REPLAY_CONFIRMED_EXAMPLE = True` replaces the form with that reviewed example for the
   matching video. **Set it to `False` when keeping your own edits or reviewing another move.**
   The immutable reference is `data/H44_H28_confirmed_reference.json`; the replay audit is
   `outputs/automatic/demo/review_audit.json`.
5. **Section 8B:** optimize and embed normal-speed and one-third-speed continuous IK videos.

For this example, the model proposes BEFORE 105 / release 113 / arrival 136 / AFTER 143.
The saved human review corrects these to **107 / 117 / 128 / 129**, with left hand H44 → H28,
right hand H33, left foot H62 and right foot H65 maintained, at 165 cm with default proportions.
The model's timing ranges do not contain the reviewed release/arrival frames; they are
uncertainty estimates, not guaranteed bounds. The original suggestion remains in annotation
provenance alongside the corrected values. Section 7B replays an existing human confirmation;
it does not represent a fresh human review or model verification of support forces.

The automatic, unreviewed IK gate remains disabled in the deployed model. The demonstrated
path is model suggestions → human-reviewed annotation → continuous 2D IK. The existing
geometric contact tables are separate and are not overwritten by learned inference.


## Trial2: new-wall analysis and hold-map repair

Open `Trial2_Analysis.ipynb` using the existing `.venv` kernel. The 976-frame,
1920 x 1080, 30 FPS analysis video is `videos/trial2.mp4`; outputs remain isolated
under `outputs/trial2/`. The source MOV is preserved outside the repository.
The import manifest and nearest-source-frame table retain source timing. The
8-bit conversion does not perform complete HDR tone mapping.

The wall map uses reference-frame-450 coordinates (15 seconds), with observed
pixels from frames 0, 225 and 975 filling the masked climber. Two frames alone
left 10,022 unresolved mask pixels. Four views cover that mask without generative
fill; this does not prove exact multi-plane geometry. The original reference
image remains the alignment target. Motion frame IDs are unchanged.

The original detector stopped at 100 candidates. The revised process keeps the
same weights and 0.70 threshold, raises the cap to 400, adds six overlapping
768-pixel tiles and suppresses duplicate masks. It produces 298 candidates.
Assistant review excludes 15 clear non-holds and withholds one uncertain object,
leaving 282 candidates before route selection. The current route is selected in Section 4B and saved with its existing IDs
in the reviewed map.

| Hold-map variant | Candidates | Fixed priority-point coverage |
| --- | ---: | ---: |
| Original frame 450, cap 100 | 100 | 6/15 (40.0%) |
| Frame 450, cap 400 | 179 | 13/15 (86.7%) |
| Observed-pixel reconstruction | 171 | 14/15 (93.3%) |
| Reconstruction plus overlapping tiles | 298 | 15/15 (100% of this sample) |

The audit is an assistant-reviewed development sample, not whole-wall AP or an
independent test. One initial point was corrected from a route marker to a
non-hold and removed from every denominator. The initial draft is preserved.
A random raw-candidate spot check found 36 likely holds, 3 clear false positives
and 1 uncertain object among 40 candidates. Candidate count is not accuracy.

The pre-selection pipeline rerun reused the 972-frame MediaPipe pose table, 976-frame RTMW
cache and verified alignment. Learned contact confidence coverage changed from
56.94% to 42.03%; movement proposals changed from 9 to 26.
No trial2 contact ground truth is available, so no contact-accuracy improvement
is claimed. All proposals require review; automatic unreviewed IK stays disabled.

See `outputs/trial2/reports/hold_map_evaluation.html` for the evaluation, source
attribution, technical project presentation and proposed grip/local-planning
extensions. Inspect `images/reconstructed_wall.png`, `images/multiframe_hold_map.png`,
`images/priority_holds_detail.png` and `videos/automatic_contacts_h264.mp4` under
the trial2 output directory. `wall_motion_h264.mp4` is observed replay, not IK.
The prior frame-450 run summary is in `reference_runs/frame_0450_single_frame_summary.json`.

To reproduce the fixed audit or build separate new candidates:

```sh
.venv/bin/python scripts/evaluate_hold_maps.py --artifacts outputs/trial2/hold_mapping
.venv/bin/python scripts/build_multiframe_hold_map.py --video videos/trial2.mp4 --outputs outputs/trial2 --destination outputs/trial2/hold_mapping/rebuild --reference 450 --frames 0 225 450 975
```

The reconstruction helper checks video/cache identity and records pixel sources.
The candidate builder does not overwrite a reviewed map. New detection IDs and
review exclusions cannot be carried into unrelated maps. Continuous trial2 IK
still needs independently measured wall scale, suitable panel geometry and
confirmed endpoint/timing/support annotations. The 165 cm target and default
ratios are assumptions, not the filmed climber's measurements. Forty-nine helper
tests pass; the original trial1 notebook, manual labels and IK remain intact.

### Current route evaluation

Section 4 shows every stored candidate. Edit `USER_ROUTE_HOLD_IDS` in Section 4B,
run that cell and save the notebook. Rerun Section 6 onward to update contacts.
The final evaluation cell reads the saved route and verified prediction files,
then refreshes route images, point-coverage tables, contact statistics, the HTML
report and the export bundle. Stale predictions are rejected.

The current-route coverage check is separate from the historical fixed 15-point
development audit. Extra route points are derived from selected masks when no
historical source-review point exists. Selection from the final detections makes
this a coverage check, not independent detection recall or segmentation AP.

Current values are in `outputs/trial2/hold_mapping/current_route_evaluation.json`
and `outputs/trial2/reports/hold_map_evaluation.html`. The source is
`scripts/update_trial2_report.py`; no model weights are changed.


### Trial2 identity tracking

The original full-frame MediaPipe pass switched to a foreground passerby in
frames 184–236. Section 2 now preserves that raw pass, then verifies the target
before Section 3's preview. It uses existing RTMW tracking boxes, torso position
and scale, and frame-to-frame motion in wall coordinates. Rejected poses are
re-detected inside the climber crop. `num_poses=1` alone does not select a stable
person identity.

The first raw pose/wall-alignment pass is retained as `pose_coordinates_raw.csv`
and `identity_anchor_alignment.npz`. The bootstrap alignment is created if absent,
using the raw pass; Section 5 separately recomputes alignment after pose correction.
RTMW features are reused, or extracted with the contact environment if missing.
The frozen whole-body cache and trained model weights are not modified.

`pose_coordinates.csv` contains observed target poses only.
`pose_display_coordinates.csv` may also contain orange, display-only estimates
for short gaps (up to two seconds) with compatible visible endpoints. Interpolation
is performed in wall coordinates and never enters contact matching or IK.
The RTMW observation guard excludes low-identity-confidence frames from learned
contact inference, including a two-frame buffer. Raw pose scores are not calibrated
probabilities. Long gaps and incompatible endpoints remain missing.

Inspect the final **Climber identity during a passing-person occlusion** section
for before/after examples, diagnostics and a comparison clip. Results are stored
in `outputs/trial2/state/identity_evaluation.json` and
`outputs/trial2/videos/identity_comparison_h264.mp4`. The HTML report includes the
same case. This is a development-case fix, not a general multi-person accuracy
benchmark; target initialization, joint visibility and wall geometry still matter.

Implementation: `beta_translator/identity_tracking.py` and
`scripts/report_identity_tracking.py`. Run the trial2 notebook from Section 2 onward
to update dependent caches and reports. The original trial1 workflow has no identity
guard and retains its behavior.


### Trial2 geometry assumptions

Section 1 now separates the filmed climber's user-reported **173 cm** height from
the **165 cm** target body. Adjacent-hole spacing is **15 cm by user assumption**,
not a measured value or a Movement-wide specification. Section 4C uses twelve
visible adjacent-hole intervals on original frame 450 to calculate local,
directional pixel scales. The editable observations and video hash are in
`data/trial2_geometry_reference.json`. F1 and F2 are floor seams whose identity and
parallel relation were confirmed by the user. Their image-line intersection gives
one vanishing point; it does not determine the floor horizon, wall tilt or camera
intrinsics. Default limb proportions are priors, not measured proportions.

These inputs are saved in `outputs/trial2/state/geometry_calibration.json` and
`geometry_observations.json`, with a local-scale table, annotated images and
`reports/geometry_calibration.html`. Section 8 validates the current geometry
context; Section 10 exports it; the final geometry report explains the limits.
Spacing or height changes require rerunning Section 4C before review/export.
Changing the video or reference frame rejects the old image coordinates.

No local pixel scale is assigned to the whole-wall `PX_PER_CM`; multi-plane IK
remains gated. A second independent floor direction plus camera calibration, and
known panel-grid geometry, are one route to estimating wall/floor normals and
angles. Current pose/contact predictions remain in image coordinates; no model
weights or measured contact accuracy change in this step.


### Trial2 direct automatic IK (user enabled)

`AUTOMATIC_IK_ENABLED = True` in Trial2 Section 1 enables an experimental run-level
bridge. Sections 7B and 8B consume decoded `candidate_hold` states and native-frame
release/arrival predictions without a manual widget confirmation. The user's
acceptance of the current prediction files is recorded with their hashes in
`outputs/trial2/state/automatic_ik_user_review.json`. Later changed predictions are
still processed in experimental mode but are not described as user-reviewed.
The trained package's validation gate and published evaluation are unchanged.

`beta_translator/direct_ik.py` retains the current solver's mathematical contract:
one moving limb, three continuous fixed hold contacts, observed poses, matching
alignment, positive wall scale and a suitable single-panel segment. It does not
replace `free`, `wall_contact`, or changing supports with invented hold IDs.
Each compatible move is passed automatically to `run_motion_analysis` in its own
`outputs/trial2/automatic/ik/move_XXX/` directory. Missing geometry, unsupported
schedules and numerical failures are listed in `state/direct_ik_plan.json`.
A zero-video result is reported explicitly, rather than displaying Section 9's
original-person replay as a retargeted result. Section 10 exports the run report.

The current trial2 audit contains 21 proposals and no interval supported by the
three-fixed-support solver. Several otherwise clear hand moves have a decoded
free foot. Wall scale is also unset and the clip spans multiple wall planes.
Enabling direct predictions removes the statistical/manual-review gate; supporting
these moves requires a solver with varying contacts and suitable wall geometry.


## Trial 3: compact automatic run

Open `Trial3_Analysis.ipynb` with the repository `.venv` kernel and run its seven sections: video/body settings, target tracking, wall alignment and hold detection, learned contacts, metric geometry, direct continuous IK, and results. The original input is `/Users/ziwei/Downloads/trial3.mov`; the upright 30 FPS analysis copy is `videos/trial3.mp4`. `outputs/trial3/tables/source_frame_map.csv` records exactly which original frame supplied each analysis frame.

Section 1 sets the target body (currently 190 cm with default proportions). Source-climber height and wall calibration are separate inputs and are not copied from Trial 2. The workflow selects a sampled reference view, fills occlusion from observed video pixels, and retains all tiled-detector candidates for contact association. The legacy `reviewed_hold_map.npz` filename is required by the existing interfaces; these automatic candidates have **not** been manually reviewed.

The existing RTMW/temporal ensemble is reused without training. MediaPipe coordinates and identity checks remain because the current IK implementation needs their body representation. The geometric contact baseline, manual annotation form, repeated replays, historical ablations and default ZIP export are omitted from the main notebook. Stage artifacts are validated before reuse, so rerunning resumes expensive completed work.

`AUTOMATIC_IK_ENABLED = True` bypasses the statistical acceptance gate; it does not invent support contacts or wall dimensions. The solver still requires one moving limb, three fixed hold supports, complete observed poses/alignment, and suitable metric geometry. Section 6 reports supported moves and renders only valid solves. Section 7's contact preview always shows the **original climber**, not the target body. No contact accuracy or climbing-success probability is claimed without trial-specific labels.

Saved summaries are `outputs/trial3/state/run_summary.json` and `direct_ik_plan.json`. `approved_cleanup.json` records the 15 obsolete files moved to macOS Trash with user approval; the active Trial 2 notebook and map were preserved.

Trial 3 execution: 503 analysis frames, 503 observed body poses, 503 valid wall alignments, 53 automatic hold candidates, and 12 movement proposals. Three proposals satisfy the current solver's contact-schedule checks; zero IK videos were generated because wall scale and suitable-panel geometry are not supplied. Model-threshold coverage is about 76.2%, **not measured accuracy**.

The initial sparse reconstruction left a gray unknown region that the hold detector misclassified. Denser observed-frame sampling and a resolution-appropriate person-mask margin recovered the visible white hold; an observed-pixel filter now prevents unknown reconstruction areas from becoming candidates. This is an inspected failure repair, not a quantified detector-accuracy evaluation.
