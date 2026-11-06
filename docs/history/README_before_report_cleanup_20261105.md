# Historical README — before report cleanup, November 5, 2026

This appendix preserves the previous project overview and experiment record. Section numbers and phrases such as “current” or “new” retain their historical meaning. Use the [current README](../../README.md) and [notebook guide](../complete_trial_notebooks.md) for the active report layout and controls. Relative Markdown links below have been adjusted for this appendix location.

**Archive update — November 5, 2026:** The approved second cleanup moved the old Codespaces guide and read-only `cleanup.sh` inventory into this history folder, and archived two duplicate Trial1 runs outside the repository. The old `cleanup.sh` command below is historical and is no longer an active command. For Trial2, the removed `images/reconstructed_wall.png` copy is preserved at `outputs/trial2/hold_mapping/reconstruction/reconstructed_wall.png`. See the [cleanup and archive map](cleanup_20261105.md) for exact paths and recovery instructions.

---

# CIS 5810 Final Project — Beta Translator

The shared model now includes an anatomical **hip–knee–ankle/subtalar chain**. The three analysis notebooks default to **175 cm** for the new comparison; all earlier 180 cm results remain available. See [lower-limb implementation and verification](../../docs/anatomical_lower_limbs.md).

## Shared body model for all three trials

Open `Trial1_Analysis.ipynb`, `Trial2_Analysis.ipynb` or `Trial3_Analysis.ipynb` with the **Python (Beta Translator)** kernel. Each now restores the complete evidence-to-anatomy workflow: wall and hold review, source skeleton, contact/timing review, historical IK/gravity/support and 3D experiments, followed by the current 175 cm shared-model comparison. Run All displays saved results; fitting is opt-in. The reusable entry point is `beta_translator.body_model.BodyModel`; shared bounds and coverage are in `beta_translator/body_constraints.py`. Trial-specific inputs and wall hypotheses live in `configs/trials/`.

```sh
source .venv/bin/activate
python scripts/run_body_model.py --trial all --height-cm 175 --frames verification
```

Use `--trial trial2 --frames all` to request a full sequence, or change `--height-cm`. New runs preserve earlier outputs. The complete physiological specification is **not fully validated**: the shared model includes anatomical chains and personalized collision, torque/fatigue and dynamics proxies, while contact geometry, individual capabilities and camera/wall parameters remain uncalibrated. See [the shared-model guide](../../docs/shared_body_model.md).

The [history archive](../../notebooks/history/pre_shared_body_20261104.zip) preserves exact copies of the previous three notebooks and their detailed review workflows. The active notebooks restore those stages alongside the current model. Interactive route/contact exports go to new `outputs/notebook_reviews/<trial>/` directories and do not silently alter the saved model inputs. Exact previous 175 cm notebooks are also [archived](../../notebooks/history/pre_complete_workflow_20261105.zip). Earlier result links retain their recorded heights. `Beta_Translator.ipynb` is unchanged.


Ziwei Wang · Justin Zou · Pengfan Zhang

Beta Translator analyzes climbing videos and fits a target body to reviewed or model-predicted contacts. It includes static 2D IK, single-move IK, and a continuous 180 cm Trial 3 adaptation with explicit constraint and uncertainty diagnostics. Personalized movement recommendations remain a project goal.

## Anny human avatar

The final section of each trial analysis now displays its **full native sequence** (Trial1: 0–416; Trial2: 0–975; Trial3: 0–502), using a **175 cm / 176 cm arm-span** Anny human mesh alongside the existing skeleton, plus an overlay on the source video. The shared adapter preserves anatomical rotation frames and checks actual palm/sole surface anchors. Fingers are a fixed display pose; source failures and mesh-wall penetration remain marked. Original videos and results stay intact.

Run `bash setup.sh` to update the **same environment and kernel**, then use `.venv/bin/python scripts/run_avatar.py --trial all` to reproduce the three saved verification adapters. Any completed short or full shared-body run can be passed with `--source-run`. See [the avatar guide](../../docs/anny_avatar.md) and [saved cases](../../configs/avatar_results.json). No retraining or Blender installation is needed.

## Notebook entry points

All notebooks use the same **Python (Beta Translator)** kernel from `.venv/bin/python`.

| Notebook | Purpose and current result |
| --- | --- |
| [Beta_Translator.ipynb](../../Beta_Translator.ipynb) | Original Trial 1 workflow: reviewed contacts, static IK, and the confirmed H44 → H28 single-move fit. Default target: 165 cm. |
| [Trial2_Analysis.ipynb](../../Trial2_Analysis.ipynb) | Complete workflow with identity tracking, wall/route/contact review, historical 190 cm 2D geometry blockers, 180 cm joined-wall/arm/contact experiments, and current 175 cm anatomy. |
| [Trial3_Analysis.ipynb](../../Trial3_Analysis.ipynb) | Complete workflow with wall reconstruction, route/contact review, calibration, 180 cm single/full-sequence IK, gravity/support, tilt/depth/contact-region studies, and current 175 cm anatomy. |
| [Trial1_Analysis.ipynb](../../Trial1_Analysis.ipynb) | Complete workflow with source tracking, hold/contact review, 165 cm single-move example, historical 180 cm gravity/ground/refinement/technique plans, and current 175 cm anatomy. |
| [Automatic_Contacts.ipynb](../../Automatic_Contacts.ipynb) | Trained contact-model inference and reports. Uses the same project environment; see [the inference guide](../../docs/automatic_contacts.md). |

### Which demo to open

| Output | What it shows | Saved Trial 3 demo |
| --- | --- | --- |
| Original climber replay | Observed motion with tracked joints/contact predictions; target height does not change the body. | [Contact preview](../../outputs/trial3/videos/automatic_contacts_h264.mp4) |
| Single-move IK | A fitted target body for one eligible move, accepted only after the strict contact-residual checks. | [180 cm H017 → H007](../../outputs/trial3/videos/trial3_target_180cm_ik_demo.mp4), frames 296–331, slowed with endpoint pauses |
| Full-sequence adaptation | All 503 target-body poses optimized together, allowing contact changes and displaying unmet constraints or uncertainty. | [180 cm full-sequence video](../../outputs/trial3/full_sequence/target_180cm_250e15b3ee81/trial3_target_180cm_full_sequence.mp4), frames 0–502 at 30 FPS, about 16.77 s |

Notebook videos use repository-relative file references; images and numerical results remain saved in the notebooks. Keep the `outputs/` video files alongside the notebooks and retrieve Git LFS assets before viewing. If a notebook preview does not render a video, open its MP4 link directly. See [full-sequence assumptions and diagnostics](../../docs/trial3_full_sequence.md) before interpreting the adaptation.

**Body model reference:** [Body Model and Constraint Specification](../../docs/body_constraint_specification.md) documents the current constraints and proposed anatomical chain, pose-dependent ROM, hand/foot orientation, collision, strength, fatigue and dynamic balance. Implemented checks and future capabilities are distinguished explicitly.

### Trial2 Stage 1: consistent hand/foot contacts (180 cm)

[Trial2_Analysis.ipynb](../../Trial2_Analysis.ipynb), the **Stage 1** sections, retain a source-checked shoulder–elbow–forearm–wrist rotation chain and one shared hand/foot patch definition for fitting, support checks and rendering. It compares the abnormal right-elbow segment, frames **550–595**, against an equal-budget control. Foot orientation remains conditional; ankle ROM and full dynamic feasibility are not established. See [Stage 1 details and commands](../../docs/trial2_stage1_contacts.md). Earlier full-length results remain in the preceding historical sections.

### Trial2 anatomical-arm update (180 cm)

Open the **Anatomical arms** sections of Trial2_Analysis.ipynb for shoulder/elbow/forearm/wrist constraints, three anomaly clips, candidate rankings, and the full fixed-camera before/after video. The earlier 3D runs remain available. The new body is a reduced MoBL-inspired chain; hand orientation and wall geometry remain uncertain, and unresolved failures are displayed. The current full run is **diagnostic-only**: joint bounds and continuity checks pass, but contact/support failures remain and neither candidate converged. See the full anatomical-arm comparison section for the before/after tradeoff.

Run `bash setup.sh` to install the added OSQP dependency into the same environment. Recompute with `.venv/bin/python scripts/run_trial2_anatomy.py --stage all` (short clips are checked first); use `--stage clips` for only the short experiments. The notebook defaults to reading the saved results. See [the constraint definitions and source references](../../docs/trial2_anatomical_constraints.md).

## Run locally on macOS

1. Clone/pull `main`, or download **Code → Download ZIP** and unzip the whole repository.
2. Double-click **`Start.command`** in that folder. The first run installs Python 3.12, FFmpeg, one `.venv`, all Python dependencies, and the required model/video assets. Later launches reuse the installation and open JupyterLab. If Finder does not launch it, open a terminal in the folder and run `bash Start.command`.
3. Open a notebook and select **Python (Beta Translator)** if asked. The merged Trial1 and Trial3 notebooks, Trial2, and Automatic Contacts support **Run → Run All Cells** with the bundled inputs. For the main notebook’s review step, see below. Keep the terminal open while using JupyterLab. In VS Code, choose **Python Environments → `.venv/bin/python`** for every notebook.

This setup is tested on an Apple Silicon Mac. First-time Homebrew/Command Line Tools installation may require macOS prompts and your password. Allow several minutes and several GB of disk space; an internet connection is needed for the first installation. No second environment or manual model copying is needed. Both Git clones and GitHub ZIP downloads are supported; setup restores missing Git LFS assets directly with checksum verification.

To install without opening JupyterLab, run `bash setup.sh`. With Python 3.12 and FFmpeg already on PATH, `bash setup.sh --skip-system` skips Homebrew. Setup never runs training or overwrites existing generated results. Existing `.venv-contact` folders can be left alone; they are no longer used. Do not copy a virtual environment between computers.

**The main notebook has a manual review step.** Run through Section 7, click **Load annotations** to use the saved review (or review your own contacts/timing and click **Save annotations**), then continue through Sections 8–10. Running all cells without loading/saving the form intentionally stops at Section 8B to protect unsaved review choices. `REPLAY_CONFIRMED_EXAMPLE = False` preserves your form choices; set it to `True` to replay the bundled, previously reviewed H44 → H28 example. Uncertain contacts and geometry failures remain visible; installing dependencies does not make every move feasible.

The three bundled trial inputs are:

| Trial | Required analysis video | Frames | Duration at 30 FPS |
| --- | --- | ---: | ---: |
| Trial 1 / main notebook / Automatic Contacts | `videos/trial1.mp4` | 417 | 13.90 s |
| Trial 2 | `videos/trial2.mp4` | 976 | 32.53 s |
| Trial 3 / Full Sequence | `videos/trial3.mp4` | 503 | 16.77 s |

These MP4s are tracked with Git LFS; `Start.command` restores their actual contents even if a ZIP/clone initially contains only pointer files. Keep the complete repository layout: the notebooks also need `data/`, `models/`, and `outputs/`. Saved demo videos live under `outputs/` and are downloaded by the same setup. Sending only an `.ipynb` is insufficient.

**No original MOV from the author's Downloads folder is needed to run the bundled trials.** The complete notebooks reuse the bundled MP4s and verified source evidence. Old printed logs can still show `/Users/ziwei/...`; they are historical output. Legacy cache paths are resolved against the current checkout, not the author's computer. For a different clip or to redo the original MOV-to-MP4 conversion, supply your own source video and use fresh video/output paths; its holds and geometry need their own review.

Training scripts use the same `.venv`. Reproducing training additionally requires the public dataset; follow [the training commands](../../docs/automatic_contacts.md#reproduce). Demo setup downloads the pretrained models, not the training dataset, and does not start a training job.

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
## Complete trial notebooks

The active analysis notebooks contain the source evidence, interactive route/contact review, historical fitted experiments and current shared anatomy in one ordered workflow. Run All displays saved results without rewriting inputs. Use `RUN_VERIFICATION` or `RUN_FULL_SEQUENCE` for new shared-model fitting; `HEIGHT_CM` defaults to 175. Historical results keep their recorded heights.

| Notebook | Common evidence/review | Historical experiments | Current shared model |
| --- | --- | --- | --- |
| Trial1 | Sections 1–8 | Sections 9–13: single move, gravity/support, ground, refinement, techniques | Sections 14–19 |
| Trial2 | Sections 1–8 | Sections 9–17: geometry gate, wall/identity evaluation, joined panels, arm and contact comparisons | Sections 18–23 |
| Trial3 | Sections 1–8 | Sections 9–15: single/full IK, gravity/support, tilt/depth and contact regions | Sections 16–21 |

The final section of each notebook indexes saved outputs, source evidence and exact previous notebook snapshots. Review exports retain map IDs and video identity; they do not automatically replace canonical inputs or retarget historical videos. See [complete workflow instructions](../../docs/complete_trial_notebooks.md).

The command-line examples below reproduce the original experiments. Legacy section numbers and settings in later development notes refer to the [archived detailed notebooks](../../notebooks/history/pre_shared_body_20261104.zip), not the current numbering. All historical outputs remain dependencies and must be retained.

## Trial1 continuous support optimization

The historical gravity/support section of `Trial1_Analysis.ipynb` displays the saved 180 cm result. The command below recomputes all 417 frames and saves a
gravity baseline and optimized video, and provides a synchronized original/result
review page. Trial1's existing uniform image scale and image-up gravity assumption
remain flagged. Its saved H44 → H28 human reference is compared separately without
changing predictions or annotations.

```bash
.venv/bin/python scripts/run_trial1_support_optimization.py --height-cm 180 --mass-kg 70
```

See [Trial1 geometry, inputs and outputs](../../docs/trial1_support_optimization.md).

Saved [optimized video](../../outputs/trial1/support_optimized/target_180cm_77aeef37c158/trial1_support_optimized_180cm_full_sequence.mp4) · [synchronized review](../../outputs/trial1/support_optimized/target_180cm_77aeef37c158/review.html).


## Trial3 conditional 3D experiment

The conditional tilt/depth section of `Trial3_Analysis.ipynb` displays a planar 3D wall and fixed-length
180 cm skeleton, comparing 0/15/30-degree tilt hypotheses and 10/25/40 cm soft
hip-depth priors. It exports GLB meshes, an offline orbit viewer, short comparison
videos and metrics. This is not a measured wall/body scan. Tilt is not identified
by matching the photograph, and fixed-center contact conflicts remain flagged.
Reviewing whole hold regions removes those particular pairwise impossibility
certificates, but does not prove a feasible full-body pose. A continued 36-frame
solve converges numerically while retaining conditional support failures.

```sh
.venv/bin/python scripts/run_trial3_3d_experiment.py
```

The existing environment is sufficient; no additional model is required. See the
[method and commands](../../docs/trial3_spatial_experiment.md) and
[measured results and iteration review](../../docs/trial3_spatial_results.md).
Depth Anything V2 Small is an optional, separately cached relative-depth probe,
not metric ground truth or a prerequisite for running the notebooks.
Start with the [converged angle comparison](../../outputs/trial3/spatial_experiments/refined_angles_180cm/refined_angles.mp4)
and its [offline 3D viewer](../../outputs/trial3/spatial_experiments/refined_angles_180cm/viewer.html).

## Trial3 support-aware optimization

The historical support-optimized section of `Trial3_Analysis.ipynb` displays the saved result. The command below adjusts the whole trajectory to
reduce contact, support-balance and pre-release unloading failures. Fixed segment
lengths and the same audit thresholds are retained; unresolved constraints and
uncertainties remain visible. Old results are preserved.

```bash
.venv/bin/python scripts/run_trial3_support_optimization.py --height-cm 180 --mass-kg 70
```

See [method, comparison metrics and limitations](../../docs/trial3_support_optimization.md).

Saved [180 cm optimized continuous video](../../outputs/trial3/support_optimized/target_180cm_2f29049b96dd/trial3_support_optimized_180cm_full_sequence.mp4).

## Trial3 continuous gravity IK

The historical gravity section of `Trial3_Analysis.ipynb` displays the saved 180 cm result. A mass of 70 kg is an
explicit demonstration assumption. Body proportions, foot friction and force-component
limits are configurable. Or run from the repository:

```bash
.venv/bin/python scripts/run_trial3_gravity_sequence.py --height-cm 180 --mass-kg 70
```

The solver jointly fits all 503 poses and contact forces with temporal penalties;
it does not stitch old clips. The 30 FPS video shows the target skeleton, center
of mass, inferred force arrows, per-frame checks and uncertainty. Gravity is a
quasi-static 2D constraint: inertia, depth, grip strength and collision are absent.
Balanced forces at a missed contact do not count as a valid fit. Free-foot support
and extrapolated geometry remain flagged. Existing results are preserved under
their original paths; new gravity outputs live under `outputs/trial3/gravity_sequence/`.
Previous gravity-only demonstration (preserved): [180 cm continuous gravity video](../../outputs/trial3/gravity_sequence/target_180cm_71a8d2f186dc/trial3_gravity_180cm_full_sequence.mp4).
New support-audit demonstration: [180 cm, frames 0–502](../../outputs/trial3/gravity_sequence/target_180cm_c1d9a4e6c5db/trial3_gravity_180cm_full_sequence.mp4).
The post-fit checks exclude missed holds, compare eligible and conditional support,
report force-component reserve with COM sensitivity, and remove departing limbs before
predicted releases. Failed and uncertain results remain visible. They do not replan
the motion or validate dynamic balance. Configure `AUDIT_SETTINGS` in the notebook.
See [method and limitations](../../docs/trial3_gravity_sequence.md).

## Gravity IK comparison in Section 8C

In `Beta_Translator.ipynb`, run Section 1 and then Section 8C to compare ordinary and
gravity IK at the confirmed H44 → H28 endpoints (frames 107 and 129). Set
`COMPARISON_HEIGHT_CM` to the target height and `COMPARISON_MASS_KG` to an assumed or
known mass; the saved demonstration uses 165 cm and an assumed 70 kg. Both methods
use the same body proportions and contact targets. Ordinary IK receives a separate
fixed-pose force check; gravity IK fits pose and forces jointly. Both may already pass.

The figure shows the skeleton, center of mass, gravity and inferred contact forces.
The table separates contact tolerance from force/moment balance and reports failures.
Results, full metrics, force vectors and provenance are kept in separate, versioned
`outputs/gravity_comparison/` directories. Existing annotations and videos are preserved.

This is a static image-plane approximation with simplified hand/foot force limits,
not physical climbing validation. Default limits scale with body weight, so changing
mass alone does not change the pose. Section 8B and Trial3 full-sequence animations
still use their existing geometric solvers; whole-video gravity requires a separate
continuous implementation with contact changes and temporal constraints.

## Trained automatic contacts

The optional [automatic contact pipeline](../../docs/automatic_contacts.md) adds frozen
whole-body pose tracking and a trained temporal model for hold association and
release/arrival proposals. Open `Automatic_Contacts.ipynb` for inference and review.
The same model is integrated into Sections 6B–8B of the main `Beta_Translator.ipynb`.


## Main notebook: automatic prefill to reviewed continuous IK

Run `Start.command` (or `bash setup.sh`) once. The notebooks and their inference subprocesses all use the same `.venv` interpreter.

Section 6B checks video, hold map, alignment, pose-cache, model and prediction-table hashes.
It reuses matching predictions or calls the current kernel’s Python interpreter to refresh them. The bundled
whole-body feature cache avoids extracting poses again for the current video.
A new video needs pose extraction and its own output folder. Raw model downloads stay
under `training_data/model_cache/` and are ignored by Git.

1. **Section 6B:** load learned contact and movement proposals.
2. **Section 7:** choose a move and click **Prefill automatic draft**. This fills contacts,
   endpoints and timing, but always clears the support-confirmation checkbox.
3. Review the source frames, edit endpoint/contact/time fields, confirm the three supports,
   and save. Any contact or timing edit invalidates the previous support confirmation.
4. **Section 7B:** optional replay of the user's saved H44 → H28 reference. Setting
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
confirmed endpoint/timing/support annotations. The current default 190 cm target
and default ratios are assumptions, not the filmed climber's measurements. Forty-nine helper
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
RTMW features are reused, or extracted with the shared environment if missing.
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
the default **190 cm** target body. Loading saved annotations may restore their
saved target height and proportions. Adjacent-hole spacing is **15 cm by user assumption**,
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


## Trial 3: unified analysis and adaptation

Open `Trial3_Analysis.ipynb` with **Python (Beta Translator)** / `.venv/bin/python`. Set the target body in Section 1, then **Restart Kernel and Run All**. The merged notebook has these sections:

| Section | Contents |
| --- | --- |
| 1. Setup & Target Body | Shared height, body proportions, mass, friction, solver and audit settings. |
| 2. Video & Climber Tracking | Bundled video, pose extraction and identity checks. |
| 3. Wall Reconstruction & Route Holds | Wall alignment, hold map and route selection. |
| 4. Contact Prediction & Move Events | Decoded contacts, releases/arrivals and original-person replay. |
| 5. Wall Calibration & Geometry Checks | White-wall metric grid and coverage limitations. |
| 6. Single-Move IK | Strict individual-move fits and target-body demo. |
| 7. Full-Sequence Geometric IK | Joint geometric fit across frames 0–502. |
| 8. Gravity IK & Support Checks | Pose/force fit with independent contact, reserve and unloading audits. |
| 9. Support-Optimized Full Sequence | Pose adjustment using support objectives and the same audits. |
| 10. Results Comparison | Shared metrics and links for all three complete trajectories; unavailable audits are N/A. |
| 11. Remaining Failures & Saved Outputs | Problem frames, release checks, solver status and output index. |
| 12. Conditional 3D Tilt & Depth Experiment | Eight keyframes, two short native-frame clips, 3D meshes/viewer and independent support audits under explicitly unmeasured geometry. |
| 13. Contact Regions, Hand/Foot Offsets & Complete 3D Plans | Five complete contact-model candidates plus native-frame refinement, ranking reasons, region/offset evidence, full videos and unresolved-constraint checks. |

The three former standalone Trial3 notebooks have been consolidated into this entry point. Earlier videos and result directories are retained, and the command-line scripts remain available. Full-sequence caches require matching inputs, body/physics/settings, code provenance and artifact hashes. Changed code or settings trigger new run directories; the first complete run can be lengthy. Section 9 verifies that it uses Section 8's gravity baseline instead of comparing unrelated runs. Trial3's free-foot support remains unresolved; Trial1 ground-support/refinement is not silently applied here.

 The default `SOURCE_VIDEO = None` reuses the bundled video; `videos/trial3.mp4` contains 503 upright analysis frames at 30 FPS. `outputs/trial3/tables/source_frame_map.csv` records the exact source-frame mapping.

Section 1 now sets the target to **180 cm with default proportions**. Source height and wall geometry are independent and are not inherited from Trial 2. The existing RTMW/contact ensemble is reused without retraining; MediaPipe and identity checks supply the IK reference poses. Verified caches avoid repeating expensive extraction.

Section 3 retains 53 detector candidates plus H054, the user-identified small white hold below H032. Its approximate manual outline is saved in `data/trial3_manual_holds.json`; original IDs and masks remain unchanged. The selected route is H001, H007, H012, H017, H021, H032, H045, H050, H053, and H054. Edit `USER_ROUTE_HOLD_IDS` to change membership, not movement order. IDs are tied to their candidate-map run. The legacy `reviewed_hold_map.npz` filename does not imply that every detection or contact was manually reviewed.

The user reports that both walls are vertical, the left door frame has vertical parallel sides and horizontal parallel top/bottom edges, and a floor line is parallel to the ceiling plane. The user also confirmed a **15 cm orthogonal hole grid**. Section 5 calibrates only the white-hold wall using `data/trial3_wall_grid.json` and `beta_translator/plane_calibration.py`. The current fit uses 301 holes, with 72 reserved from the final fit; reserved RMS reprojection error is 0.365 px (0.229 cm conditional on the grid dimensions). This is internal grid consistency, not physical measurement accuracy. H053 lies outside observed-hole coverage and is excluded from the strict single-move IK workflow. The full-sequence stages use it only as a reduced-weight, explicitly flagged extrapolated target. In this planar workflow, the second wall, relative wall angle, camera intrinsics, body depth, and hold depth remain unmodeled. Section 12 adds explicit camera/body-depth hypotheses for selected clips; they remain unmeasured. See the [OpenCV homography tutorial](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) for the planar mapping method.

Section 6 passes the full homography into `pose_adapter.py` and `motion_pipeline.py`; no uniform raw-image pixel scale is substituted. `plane_ik.py` verifies calibration provenance, reference identity, and contact coverage. A conservative reach bound detects contact separations beyond the configured body's span. Converged trajectories must pass the existing 0.1 cm numerical contact tolerance before export. This tolerance is not a claim about real-world contact accuracy.

The saved 180 cm run has 13 automatic movement proposals and seven contact-compatible schedules. One **right-hand H017 -> H007** move (frames 296-331) is generated; four exceed the model's body-span bounds, one uses H053 outside calibration coverage, and one reaches the optimizer iteration limit. The generated trajectory has 0.0051 cm maximum numerical contact residual. Numerical solver failure is not proof of physical impossibility, and a successful solve does not predict climbing success. This single-move model omits forces, balance, friction, collisions, and out-of-plane motion; Sections 8–9 add quasi-static force/support modeling.

The single-move target-only demo is [trial3_target_180cm_ik_demo.mp4](../../outputs/trial3/videos/trial3_target_180cm_ik_demo.mp4). It replays the fitted 1.2-second clip at 0.33x with one-second endpoint pauses (5.6 seconds total). `automatic/ik/move_006/videos/` also contains the normal-speed target clip, aligned original/target comparison, and a slow comparison. `target_ik_demo.json` records timing and provenance. The demo is one movement, not a continuous reconstruction of the complete route. Section 4's contact preview remains explicitly labeled **original climber**. Model-threshold coverage is approximately 81%, not measured contact accuracy.

`AUTOMATIC_IK_ENABLED = True` uses decoded predictions without the statistical review gate, while retaining actual support, timing, calibration, and solver checks. Change the target in Section 1, then restart the kernel and run all sections to refit; the new body may yield different eligible or successful moves. Summaries are saved in `state/run_summary.json`, `geometry_status.json`, and `direct_ik_plan.json`.

The earlier gray-region hold false positive was repaired with denser observed-frame reconstruction and an observed-pixel filter. H054 is a subsequent manual missed-hold correction, not a demonstrated detector-accuracy gain. `state/approved_cleanup.json` records the 15 obsolete files moved to macOS Trash with user approval; active Trial 2 data was preserved.


### Trial 3: full-sequence 180 cm adaptation

Section 7 of [Trial3_Analysis.ipynb](../../Trial3_Analysis.ipynb) provides the geometric full-sequence workflow with configurable height. The preserved [180 cm demonstration](../../outputs/trial3/full_sequence/target_180cm_250e15b3ee81/trial3_target_180cm_full_sequence.mp4) contains every frame from 0 through 502 at 30 FPS. All 503 poses are jointly optimized with fixed segment lengths and temporal smoothness; this is not a concatenation of the existing single-move clips. Each limb can change its decoded contact independently.

Known hold centers are soft targets. Incompatible contacts retain visible residual gaps instead of stretching the body. The saved run has 245 frames above the 0.1 cm numerical contact tolerance; 31 of those exceed 5 cm. H053 extrapolation, unresolved free-foot support, low-confidence contacts, low-visibility reference joints, and rapid pose changes are flagged separately. Every frame has at least one uncertainty flag. This diagnostic adaptation does not establish physical feasibility or climbing success; forces, balance, friction, collisions, and body depth remain unmodeled.

The [saved run directory](../../outputs/trial3/full_sequence/target_180cm_250e15b3ee81/) contains the trajectory, per-frame and per-limb diagnostics, issue intervals, summary, and preview images. The strict single-move results remain separate. See [the full-sequence guide](../../docs/trial3_full_sequence.md) for assumptions, warning meanings, and reproduction steps.

## Inspect cleanup candidates

Run `bash cleanup.sh` (or `bash cleanup.sh --dry-run`) to list the paths considered by the old reset script and their current sizes. The script is read-only and has no deletion mode. `outputs/` includes retained videos, reviewed maps, annotations, caches, and diagnostics; the listing is not a recommendation to remove it. Review and back up any specific files before a separate, explicitly approved cleanup.


## Trial1: conditional ground support

Section 7 of [Trial1_Analysis.ipynb](../../Trial1_Analysis.ipynb) adds conservative ground-contact evidence, an explicit ankle-to-sole offset, conditional floor forces and support-ID-based unloading to the complete 417-frame Trial1 optimization. Set target height in Section 1, select **Python (Beta Translator)**, and run the merged notebook in order. CLI: `.venv/bin/python scripts/run_trial1_ground_optimization.py --height-cm 180`.

This first version uses manually estimated floor regions and assistant visual interval assessment, not automatic wall segmentation or user-confirmed contacts. Candidates and ambiguous observations get no ground force; approximate ground geometry stays conditional. Raw predictions and historical results are retained. See [the ground-support guide](../../docs/trial1_ground_support.md) for evidence, reproduction and limitations.

Saved 180 cm run: [continuous video](../../outputs/trial1/ground_support/target_180cm_bd8c3dc75cd8/trial1_support_optimized_180cm_full_sequence.mp4) · [synchronized review](../../outputs/trial1/ground_support/target_180cm_bd8c3dc75cd8/review.html). This run retains 18 support-balance failure frames, 96 uncertain-support frames and final optimizer nonconvergence; see the linked guide for the full comparison.


## Trial1: convergence and contact refinement

Section 8 of [Trial1_Analysis.ipynb](../../Trial1_Analysis.ipynb) jointly refines all 417 frames with sparse-direct steps, explicit inner/outer convergence checks, nominal singleton COM constraints and a soft contact-restoration term. Set height in Section 1, select **Python (Beta Translator)**, and run the merged notebook in order. CLI: `.venv/bin/python scripts/run_trial1_refinement.py --height-cm 180`.

Source contacts, ground evidence and audit thresholds are retained. Exact nominal point balance can still have inadequate reserve; those warnings stay visible. Historical results are preserved. See [the refinement guide](../../docs/trial1_refinement.md) for interpretation and reproduction.

Saved 180 cm run: [continuous video](../../outputs/trial1/refined_support/target_180cm_f1456408fd1d/trial1_support_optimized_180cm_full_sequence.mp4) · [synchronized review](../../outputs/trial1/refined_support/target_180cm_f1456408fd1d/review.html). Inner and outer convergence checks pass; contact and pre-unload failure counts are zero under unchanged thresholds. **40 frames still have low support margin**, and all geometry/contact uncertainty remains visible.


## Trial1 technique-informed planning

[Beta_Translator.ipynb](../../Beta_Translator.ipynb), Section 8D, contains the English technique table and separates coaching principles from executable model assumptions. [Trial1_Analysis.ipynb](../../Trial1_Analysis.ipynb), Section 12, compares weight shift, leg-drive, non-contact flagging and optional straight-arm / upper-limb-load preferences under the same source contacts.

```sh
.venv/bin/python scripts/run_trial1_techniques.py
```

Every candidate is optimized across all 417 native frames; the video shows one selected complete plan with phase explanations. Numerical feasibility and evidence reliability precede hand-load and joint-moment proxies. The original approach poses are preserved, and uncertainty remains visible. The Way Up contact model stays frozen because its data does not supply expert technique-quality or effort labels. See [method, results and limitations](../../docs/trial1_techniques.md).

Saved 180 cm result: [selected continuous plan](../../outputs/trial1/technique_plans/target_180cm_cf107c3b5dfe/selected_plan.mp4) · [phase explanations and candidate review](../../outputs/trial1/technique_plans/target_180cm_cf107c3b5dfe/review.html). Weight shift ranks first under the declared proxy rules; all earlier uncertainty remains visible.

## Trial3 complete 3D contact-region plans

[Trial3 Analysis, Section 13](../../Trial3_Analysis.ipynb) compares hold-center, evidence-point, region, and explicit wrist/ankle-offset candidates across all **503 native frames**. It saves complete candidate trajectories, ranking reasons, an offline 3D viewer and one selected video, with unresolved constraints and uncertainty visible. The [contact-region guide](../../docs/trial3_contact_regions.md) explains the unmeasured offsets, continuity checks and gravity assumptions. The existing environment is sufficient:

```sh
.venv/bin/python scripts/run_trial3_contact_regions.py --height-cm 180
```

The selected trajectory is the least-failing tested candidate under declared assumptions; it is not a physically validated climb. Previous 2D and short-clip 3D results remain unchanged.


## Trial2 complete 3D plans across a wall hinge

[Trial2 Analysis](../../Trial2_Analysis.ipynb), Sections 11–15, compares full **180 cm, 0–975 frame** trajectories on joined wall panels. Different hypothetical middle/upper inclines change geometry and support normals; the selected video shows the body and wall in one 3D coordinate box, with an offline orbit viewer, full-candidate metrics, selection reasons and unresolved-constraint labels. The reported source height remains 173 cm. Earlier 190 cm reports are preserved as historical results.

```sh
.venv/bin/python scripts/run_trial2_spatial.py --height-cm 180
```

The existing environment is sufficient. The experiment includes evidence-bounded contact regions, wrist/ankle offsets, conservative ground proposals, support optimization and technique preferences. Angle selection is conditional, not a measured wall reconstruction. See [method, outputs and limitations](../../docs/trial2_spatial_plans.md).

Saved Trial2 180 cm experiment: [selected 3D video](../../outputs/trial2/spatial_plans/target_180cm_5f9e7fcd448b_native/selected_plan_3d.mp4) · [angle comparison](../../outputs/trial2/spatial_plans/target_180cm_5f9e7fcd448b_native/presentation_60434267bc36/angle_comparison_3d.mp4) · [interactive 3D viewer](../../outputs/trial2/spatial_plans/target_180cm_5f9e7fcd448b_native/presentation_60434267bc36/viewer.html). Selected `native_refinement` under the assumed 15°/0° wall panels; remaining failures and strict convergence are reported in [the experiment guide](../../docs/trial2_spatial_plans.md#saved-180-cm-experiment).

### Personalized shared model

Trial1–3 now include the editable [175 cm / 176 cm arm-span profile](../../configs/users/trial_user_175.json), preserved earlier stages and new short comparisons. [Profile setup, active constraints and limitations](../../docs/personalized_body.md) distinguish numerical checks from personal physiological calibration. Open a trial notebook with **Python (Beta Translator)** and run all to view saved results. Edit `USER_PROFILE` and enable `RUN_VERIFICATION` to fit a different person.

```sh
python scripts/run_body_model.py --trial all --user-profile configs/users/trial_user_175.json --frames verification
```
