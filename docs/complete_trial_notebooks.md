# Complete trial notebooks

**Current report organization — November 5, 2026.** All three Analysis notebooks use ten main sections, with historical experiments in Appendix A, detailed checks in Appendix B and preserved artifacts in Appendix C. The main `Beta_Translator.ipynb` is unchanged. The earlier guide is retained below as historical documentation.

## Recommended reading order

Start with the [README](../README.md), then use the same main-section structure across [Trial1](../Trial1_Analysis.ipynb), [Trial3](../Trial3_Analysis.ipynb) and [Trial2](../Trial2_Analysis.ipynb).

| Review pass | Order | What to establish |
| --- | --- | --- |
| Group overview | Sections **1 → 2 → 7 → 9 → 10** in each report | Which inputs and assumptions produced the full result, what the videos show, and what remains unresolved. |
| Complete method review | **Trial1 Sections 1–10 → Trial3 Sections 1–10 → Trial2 Sections 1–10** | Learn the shared workflow first, then inspect calibration/contact regions, then joined-wall and identity-tracking complications. |
| Investigate a finding | Sections **3–5** for observations/contacts; **6 and Appendix B** for model assumptions; **8** for the short comparison | Trace a visible failure or numerical claim back to its evidence and definition. |
| Historical or code review | **Appendix A → relevant result files and implementation**; use **Appendix C** to locate artifacts | Understand why a method changed and inspect the exact saved experiment supporting the discussion. |

For each finding, note the trial, section, run or frame interval, observation, relevant assumption and linked evidence. This makes it easier to discuss a contact error or solver failure without mixing experiments.

Reading can jump between sections. Execution must start at the first code cell and proceed downward because later cells use objects created earlier. For ordinary group review, keep `RUN_FULL_SEQUENCE`, `RUN_COMPARISON`, `RUN_AVATAR` and `RENDER_COMPARISON` set to `False`; the stored output and linked files are the starting point.

If displays need refreshing, use **Python (Beta Translator)** and **Restart Kernel → Run All** with those defaults. Execution reads saved files and refreshes notebook output without motion fitting, avatar generation, implicit video rendering or review export. Source code cells are collapsed for reading except the settings cell; expand them to inspect the implementation. These execution instructions apply to the three Analysis reports. The original `Beta_Translator.ipynb` has a manual review step, and `Automatic_Contacts.ipynb` is an inference/export workflow.

## Section guide

The main flow is source video and scene assumptions → observed pose, holds and contacts → target-body model → full-motion diagnostics → Anny appearance transfer. Section 8 branches into a separate short comparison. **Sections 7 and 9 describe the same full trajectory.**

| Section | What it does | Questions for review |
| --- | --- | --- |
| **1. Overview and key findings** | Loads configuration and saved-run identities; collects all runtime and display switches. | Which full and short runs are selected? Do their frame ranges, budgets and candidate sets differ? |
| **2. Inputs and scene assumptions** | Shows the source video and explains scale, wall geometry, depth and any ground evidence. Trial3 includes a separate calibration-evidence subsection. | Which quantities come from observations or annotations, and which are hypotheses? Where does the calibration apply? |
| **3. Pose tracking and data quality** | Shows source-person joints, confidence and observation quality, with an optional frame viewer. | Are joints occluded, missing or assigned to another person? Is display interpolation being mistaken for evidence? |
| **4. Holds and route** | Shows the saved wall reconstruction, candidate hold masks and selected route, with optional route review. | Are hold IDs, positions and route membership consistent with the video? |
| **5. Contacts and movement timing** | Shows contact proposals, release/arrival times and source-person contact replay; offers optional BEFORE/AFTER review. | When does a limb leave its old hold and reach its new one? Are free, unknown and supported contacts distinguished? |
| **6. Personalized body and constraints** | Shows the requested dimensions, shared anatomical model and constraint inventory. | Which parameters are measured versus assumed? Which checks are hard bounds, soft objectives, audits or missing coverage? |
| **7. Full-sequence results and diagnostics** | Selects `FULL_RESULT`, verifies its identity and full frame range, and displays solver and constraint diagnostics. | Does it cover every native frame? Which checks fail, and did the solver converge? |
| **8. Short-clip candidate comparison** | Uses a separate `COMPARISON_RESULT` to compare the anatomical control and active mechanics on the same short interval and user geometry. | What improves or worsens within this clip? Are full-run failure totals being incorrectly compared with short-clip totals? |
| **9. Anny visualization of the selected full sequence** | Checks the avatar against Section 7's source hashes/profile/candidate/range, then shows skeleton/mesh comparison and source overlay. | Does the mesh reproduce the source motion? Distinguish mesh-to-solver agreement, mesh-to-hold error, wall penetration and source constraint failures. |
| **10. Conclusions and reproducibility** | States supported conclusions and remaining limitations, and links the full run, short comparison and matched avatar. | Which claims are supported by saved evidence? Which geometry, contact or capacity assumptions should be checked next? |

The default report shows four videos: source video, source-person contact replay, full skeleton/mesh comparison and full source overlay. Short-comparison playback and historical media can be expanded separately. All three saved full solves remain unconverged with constraint failures and `physical_validation=False`; successful avatar mapping does not establish physical feasibility.

## Trial-specific review priorities

| Trial | Full sequence | Current scene assumptions | Priority questions |
| --- | --- | --- | --- |
| **Trial1** | Frames 0–416 | Projected image-plane model, a 25 cm depth prior, scale from a 46 cm reference-image measurement, and conditional ground evidence. | Are ground intervals and support changes justified? How sensitive is the interpretation to projected scale and assumed depth? |
| **Trial2** | Frames 0–975 | Joined panels with a 15° middle panel, 0° upper panel and assumed 15 cm hole spacing; camera and surface directions remain hypotheses. | Are identity gaps handled correctly? Do panel transitions, contact timing and hand/foot directions explain the failures? |
| **Trial3** | Frames 0–502 | Reviewed 15 × 15 cm bolt-grid calibration and a conditional 15° wall tilt; depth is not fully measured. | Where is calibration valid? Are contact regions, hand/foot offsets and inferred surface directions reasonable? An arbitrary height origin does not establish ground contact. |

The current demonstration profile is 175 cm tall with a 176 cm bilateral arm span. Historical results retain their recorded bodies and assumptions; changing the current profile does not retarget an old video.

## Historical experiments: Appendix A

Read Appendix A when discussing method development or a specific older result. Its entries display saved artifacts; they do not rerun historical optimization. Earlier failure totals use different bodies, intervals and model definitions, so they are not a single progress score.

### Trial1

| Subsection | Role in the method history |
| --- | --- |
| **A.1. Reviewed endpoints and H44 → H28 single-move IK** | Preserves the reviewed endpoint and continuous single-move demonstration for the original 165 cm target. |
| **A.2. Gravity IK and full-sequence support optimization** | Introduces modeled gravity, center of mass, contact-force allocation and support/release checks across the full sequence. |
| **A.3. Ground evidence and ground-aware adaptation** | Adds conditional support from image regions, reviewed intervals and visible foot evidence; missing wall contact alone is insufficient. |
| **A.4. Full-sequence refinement and convergence review** | Examines contact restoration, convergence, unloading and support reserve separately. |
| **A.5. Technique candidates, complete-trajectory ranking and selected plan** | Compares weight-shift, leg-drive, flagging and upper-limb-load templates as complete trajectories; no local winners are stitched together. |
| **A.6. Previous 175 cm anatomical verification** | Preserves the earlier anatomical comparison before personalized geometry and active mechanics. |

### Trial2

| Subsection | Role in the method history |
| --- | --- |
| **A.1. Single-move IK gate and historical geometry blockers** | Records why the earlier attempt lacked sufficient metric geometry; there is no fitted video for that blocked attempt. |
| **A.2. Wall reconstruction and identity-tracking evaluation** | Reviews multi-frame wall evidence, rejection of bystander switches and remaining observation gaps. |
| **A.3. Joined-wall angle candidates and complete 3D trajectories** | Compares fixed joined-panel angle hypotheses and complete motion plans; ranking does not identify the true wall angle. |
| **A.4. Earlier selected 3D plan, gravity/support and selection reasons** | Shows the selected full historical plan, phase explanations and separate contact/support/unloading checks. |
| **A.5. Anatomical arms: abnormal short-clip comparisons** | Tests the shoulder–elbow–forearm–wrist chain on abnormal clips before the full solve. |
| **A.6. Anatomical-arm full-trajectory ranking and failure review** | Optimizes and ranks full candidates independently from the clips; a selected failing plan remains diagnostic. |
| **A.7. Full anatomical-arm video and before/after comparison** | Compares matched timestamps and fixed views alongside arm, reach, contact and solver metrics. |
| **A.8. Stage 1: verified arm chain and common hand/foot contact audit** | Restores source wrist coupling and uses one contact-patch definition in fitting, force auditing and rendering. |
| **A.9. Stage 1 continuous contact comparison** | Shows contact-patch connectors, residual errors and inferred shoe orientation over time. |
| **A.10. Previous 175 cm anatomical verification** | Preserves the earlier anatomical comparison before personalized geometry and active mechanics. |

### Trial3

| Subsection | Role in the method history |
| --- | --- |
| **A.1. Calibrated single-move IK** | Fits a compatible move using the bolt-grid wall-plane calibration and retains extrapolation warnings. |
| **A.2. Full-sequence geometric IK** | Fits every pose jointly with fixed dimensions, changing contacts and temporal regularity; force/support checks are absent at this stage. |
| **A.3. Gravity IK, contact validity, balance reserve and unloading** | Adds modeled gravity and force allocation with separate contact, balance and release audits. |
| **A.4. Support-optimized full sequence and comparison** | Includes contact, balance and unloading terms in full-sequence optimization and compares with its gravity initialization. |
| **A.5. Conditional 3D tilt and depth experiments** | Tests tilt/depth sensitivity; these hypotheses do not establish measured wall geometry. |
| **A.6. Refined angle comparison** | Further optimizes the short trajectory under the tested scene assumptions. |
| **A.7. Contact-region and hand/foot-offset candidates** | Compares complete plans using different contact regions and illustrative offsets, including native-frame refinement. |
| **A.8. Previous 175 cm anatomical verification** | Preserves the earlier anatomical comparison before personalized geometry and active mechanics. |

## Detailed checks and artifact index: Appendices B and C

**Appendix B** links the selected full run's constraint checks, anatomical source-chain verification and native-frame input validation. `SHOW_DETAILS=True` expands the inventory and lightweight anatomical transform checks without fitting motion.

**Appendix C** locates notebook snapshots, the historical README, the artifact catalog and saved Anny cases. These additional cases may be historical or short clips; selecting a link does not change the main full result. The [cleanup and archive map](history/cleanup_20261105.md) records the two duplicate Trial1 runs moved outside the repository, their retained counterparts and restoration instructions. The current reports do not require those ZIPs to be extracted. Older validation records retain their original paths and claims.

## Finding the evidence and implementation

Use the report's result links to identify a run before browsing its files. Trial1's original evidence is under `outputs/`, while later fits are under `outputs/trial1/`; Trial2 and Trial3 evidence lives under their respective trial directories. Some older runs still supply initialization inputs.

| Review question | Start here |
| --- | --- |
| Which user and scene are being evaluated? | [User profiles](../configs/users/) and [trial configurations](../configs/trials/). |
| Which saved run feeds the display? | [Anny cases](../configs/avatar_results.json), [personalized short comparisons](../configs/personalized_results.json) and [historical artifact catalog](../notebooks/workflow_catalog.json). |
| What supports a numerical or visual claim? | The linked run's `summary.json`, trajectory, metric/audit tables and corresponding videos; keep the run identity and frame interval together. |
| How are results selected and matched to Anny? | [Report helpers](../notebooks/report_workflow.py). |
| How are source evidence and review widgets handled? | [Evidence and review helpers](../notebooks/trial_workflow.py). |
| Where do fitting and constraints live? | [Shared model entry point](../beta_translator/body_model.py), [pipeline](../beta_translator/body_pipeline.py), [constraint definitions](../beta_translator/body_constraints.py) and their imported modules. |
| Where are appearance transfer and its limits described? | [Anny guide](anny_avatar.md) and [avatar pipeline](../beta_translator/avatar_pipeline.py). |
| How can implementation behavior be checked? | [Tests](../tests/) and [recorded validation](validation/); use [scripts](../scripts/) only when deliberately reproducing processing or experiments. |

## Full motion, short comparison and avatar selection

`FULL_RESULT` is selected in Section 7. By default the report reads the `full` avatar entry in `configs/avatar_results.json`, loads that avatar's exact source run and verifies its files, trial configuration, profile and complete native interval. `FULL_SOURCE_RUN` can explicitly select another saved full run. `RUN_FULL_SEQUENCE=True` calls `model.fit(..., frames='all', render=False)` and selects the returned full run for the remainder of the report.

`COMPARISON_RESULT` belongs only to Section 8. Its saved intervals are Trial1 113–136, Trial2 550–595 and Trial3 296–331. `RUN_COMPARISON=True` fits that verification interval without changing the selected full run. Saved comparisons use 35 iterations per candidate; saved full runs use 15 iterations with one active `steady` candidate. They are not a like-for-like candidate ranking. If a changed user profile has no matching saved short comparison, that optional display is skipped with an explanation.

Section 9 uses only `FULL_RESULT`. `RUN_AVATAR=True` passes that full run to `run_avatar`; otherwise a matching saved avatar is located. Matching checks the source summary, trajectory and metrics hashes, candidate, profile, trial, FPS and frame range. A missing matching avatar is reported without substituting an older run or short clip. `AVATAR_CASE` and the ambiguous shared `OUT` variable are no longer used by the report notebooks; preserved older cases remain linked in Appendix C.

`RENDER_COMPARISON=True` explicitly requests short-comparison rendering. A missing presentation otherwise yields a message and the numerical metrics; it never starts implicit work. `ITERATIONS` applies only to explicitly requested new fits.

## Optional detail and review

- `SHOW_REVIEW_TOOLS` enables frame, route and BEFORE/AFTER widgets. Exports remain separate artifacts; they do not change canonical model inputs.
- `SHOW_COMPARISON_VIDEO` plays the short comparison inline. Video and metric links remain available when it is false.
- `SHOW_DETAILS` expands constraints and lightweight source-chain checks. It does not fit a trajectory.
- `SHOW_HISTORY` expands historical tables, figures and media. Historical optimization is not run.

Each historical appendix entry retains its original scope and artifact links. Earlier 165/180/190 cm results and the previous 175 cm anatomical model remain distinct from the current personalized profile. Native result manifests and all existing numerical files remain unchanged.

For current model scope, see [personalized mechanics](personalized_body.md). For the project entry point, see [README](../README.md). The [previous README](history/README_before_report_cleanup_20261105.md) preserves the detailed historical experiment record.

## Earlier workflow documentation

<details>
<summary>Preserved pre-cleanup guide, section numbers and historical validation counts</summary>

The following text describes earlier notebook layouts and checks. Its execution counts are historical, not new validation of the reorganized reports.

# Complete trial notebooks

**Personalized-model update (November 5, 2026):** The shared entry now reads an editable 175 cm / 176 cm arm-span user profile and actively fits/audits shoulder envelopes, capsule collision, joint-moment capacity, fatigue coupling and inertial balance. Defaults are uncalibrated assumptions. Earlier descriptions and results below remain historical; see [current implementation and boundaries](personalized_body.md).

`Trial1_Analysis.ipynb`, `Trial2_Analysis.ipynb` and `Trial3_Analysis.ipynb` now combine the earlier evidence/review workflows, historical experiments and the current shared anatomical body model. The original `Beta_Translator.ipynb` is unchanged.

## Open and run

Use **Python (Beta Translator)** (`.venv/bin/python`) in VS Code or JupyterLab, then **Restart Kernel and Run All**. All ordinary cells read saved evidence/results. They do not retrain models, rerun long optimization or replace reviewed maps. Images and tables are stored in the notebooks; videos use relative references to existing MP4 files. Keep the repository layout and its Git LFS assets.

| Notebook | Sections 1–8 | Restored historical experiments | Current shared model | Final index |
| --- | --- | --- | --- | --- |
| Trial1 (20 sections) | Setup, source skeleton, wall map, hold selection, geometry, predictions, BEFORE/AFTER review, source replay | 9–13: 165 cm single move; 180 cm gravity/support, ground, refinement and technique candidates | 14–19 | 20 |
| Trial2 (24 sections) | Same common evidence/review workflow | 9–17: 190 cm 2D blockers; source/identity evaluation; 180 cm wall-angle, full 3D, arm and contact-consistency comparisons | 18–23 | 24 |
| Trial3 (22 sections) | Same common evidence/review workflow | 9–15: 180 cm single move, full geometric/gravity/support, tilt/depth and contact-region plans | 16–21 | 22 |

Historical stages retain their original assumptions and heights. They are not silently retargeted when `HEIGHT_CM` changes. Trial2's old 190 cm 2D attempt produced zero fitted videos because geometry was insufficient; this is displayed as a blocker. Trial3's tilt experiments are hypotheses, distinct from user-stated wall relationships. “Gravity” means the modeled direction and contact-force calculation, not an independently measured gravity vector from video.

## Review controls

- The source-frame slider displays observed skeleton joints, aligned hold IDs and saved model contact labels. Low-visibility and identity-rejected observations are excluded.
- The route selector updates the preview and retains every candidate mask. **Export route review** writes a new map and provenance record under `outputs/notebook_reviews/<trial>/<unique-id>/`.
- **Set BEFORE / Set AFTER** choose endpoint frames. Automatic prefill loads a suggestion and leaves support continuity unconfirmed. Contact/endpoint edits reset motion timing and confirmation. Timing edits clear support confirmation.
- **Export annotations** saves a new file alongside the corresponding route map. Unconfirmed timing stays in `suggested_motion`; only a fully validated, explicitly confirmed single-limb move is stored as `motion`.
- **Load file** accepts the repository-relative path to an exported annotation or a historical annotation. Loading validates video/map/route identity and leaves the current target height unchanged.
- A changed route invalidates the old contact form and prevents automatic prefill from using predictions for the old selection. Re-run the review cell to create a new form. Missing wall contact never automatically becomes ground support.

Review exports are separate review artifacts. Saved videos, prediction caches and the shared model's canonical input schedule remain unchanged. To compute a result from changed evidence, that evidence must be explicitly adopted in the analysis pipeline and inference recomputed; exporting a draft does not claim that this has happened. The archived original workflows retain the detailed inference/review/export implementation, and `docs/automatic_contacts.md` documents the inference commands.

## New fitting and earlier implementations

The current shared-model cells retain `RUN_VERIFICATION`, `RUN_FULL_SEQUENCE`, `ITERATIONS` and `HEIGHT_CM` (175 cm by default). Turning on either run flag calls the same `BodyModel` API used previously. New results get separate content-addressed directories. The saved 175 cm comparisons are short intervals, not new complete-video optimizations.

```sh
source .venv/bin/activate
python scripts/run_body_model.py --trial all --height-cm 175 --frames verification --iterations 150
```

Historical full-sequence recomputation remains available through the existing scripts: `run_trial1_support_optimization.py`, `run_trial1_ground_optimization.py`, `run_trial1_refinement.py`, `run_trial1_techniques.py`, `run_trial2_spatial.py`, `run_trial2_anatomy.py`, `run_trial2_stage1.py`, `run_trial3_full_sequence.py`, `run_trial3_gravity_sequence.py`, `run_trial3_support_optimization.py`, `run_trial3_3d_experiment.py`, and `run_trial3_contact_regions.py`. Read each script's `--help` and linked method document for the original settings.

`notebooks/trial_workflow.py` shares the presentation/review controls. `notebooks/workflow_catalog.json` explicitly selects historical files and their hashes. No fitting/constraint code is duplicated in the notebooks or changed by this merge. The catalog checks historical output integrity without falsely requiring old experiments to match today's code.

Exact notebook snapshots remain in `notebooks/history/pre_shared_body_20261104.zip`, `pre_anatomical_legs_20261105.zip`, and the new `pre_complete_workflow_20261105.zip`. The newest archive preserves the previously executed 175 cm notebooks byte for byte. Extract historical notebooks into the repository root when their original code/settings are needed.

## Verification

All 66 code cells (20 + 24 + 22) executed in order in the actual repository with zero errors. Five review-workflow tests passed, including all-trial export preservation, draft confirmation handling, stale-route rejection and keeping the current target height when loading historical annotations. All 38 displayed videos are file references. Hash checks preserved 1,777 pre-existing output files, source videos, all numerical model code and `Beta_Translator.ipynb`; exact previous notebook snapshots were verified. See [the machine-readable record](validation/complete_notebooks_verification.json). No previous numerical results were recomputed, committed or pushed.

The current personalized notebooks have 21 / 25 / 23 sections. Each retains the prior 175 cm verification in a separate section before the new model. New result defaults use `configs/personalized_results.json`; the prior indices remain unchanged. The older section numbers and execution counts above describe the preserved merge.

The Anny appearance update appends Section 22 / 26 / 24 to Trial1 / Trial2 /
Trial3, respectively. Those sections display the calibrated human surface and
source-video overlay without rerunning optimization by default. The saved case
index is `configs/avatar_results.json`; see [calibration and limits](anny_avatar.md)
and [recorded checks](validation/anny_avatar_20261105.md). Earlier sections and
historical outputs remain intact.


</details>
