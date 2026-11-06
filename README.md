# Beta Translator — CIS 5810 Final Project

Beta Translator turns climbing-video observations into wall/hold maps, reviewed contact schedules and target-body motion diagnostics. A shared anatomical model adapts motion to a user profile; Anny maps a saved trajectory to a human surface.

**Current outcome:** all three trials have saved full-sequence displays for a 175 cm height / 176 cm arm-span demonstration profile. The full source solves did not converge and retain constraint failures. These are complete diagnostic trajectories, not validated climbing recommendations.

## Start with the three reports

| Report | Source sequence | Scene and main review question |
| --- | --- | --- |
| [Trial1](Trial1_Analysis.ipynb) | 0–416; 13.90 s | Projected wall and conditional ground support: are the contact/support assumptions consistent? |
| [Trial2](Trial2_Analysis.ipynb) | 0–975; 32.53 s | Joined wall panels and identity tracking: how do panel transitions and missing observations affect the result? |
| [Trial3](Trial3_Analysis.ipynb) | 0–502; 16.77 s | Calibrated wall-plane observations with assumed depth/tilt: how do contact regions and surface directions affect the result? |

All sequences use their native 30 FPS. Each notebook has the same ten main sections:

| Section | Purpose |
| --- | --- |
| 1. Overview and key findings | Identify the saved full run, separate short comparison and report controls. |
| 2. Inputs and scene assumptions | Inspect the source video and distinguish observed geometry from assumptions. |
| 3. Pose tracking and data quality | Review source-person joints, confidence and missing or rejected observations. |
| 4. Holds and route | Inspect the saved wall reconstruction, candidate holds and selected route. |
| 5. Contacts and movement timing | Review inferred hand/foot contacts, release/arrival times and source replay. |
| 6. Personalized body and constraints | Check the target user's dimensions, shared body model and constraint coverage. |
| 7. Full-sequence results and diagnostics | Inspect one complete target-body trajectory and its convergence/constraint failures. |
| 8. Short-clip candidate comparison | Compare control and active candidates within a separate short interval. |
| 9. Anny visualization of the selected full sequence | View the exact Section 7 motion as a human surface and source-video overlay. |
| 10. Conclusions and reproducibility | Review supported conclusions, remaining limits and result locations. |

Appendix A preserves historical experiments; Appendix B holds detailed constraints and source checks; Appendix C links preserved artifacts. The default report shows concise tables and four videos: source, decoded contacts, full skeleton/mesh comparison and source overlay. Review widgets, short-comparison playback and expanded historical media are optional.

The original [Beta_Translator.ipynb](Beta_Translator.ipynb) remains the separate Trial1 teaching/demo workflow with manual review and the H44 → H28 example. [Automatic_Contacts.ipynb](Automatic_Contacts.ipynb) is the contact-inference entry point, not a presentation report.

### Recommended reading order for group review

1. **Get oriented:** read this README, then visit **Sections 1 → 2 → 7 → 9 → 10** in each report to connect the inputs, full results, visualizations and limitations.
2. **Understand the method:** read **Trial1 Sections 1–10** in order, then **Trial3**, then **Trial2**. Trial1 introduces the common workflow; Trial3 adds wall calibration and contact-region questions; Trial2 adds joined panels and identity-tracking complications.
3. **Investigate questions:** return to Sections **3–5** for evidence/contact issues, Section **6 and Appendix B** for model/constraint issues, **Appendix A** for method development, and **Appendix C** for artifact locations.

Reading may skip sections; execution still proceeds from the first code cell downward. For ordinary review, use saved outputs and keep `RUN_FULL_SEQUENCE`, `RUN_COMPARISON`, `RUN_AVATAR` and `RENDER_COMPARISON` set to `False`. Sections **7 and 9 share one full run**; Section **8 is an independent short comparison**. See the [detailed section and appendix guide](docs/complete_trial_notebooks.md#section-guide) for review questions and the role of each historical experiment.

## Understand which result you are viewing

| Output | Meaning | Source |
| --- | --- | --- |
| Observed skeleton/contact replay | The filmed climber and saved inferred contacts | Reviewed evidence and prediction caches |
| `FULL_RESULT` | One continuous target-body trajectory covering every native frame | The full source run referenced by [the avatar index](configs/avatar_results.json), or an explicitly selected/generated full run |
| `COMPARISON_RESULT` | A separate short interval comparing the anatomical control with active mechanics | [Personalized comparison index](configs/personalized_results.json) |
| Anny full-sequence videos | Appearance transfer of exactly `FULL_RESULT` | Matching source summary, trajectory and metrics hashes, profile, candidate and frame range |

The saved full runs tested one active `steady` candidate with 15 iterations. The saved short comparisons used `anatomical_control` and `steady`, with 35 iterations each: Trial1 frames 113–136, Trial2 550–595, Trial3 296–331. These do not establish a ranking among full-sequence alternatives. Candidate failure totals from different models, intervals or bodies are not directly comparable.

If a new full run has no matching Anny output, the report says so. It never substitutes the saved short clip or an avatar of another run. Changing the display mesh alone is not a new user's motion adaptation.

## Current full-sequence diagnostics

| Trial | Source solver converged | Frames with source constraint failure | Frames with mesh-to-hold gap > 0.1 cm | Mesh-to-solver mapping passed |
| --- | --- | --- | --- | --- |
| Trial1 | False | 417 / 417 | 400 | True |
| Trial2 | False | 976 / 976 | 914 | True |
| Trial3 | False | 503 / 503 | 464 | True |

At least one recorded source constraint fails on each frame; this does not mean every constraint fails. Mesh mapping agreement is different from reaching a hold. Assumed-wall penetration and source failures remain visible. `physical_validation` is false for all three. See the [saved validation record](docs/validation/anny_avatar_20261105.md) and per-run manifests for definitions and historical checks.

## Open locally

From the repository root on macOS:

```bash
bash setup.sh
```

Select **Python (Beta Translator)** / `.venv/bin/python` in VS Code. Open one of the three Analysis notebooks and choose **Restart Kernel → Run All**. Default execution reads saved results and refreshes notebook displays; it does not fit motion, generate avatars, render missing motion videos or export review files.

`bash Start.command` opens JupyterLab. With Python 3.12 and FFmpeg already installed, `bash setup.sh --skip-system` skips Homebrew. The repository uses one `.venv`; any old `.venv-contact` directory is unused. Keep Git LFS assets available; setup checks the saved asset manifest.

The original `Beta_Translator.ipynb` has a manual annotation step before downstream IK, so its execution is different from the three report notebooks. Follow its review controls before continuing. `Automatic_Contacts.ipynb` performs inference/export operations and also has a different purpose.

## Report controls

All switches are in Section 1. Every generation switch defaults to `False`.

| Control | Effect |
| --- | --- |
| `USER_PROFILE` | Requested dimensions and assumed capacity parameters; saved motion must match the profile |
| `FULL_SOURCE_RUN` | Optional path to an already saved full shared-body run; default follows the full avatar index |
| `RUN_FULL_SEQUENCE` | Explicitly fit all frames and select the new `FULL_RESULT`; no automatic motion-video rendering |
| `RUN_COMPARISON` | Explicitly fit the separate verification interval; does not replace `FULL_RESULT` |
| `RUN_AVATAR` | Explicitly generate Anny from `FULL_RESULT`, without refitting it |
| `RENDER_COMPARISON` | Explicitly render the short comparison's figures/videos |
| `ITERATIONS` | Budget for newly requested fits; does not change a saved run's recorded budget |
| `SHOW_REVIEW_TOOLS` | Show frame, route and BEFORE/AFTER review widgets |
| `SHOW_COMPARISON_VIDEO` | Play the saved short comparison inline |
| `SHOW_DETAILS` / `SHOW_HISTORY` | Expand detailed checks / historical tables, figures and videos |

Route/contact exports are separate artifacts under `outputs/notebook_reviews/<trial>/`. They do not automatically replace the canonical evidence or recompute existing motion. Revised evidence must be deliberately adopted by the pipeline before a new fit. See the [notebook workflow guide](docs/complete_trial_notebooks.md).

## Model and evidence boundaries

- One `BodyModel` is shared across trials: 32 pose coordinates, anatomical arms and legs, fixed dimensions, common contact patches and native-frame temporal bounds.
- The [default profile](configs/users/trial_user_175.json) specifies 175 cm height and **176 cm bilateral arm span**. Its other proportions, 70 kg mass and physiological parameters are mainly demonstration assumptions, not measurements of the filmed climber.
- Shoulder envelopes, capsule collision, joint-moment capacity, fatigue and inertial balance participate in fitting and candidate checks. They remain simplified, uncalibrated proxies.
- Trial1 depth/ground, Trial2 joined-wall geometry and Trial3 tilt/depth are conditional. Missing foot contact is not evidence of ground support.
- Anny transfers saved rotations and surface contacts. Fingers use a template pose; original-person removal, completed occlusion, impact simulation and forward dynamic validation are absent.

Review geometry, contact timing, support modes, shoe directions and capacity inputs before interpreting persistent failures as optimizer problems or real-human impossibility.

## Repository map

| Location | Purpose |
| --- | --- |
| `Trial*_Analysis.ipynb` | Current reports and historical appendices |
| `beta_translator/` | Evidence processing, shared body/constraints, solvers and avatar implementation |
| `notebooks/report_workflow.py` | Read-only report summaries and strict motion–avatar matching |
| `notebooks/trial_workflow.py` | Evidence viewers, review widgets and historical displays |
| `configs/` | User/trial configuration and saved-result indices |
| `videos/`, `data/`, `models/` | Source videos, reviewed/calibration data and model assets |
| `outputs/` | Evidence caches, trajectories, audits, videos and provenance; some older runs are current initialization inputs |
| `scripts/`, `tests/` | Explicit processing/generation commands and regression checks |
| `docs/`, `notebooks/history/` | Methods, validation records and preserved notebook snapshots |

Trial1's original evidence is under `outputs/`; its later fits are under `outputs/trial1/`. Trial2 and Trial3 evidence lives under their respective trial folders. This layout is retained so existing inputs and manifests continue to resolve.

## Method details and historical appendix

- [Active personalized model](docs/personalized_body.md), [constraint specification](docs/body_constraint_specification.md), [lower-limb chain](docs/anatomical_lower_limbs.md).
- [Contact inference and training](docs/automatic_contacts.md), [contact model card](models/contact_v1/MODEL_CARD.md), [Anny adapter and licensing](docs/anny_avatar.md).
- [Historical README and experiment record](docs/history/README_before_report_cleanup_20261105.md): preserves prior commands, results, methods and references. Its section numbers and “current/new” descriptions are historical.
- Appendix A links retained historical experiments through the [artifact catalog](notebooks/workflow_catalog.json); [notebook snapshots](notebooks/history/) remain available. The [cleanup and archive map](docs/history/cleanup_20261105.md) records the two duplicate Trial1 runs archived outside this repository and the relocated setup/inventory documents.

Older method documents contain dated model states; use their update notices and the saved run's provenance to interpret them. This report cleanup changes presentation and result selection, not solver behavior or physical-validity claims.
