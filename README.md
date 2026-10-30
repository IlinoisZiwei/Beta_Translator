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

Section 1 contains the video path, output folder, pose thresholds, target height, and wall scale. `RATIOS` is defined once in the IK file. Override individual entries with measured segment length divided by height:

```python
RATIOS = {**RATIOS, 'forearm': measured_forearm_cm / TARGET_HEIGHT_CM}
```

For a new video, choose a separate output folder and set `USE_SAVED_HOLD_MAP = False`. Review the new detector output and fill in `NEW_ROUTE_HOLD_IDS` in Section 4. Fresh detection needs `models/experiment_config.yml` and `models/model_final.pth`. IDs belong to one map; never reuse an old selection without review.

All wall geometry uses frame 0. The supplied clip's scale is approximately 1.798 pixels/cm, based on a measured 46 cm edge-to-edge gap. The physical endpoints are H44/H40 in the reviewed 65-hold map, formerly H45/H41. The default scale is enabled only for the original video's hash. A new video requires its own measured scale, independent of body height.

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
