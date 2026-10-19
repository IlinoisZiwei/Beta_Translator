# CIS 5810 Final Project — Beta Translator

Ziwei Wang · Justin Zou · Pengfan Zhang

Beta Translator analyzes a climbing video and fits a target body to reviewed contact positions. The current outputs are a static 2D IK fit and a replay of the original tracked movement. Personalized movement recommendations remain a project goal.

## Run locally on macOS

1. Retrieve the actual Git LFS video/model assets with `git lfs install` and `git lfs pull`. Install Git LFS with `brew install git-lfs` if needed.
2. For first-time setup, run `bash setup.sh` from the repository. An already working `.venv` can be reused.
3. Open `Beta_Translator.ipynb` in VS Code/Jupyter with the repository as the working directory. Select `.venv/bin/python` as its kernel.
4. Run the notebook in order. Section 1 checks dependencies without installing packages or transferring files.
5. In Section 7, review drafts, choose BEFORE/AFTER frames, and manually confirm four contacts per endpoint. Click **Save annotations**.
6. Run Section 8 for the target-body fit, Section 9 for the original-motion replay, and Section 10 for the export.

Use **Load annotations** after restarting. Changing an endpoint resets its contact labels. Automatic drafts never replace manual choices. After editing a helper `.py` file, restart the kernel to import the changes.

## Files

| File | Responsibility |
| --- | --- |
| `Beta_Translator.ipynb` | Settings, pipeline, review controls, and visual results |
| `beta_translator/inverse_kinematics.py` | `RATIOS`, body construction, forward kinematics, and IK |
| `beta_translator/pose_adapter.py` | Coordinate conversion and pose-vector construction |
| `beta_translator/contact_matching.py` | Hold-mask proximity and persistent contact intervals |
| `beta_translator/project_io.py` | Video I/O, reviewed maps, caches, and endpoint persistence |
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

- `pose_coordinates.csv` and `pose_output.mp4`: one tracking pass and one set of thresholds.
- `pose_manifest.json`: video/model/settings and output hashes for reuse checks.
- `wall_reference.jpg` and `reviewed_hold_map.npz`: frame-0 image, masks, stable IDs, and reviewed route.
- `alignment_cache.npz`: current transforms, status, settings, and matching input hashes.
- `endpoint_annotations.json`: explicitly saved endpoint frames, contact labels, moving limb, notes, body ratios, height, and wall scale.
- `contact_matching/`: draft contacts, intervals, sample images, settings, and summary.
- `wall_motion_h264.mp4`: original tracked motion on the fixed hold map.
- `reviewed_contact_data.zip`: current analysis inputs/results and saved endpoints when available.

Old or mismatched alignment caches are recomputed. Export checks that contact results and alignment match the current saved poses. Video and model weights remain separate inputs.

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
