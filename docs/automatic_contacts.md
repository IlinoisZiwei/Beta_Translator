# Automatic contact model

This branch adds offline whole-body tracking, learned hold association, and release/arrival proposals. It preserves the existing reviewed hold detector, annotations, and IK implementation.

## Scope

- Pose: frozen RTMW-x whole-body model, 133 landmarks including fingertips and toes, with YOLOX and a continuous climber track.
- Contact: shared candidate encoder and six residual temporal convolution blocks, width 96, 239,233 trainable parameters per model. A softmax compares every supplied hold with `free` and `wall_contact`.
- Time: 127-frame receptive field at 25 FPS, bidirectional and offline. Viterbi decoding selects a consistent state sequence. Release is the first free frame; arrival is the first new-contact frame.
- Training: transition-centered and uniform windows, mild geometric noise, weighted cross entropy emphasizing free motion and contact boundaries, AdamW, and validation checkpoint selection.
- Review: validation-calibrated confidence determines which frame labels can be accepted. A separate movement gate checks contact/free/contact evidence, local transition ambiguity, and unchanged support contacts. Missing alignment and unresolved transitions block automatic IK proposals.

There is no learned hold-ID vocabulary: a new wall supplies its own reviewed hold boxes. There is also no force, friction, balance, climbing-success, or body-size adaptation predictor in this model. The existing IK solver still handles body-size adaptation.

## Data and evaluation rules

The public [The Way Up dataset](https://zenodo.org/records/15196867) provides 22 videos of 10 people on two routes, at 720 × 1280 and 25 FPS. Its archive contains 38,692 processed video frames and 940 annotation rows. One row identifies an unspecified off-route hold; it cannot supervise a particular hold ID. The 939 identifiable contact intervals are retained before conflict masking.

The source videos and annotations are CC BY 4.0, by Anna Maschek and David Schedl. The downloader preserves a SHA-256 manifest and verifies each ZIP member's CRC. It retrieves the processed 546.5 MB subset without downloading the 4K originals.

Splits were fixed before model training:

| Split | Participants | Videos |
|---|---|---:|
| Train | p1, p2, p3, p4, p5, p6 | 14 |
| Validation | p7, p8 | 4 |
| Test | p9, p10 | 4 |

Both p2 clothing sessions stay in training. No frames from the project's `trial1.mp4` are used in training, model selection, or calibration. The project clip is a separate integration smoke test, not an independent population-level benchmark.

Parsing rules are explicit: contact endpoints are interpreted as inclusive, conflicting simultaneous hold IDs are excluded (79 limb-frames), and the unidentified hold is excluded (11 limb-frames). A terminal `endFrame` equal to decoded frame count is clipped to the last frame and logged. Frames outside each limb's first/last annotated contact are excluded from scoring and supervision. Occlusion flags are evaluation labels, never input features.

Validation selects checkpoints, temperature, decoder penalty, and acceptance threshold. The target for accepted validation labels is 97% accuracy with at least 10% coverage; if unattainable, automatic acceptance is disabled. This target is not a guarantee on a different wall. Test people are evaluated after configuration is locked.

The movement gate considers local old-contact/free/new-contact paths and exports a central 90% path-mass range for each boundary. These are model ambiguity ranges, not guaranteed statistical coverage intervals. Its maximum range width is selected on validation: at least ten accepted proposals must have at least 95% correct hold/support identities and both boundaries within 0.20 seconds. If no setting qualifies, automatic IK is disabled while per-frame contact predictions remain available. This avoids requiring every ambiguous boundary frame to exceed the frame-label confidence threshold.

Metrics include hold/free/wall frame accuracy, accuracy under occlusion, accepted-label accuracy and coverage, event precision/recall/F1 at temporal IoU 0.5, and matched-event release/arrival errors. Timing error is conditional on matching an event and must be read alongside event recall. Coverage counts limb-frames; it is not a measured reduction in human annotation time.

The comparison baseline uses the same whole-body landmarks, normalized point-to-hold-box distance, temporal voting and decoding. Hand/foot distance settings, voting duration and transition cost are selected for event F1 on the same validation people. Its vote fractions are not calibrated confidence scores, so automatic-acceptance figures are not reported for it. It is not a numerical rerun of the older MediaPipe notebook baseline.

## Reproduce

Use an isolated environment. Existing notebook dependencies need not be changed.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-contact.txt
.venv/bin/python scripts/download_wayup.py --output training_data/the_way_up
.venv/bin/python scripts/extract_contact_poses.py \
  --dataset training_data/the_way_up --output training_data/poses_v2 \
  --device coreml --threads 2
.venv/bin/python scripts/prepare_contact_data.py \
  --dataset training_data/the_way_up --poses training_data/poses_v2 \
  --output training_data/features
```

Use `--device cpu` where CoreML is unavailable. Training uses `--device mps` on an Apple GPU, `cuda` on a compatible GPU, or `cpu`.

```bash
.venv/bin/python scripts/train_contact_model.py \
  --data training_data/features --output training_runs/seed17 \
  --seed 17 --epochs 24 --steps 60 --device mps
# Repeat with seeds 29 and 43 and separate output directories.
.venv/bin/python scripts/evaluate_contact_model.py \
  --data training_data/features --package models/contact_v1 --phase calibrate \
  --checkpoints training_runs/seed17/best.pt training_runs/seed29/best.pt \
  training_runs/seed43/best.pt --device mps
.venv/bin/python scripts/evaluate_contact_model.py \
  --data training_data/features --package models/contact_v1 --phase test --device mps
```

Training writes epoch logs, a validation-selected checkpoint, and an atomic resume checkpoint. Resume with the same command plus `--resume`. Source datasets, feature caches, and temporary training runs are ignored by Git. The deployment package holds the small trained checkpoints and their hashes.

## Use in the existing pipeline

1. Keep the current reviewed hold map and verified camera alignment.
2. Extract RTMW landmarks from the new clip. Existing MediaPipe columns are not interchangeable with these learned features.
3. Run automatic contact inference into a separate output folder.
4. Use accepted events and eligible motion proposals. Review uncertain candidates in the existing endpoint UI.

```bash
.venv/bin/python scripts/extract_contact_poses.py \
  --video videos/trial1.mp4 --output outputs/automatic/wholebody.npz --device coreml
.venv/bin/python scripts/predict_contacts.py \
  --video videos/trial1.mp4 --pose-cache outputs/automatic/wholebody.npz \
  --reviewed-output outputs --package models/contact_v1 \
  --destination outputs/automatic/predictions --device mps
```

Outputs:

- `limb_contacts.csv`: each limb's candidate, accepted hold or `unknown`, confidence and review status for every native video frame.
- `contact_events.csv`: candidate hold-contact intervals and review flags. Event records may require review even when some frames are accepted.
- `automatic_contacts.json`: release/arrival proposals, support contacts, uncertain frame ranges, and input/model hashes.
- `contact_probabilities.npz`: full distributions for visualization and audit.

The model operates at 25 FPS. Other frame rates are resampled by timestamp, then predictions are mapped back to native frames. This does not create finer temporal evidence than the 40 ms model grid. Temporal predictions use future frames; this is not a live detector.

`motion_annotation_from_prediction(template, move, prediction)` returns a new annotation compatible with `contact_schedule` and `solve_motion_sequence`. It requires the calibrated movement gate and three unchanged accepted support contacts, and checks video/hold-map identity. The template supplies measured wall scale and body proportions. It never overwrites reviewed endpoint annotations or interprets predicted visual contact as a force measurement.

`run_automatic_motion(video, reviewed_output, prediction_file, move_index, destination)` runs the existing IK pipeline for an eligible proposal in a separate output directory. It verifies input hashes and reuses the existing IK pose references and alignment. Uncertain proposals raise an error before creating an IK run. The new whole-body cache drives contact inference; the older pose table remains the IK reference in this version.

When the movement gate is disabled or a proposal needs review, `review_annotation_from_prediction(template, move, prediction, contacts)` prefills a separate review draft using accepted endpoint labels and proposed timing. Its `suggested_motion` is not a confirmed `motion` record, so the existing IK validator rejects it until reviewed. The supplied deployment package's model card states whether automatic IK is enabled.

## Limitations and next validation

The held-out people use the same two public routes and wall. These results do not measure transfer to an unseen gym, a steep wall, a moving camera, or unusual climbing techniques. Before accepting predictions broadly in the project, evaluate an independently annotated set of new-wall moves, including difficult occlusions. Collect errors and label only the uncertain or incorrect examples for a subsequent training round.

The Way Up has no ground-truth anatomical keypoint annotations, so it supports supervised contact training but not a new pose-accuracy claim. RTMW pose weights remain frozen. Climber tracking is designed for one foreground climber and can still fail under long occlusions or crowded scenes.

References: [RTMLib](https://github.com/Tau-J/rtmlib), [MMPose RTMPose/RTMW](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose), [The Way Up](https://zenodo.org/records/15196867). Retain the upstream code/model licenses and dataset attribution when redistributing.
