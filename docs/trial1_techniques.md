# Trial1 technique candidates and selection

The English technique table is in **Beta_Translator.ipynb, Section 8D**. The
executable experiment is in **Trial1_Analysis.ipynb, Section 12**. It adds a
candidate-planning layer to the existing IK model while preserving the trained
contact network, contact labels, source video and previous results.

```sh
.venv/bin/python scripts/run_trial1_techniques.py
```

Use the existing **Python (Beta Translator)** kernel. No new environment,
package, training service, model download or account is required. The default
body is 180 cm with the preceding refined Trial1 body's proportions and physics.
Run Section 8 for a different body first, then pass its output as `--baseline-dir`
and the matching `--height-cm`. Section 12 passes the preceding notebook run
automatically and checks body, mass, friction and force limit before reuse.

## What this upgrade implements

The model generates six whole-trajectory alternatives in addition to the
preserved baseline:

| Candidate | Added preferences |
| --- | --- |
| `control` | Same solver and constraints, with technique terms disabled. |
| `weight_shift` | A bounded horizontal COM guide toward the usable feet's horizontal support interval. No compulsory hip lift. |
| `leg_drive` | Weight shift plus a phased 6 cm hip-height guide around hand movement. This is a pose preference, not a measured leg-work model. |
| `flag_assist` | Leg-drive preferences plus a bounded non-contact free-leg path adjustment during an existing foot-transfer interval. |
| `arm_economy` | Weight shift and hip-height guides plus optional elbow-flexion, hand-force and simplified shoulder/elbow-moment penalties. |
| `combined` | All the preceding preferences. More techniques do not automatically rank higher. |

The guides are soft and may not be realized if other objectives dominate.
Realized hip changes, knee extension and free-foot displacement are saved in
`candidate_comparison.csv`. A profile name describes its requested preferences,
not proof that it performed a human technique correctly.

Each candidate jointly optimizes **all 417 frames, 0–416 at 30 FPS**. Approach
poses through frame 102 and the first/last two poses are fixed to the preceding
trajectory. Candidate contact forces can still be reoptimized. Unknown/free
contacts receive exactly zero force. Segment lengths, source contact IDs,
automatic release/arrival timing, ground evidence and audit thresholds do not
change. The final video is one selected complete trajectory; per-move winners
are not concatenated. New hold orders, foot swaps and hand-match schedules are
not searched in this experiment.

The optimizer retains sparse-direct bounded fitting, exact nominal singleton
COM constraints, contact restoration and explicit inner/outer convergence.
New posture and moment derivatives are checked against finite differences,
including the temporal COM-coordinate transformation. A candidate is not
recommended merely because its animation looks plausible.

## Observations in Trial1 versus technique labels

`source_evidence.jpg` shows selected native source frames. The accompanying
`technique_review_queue.csv` measures projected hip changes and knee-extension
changes around automatic contact events. These measurements use the existing
uniform image scale and uncertain observed pose; they are not 3D measurements.

Useful review intervals include the left-hand H44→H28 movement, right-foot
H65→H60 transfer, left-foot H62→H51 transfer and later right-foot H60→H45 transfer.
They provide examples in which body positioning and foot support can be tested.
A free foot may simply be moving to its next hold. **No observed flagging class
is declared**, and all expert technique labels in the queue remain blank.

The automatic H44→H28 release/arrival frames are 113/136. The separate preserved
human-confirmed example uses 117/128. This disagreement remains visible and is
not silently corrected to make an experiment appear more certain. The technique
planner compares candidates under the same automatic full-sequence schedule.

## Fair load proxies and ranking

Different feasible contact-force distributions can produce different apparent
hand loads for the exact same pose. Comparing whichever forces a candidate's
optimizer happened to choose would therefore be misleading. Every candidate,
including the old baseline, is evaluated with the **same two-stage LP**:

1. Enforce planar equilibrium, original component capacity and foot friction on
   reached conditional contacts; minimize total absolute hand-force components.
2. Keep that hand-load optimum within a small numerical tolerance and minimize
   absolute simplified shoulder/elbow moments.

Shoulder and elbow moments include the modeled upper-arm and forearm/hand
segment weights and distal contact forces. They omit muscle moment arms,
co-contraction, finger grip, depth and fatigue. The reported L1 hand-load peak
uses body-weight units; its integral uses BW·s. Moment integrals use BW·cm·s.
These are **conditional model proxies**, not measured forces, calories, endurance
or percent energy saved.

In particular, this 2D foot-force model permits zero hand load when the feet can
balance the body alone. The count of such frames is reported for every candidate.
A zero lower bound cannot demonstrate that a real climber could let go, and it
cannot distinguish techniques in those frames. The current model also omits
outward tipping and real foothold surface orientations.

Ranking is a project rule, not an association's coaching standard:

1. Contact, equilibrium, friction, joint bounds, motion limits, support/unloading
   checks and numerical convergence precede proxy improvement.
2. Compare uncertain-support, low-margin and uncertain-unloading counts.
3. For comparable reliability, compare hand-load integral, peak, arm-moment
   integral and then maximum angular step.

Differences are grouped into numerical tie bands of 0.005 BW·s, 0.005 BW and
0.05 BW·cm·s. These are comparison conventions, not physical measurement
confidence intervals. `candidate_comparison.csv` states the first ranking
criterion that separates each rejected alternative from the selected one.
All frames still retain geometry uncertainty even when the numerical gates pass.

The video shows intended phases inferred from the contact schedule: preparation,
load transfer / release preparation, hand reach or foot transfer, arrival and
stabilization. Per-frame explanations report actual computed load differences
or explicitly state that the frame has no meaningful distinction. The phase
labels do not claim that a human technique was recognized or that a movement
will work in the real world.

## Why The Way Up was not used for technique retraining here

[The Way Up dataset](https://zenodo.org/records/15196867) annotates hold locations,
usage order, use times and occlusion. The existing network predicts contacts and
timing; it has no supervised technique-quality, force, effort or target-body
success labels. Training a technique classifier on the planner's own unreviewed
preferences would not supply independent evidence of technique quality.

The saved review queue is a starting point for a later expert-labeled dataset:
label technique presence and phases, mark occlusion/ambiguity, compare candidate
pairs, and record whether contacts and movement outcomes are actually supported
by evidence. Split evaluation by person and route, and keep independent gym
footage for transfer testing. Report class accuracy and phase timing separately
from IK feasibility and proxy-score improvement. Contact-model retraining and
technique-policy learning are separate tasks; neither was run in this experiment.

## Teaching sources

- [CEC Athlete Development Matrix](https://climbingcanada.ca/wp-content/uploads/2021/09/CEC_Athlete-Development-Matrix-ADM-v1-2021.pdf), pp. 4–5 and 10–11: skill vocabulary and development progression, including weight transfer and flagging. This is not a mandatory per-move order.
- [Mountain Training Indoor Climbing Assistant, January 2025](https://www.mountaineering.ie/_files/Indoor-Climbing-Assistant-candidate-handbook-Jan25.pdf), p. 12, hosted by Mountaineering Ireland: body positioning and leg use as general movement principles.
- [BMC FUNdamentals guidance](https://services.thebmc.co.uk/media/files/Skills/FUNdamentals%20Docs/FUNdamentals%201%2C%202%20and%203%20-%20Guidance%20Notes.pdf), sections 4.2–4.5: balance, feet/hand relationships and context-dependent straight-arm positioning.
- [Alpine Club of Canada, Body Position](https://handbook.alpineclubofcanada.ca/docs/leadership-handbooks/top-rope-leader/rock-climbing-movement/#body-position): a general face-climbing movement sequence whose application depends on holds.

The candidate formulas, weights, comparison bands and force models are our
implementation decisions. These sources do not validate the numerical model.

## Reproducibility and retained uncertainty

Each run is addressed by its source/body/settings/code hashes. All exported
artifacts are hashed; changed inputs or code create a new output directory.
Partial outputs are not accepted as a completed cached run. Original results
are never overwritten. One historical Trial3 orchestration file differs from
the old Trial1 baseline's provenance; this is explicitly recorded, all source
evidence and other baseline dependencies are checked, and the saved baseline
trajectory is independently re-audited under the current comparison rules.

The limits of the preceding ground model remain: approximate floor regions,
image-up gravity, a uniform scale, automatic contacts and a fixed ankle/sole
offset. Singleton nominal support can pass while having no reserve. No new
ground support is inferred from `free`. Hooks, smearing, true hip rotation,
self-collision, dynamic momentum and fatigue remain outside this model.


## Measured 180 cm results (2026-11-04)

Selected: **weight_shift**. The final complete comparison took 247.2 seconds in this local run. All six newly fitted alternatives converged; the preserved baseline retains its historical convergence result and was independently re-audited.

| Rank | Candidate | Hand L1 integral (BW·s) | Arm-moment integral (BW·cm·s) | Near-zero hand-bound frames |
| --- | --- | ---: | ---: | ---: |
| 1 | `weight_shift` | 0.959144 | 53.1960 | 257 |
| 2 | `flag_assist` | 0.961845 | 53.5670 | 257 |
| 3 | `leg_drive` | 0.960797 | 53.5814 | 257 |
| 4 | `arm_economy` | 0.958902 | 53.7229 | 256 |
| 5 | `combined` | 0.959711 | 53.6869 | 256 |
| 6 | `preserved_baseline` | 1.081099 | 55.8596 | 249 |
| 7 | `control` | 1.081099 | 55.8596 | 249 |

The hand-load integrals of the five technique candidates fall in the same numerical comparison band. Weight shift wins the next differentiating criterion: the lower simplified arm-moment integral. It is not selected merely because its name follows a teaching preference. The `control` reproduces the preserved baseline's proxy metrics, showing that rerunning the solver alone does not produce the apparent improvement.

Relative to baseline, the selected hand-load lower-bound integral changes from **1.081099 to 0.959144 BW·s** (11.28% within this model), and the arm-moment proxy from **55.8596 to 53.1960 BW·cm·s** (4.77%). These percentages do not measure energy, endurance or human effort. Peak hand L1 load stays approximately **1.0 BW**. The lower bound is near zero on 257/417 selected frames, versus 249/417 baseline frames, which limits its ability to distinguish real techniques.

All candidates retain zero contact-error, nominal support-balance and pre-unload failure frames. They also retain **74 uncertain-support frames, 40 low-margin frames and 43 uncertain-unloading frames**. No evidence is upgraded and every frame retains geometry uncertainty. Maximum contact residual remains 0.080104 cm under the unchanged 0.1 cm tolerance. Maximum fixed-segment error is below 7 × 10⁻¹⁴ cm.

There is a motion tradeoff: selected maximum hip step increases from **4.034 to 4.751 cm/frame**, and maximum angle step from **11.725 to 13.804 degrees/frame**, still below the unchanged 5 cm / 15 degree warning limits. The selected average hip height after frame 102 is about **1.24 cm lower** than baseline; mean knee flexion increases slightly. Thus this run does not support a universal rule to raise the hips or straighten the legs first.

Source observations worth reviewing include left-hand H44→H28 (automatic 113–136), right-foot H65→H60 (153–173), left-foot H62→H51 (180–206), and right-foot H60→H45 (266–286). Their projected hip changes differ substantially. Ground-supported approach events and same-support recontacts are explicitly distinguished from new climbing moves. No flagging label was confirmed from these observations.

- [Selected complete video](../outputs/trial1/technique_plans/target_180cm_cf107c3b5dfe/selected_plan.mp4)
- [Phase-by-phase review and candidate rankings](../outputs/trial1/technique_plans/target_180cm_cf107c3b5dfe/review.html)
- [All candidate metrics](../outputs/trial1/technique_plans/target_180cm_cf107c3b5dfe/candidate_comparison.csv)
- [Unlabeled expert-review queue](../outputs/trial1/technique_plans/target_180cm_cf107c3b5dfe/technique_review_queue.csv)
- [Source evidence contact sheet](../outputs/trial1/technique_plans/target_180cm_cf107c3b5dfe/source_evidence.jpg)

Validation: 27 targeted tests pass. Fresh stationarity was independently recomputed for every newly fitted candidate. All stored fair-force solutions were checked for force/moment equilibrium, capacity, friction and zero forces on unknown contacts; fixed bones, unchanged labels, targets and offsets were verified. The final video decodes to 417 frames at 30 FPS. The new Trial1 section executes successfully in a fresh Jupyter kernel using the saved verified baseline. Review controls pass syntax and mock-DOM seek checks; these are not a browser-rendering test. The original notebook code/output cells are preserved.
