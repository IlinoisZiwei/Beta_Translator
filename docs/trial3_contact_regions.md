# Trial3: contact regions and wrist/ankle offsets

Section 13 of [Trial3 Analysis](../Trial3_Analysis.ipynb) compares complete
**180 cm, 0–502 native-frame** spatial trajectories. It adds evidence-bounded
contact locations and explicit joint-to-contact connectors to the conditional
3D camera/wall model from Section 12. All historical studies remain available.

## Candidate definitions

| Candidate | Surface contact | Wrist/ankle connector |
| --- | --- | --- |
| `center` | Existing hold center, fixed for the episode | Zero; the previous point-center approximation |
| `evidence_point` | Source-derived point inside an observed mask | Zero |
| `region` | Optimized within a conservative evidence patch | Zero |
| `region_offset_half` | Same evidence patch | Half of the illustrative body-scaled connector |
| `region_offset` | Same evidence patch | Full illustrative body-scaled connector |

These are **contact-representation hypotheses**, not five learned climbing
techniques. They keep identical target height/ratios, native frames, raw contact
IDs/confidences, camera, gravity assumption, optimizer settings and audit limits.
The contact constraints intentionally differ between the ablations; a better
score must not be attributed solely to a stronger optimizer. No new foothold
sequence, expert skill label or The Way Up network is trained in this step.

## Image evidence and region topology

The existing RTMW fingertip and toe/heel observations are mapped to the wall
reference image. Raw scores above 0.5 and points within 12 reference pixels of
the assigned hold provide **unreviewed proposals**. Scores are not calibrated
probabilities. Three observations are required to permit a region search.
Source height is unknown; source pixel distances are not target body lengths.

Each continuous limb/hold episode receives one anchor derived from the source
observations, independently of target-body feasibility. An axis-aligned rectangle
grows inside the observed mask after one-pixel erosion where available, with a
24-pixel limit per edge. Every pixel in the rectangle must belong to the mask;
concave gaps and holes are not filled by a convex hull. Degenerate thin regions
stay thin. The rectangle is deliberately more conservative than the entire hold.
Missing source evidence would freeze the point instead of granting extra freedom.

Two bounded coordinates choose a point inside this rectangle. One pair of
coordinates is shared by **every frame in the same contact episode**. The
optimizer therefore cannot slide a grip around the hold each frame to repair
reach errors. A soft six-centimeter evidence prior discourages unnecessary
movement, normalized by episode duration. This prior is a modeling choice,
not measured uncertainty. All chosen points, evidence counts and changes from
the center approximation are exported for review.

## A contact is not a wrist or ankle

The hand/foot touches the surface at `contact`; the two-bone limb terminates at
`contact + offset`. The audit checks `joint_endpoint - offset - contact` against
the unchanged 0.1 cm numerical threshold. Quasi-static forces and moments act
at **the surface point**, not at the shifted wrist/ankle. Unmatched contacts are
excluded from independent force feasibility checks.

The source observations suggest only a planar connector direction. Its magnitude
uses explicit, unmeasured defaults: hands 0.035 × target height in the wall plane
plus 0.012 × height outward; feet 0.045 × height in-plane plus 0.025 × height
outward. The half-offset candidate tests sensitivity. This does not calibrate
real palm length, shoe size, hold thickness or anatomical orientation. Connector
lengths/directions remain fixed within an episode; articulated hands/feet,
contact torques, grip shape and collision are not modeled. The virtual connector
has no separately modeled mass; the existing approximate segment mass model is
retained. These choices are a prototype to review, not anthropometric facts.

## Whole-trajectory solve and timing conflicts

Every candidate is fitted against **all 503 native frames together**. A shared
piecewise-linear pose-control basis uses a knot every four frames plus contact
transition neighbors. Every native frame contributes observation, contact,
geometry and temporal residuals. Convergence is stationarity of this declared
finite control basis, not of an unrestricted per-frame or global optimum.
The previous complete 180 cm 2D trajectory supplies an initial guess only; the
spatial trajectory is solved again. No local candidate winners are concatenated.

Known-contact limbs normally use exact two-bone geometry with elbow/knee swivel.
Direct known-hold-to-different-known-hold label changes have no observed free
interval. Within six frames before and after such a change, the solver uses
ordinary spatial limb angles so the endpoint can miss the target rather than
being forced to teleport. **Raw labels, target penalties and contact audits are
unchanged.** A missed target stays a reported failure and receives no support.
This is not a silently invented release time or a newly confirmed contact.

Joint-position velocity/acceleration terms act on actual XYZ coordinates.
Soft optimization buffers are 6 cm per native frame and 3 cm per frame squared;
reported warning limits remain 8 cm and 4 cm respectively at the native 30 FPS.
These are numerical continuity limits, not validated human speed limits.
Fixed bone lengths, 155° bend approximation, wall penetration, mask membership
and camera visibility are checked independently. Collision and full anatomical
joint limits remain outside scope.

Sparse damped Gauss–Newton, a bounded L-BFGS fallback and a final direct
refinement retain all termination information. A fresh normalized projected
gradient and cost stabilization are required for convergence. Iteration limits
and stalls are displayed as nonconvergence, even if the video looks plausible.

## Native-frame refinement

A sixth candidate, `region_offset_native`, converts the complete region-offset
trajectory into physical segment directions and refines **all 503 poses
independently but jointly in time**. All limbs use smooth two-segment angle
kinematics; contact errors are explicit penalties rather than analytically
clipped reach circles. Bone lengths stay exact, contact points remain shared
by episode, and raw labels and independent audit limits remain unchanged.
A weak angle prior anchors this refinement to its complete starting trajectory.

This stage has more degrees of freedom and extra solver time. It is an explicit
refinement, not a sixth equal-budget contact-model ablation. Both its parent
comparison and its final result are preserved. The final ranking includes all
six complete candidates, so additional complexity does not guarantee selection.

## Comparison, selection and explanation

The same independent six-equation force LP checks every candidate using contact
positions, matched-contact masks and the existing confidence/grid criteria.
The primary scenario is an **assumed 15° tilt and a 25 cm hip-depth prior**.
Independent 0°/15°/30° audits of the selected geometry show gravity sensitivity;
this does not identify the true wall angle. Foot-normal and strength/friction
assumptions remain those of the earlier spatial experiment.

Ranking compares whole trajectories in this order: geometric-failure frames,
continuity-warning frames, nonconvergence, conditional-balance failures,
pre-release failures, missing ±2 cm reserve, maximum contact gap and offset
magnitude. This is a declared project rule, not a coaching standard. The first
different metric is saved as each rejected candidate's explanation. A winner
with remaining failures is called the **least-failing tested candidate**, never
a physically validated climb. Confidence, geometry and contact-region uncertainty
remain visible on all frames.

`phase_explanations.csv` derives phases from the preserved contact schedule:
maintain/reposition, transfer or unresolved limb, and pre-release support check.
It does not infer muscular effort or assert that a free limb is flagging.
The final video contains only the selected whole candidate; the separate
comparison video and offline orbit viewer expose the alternatives.

## Run and inspect

```sh
.venv/bin/python scripts/run_trial3_contact_regions.py --height-cm 180
```

The existing environment is sufficient; no model download is added. Height
and proportions in notebook Section 1 are forwarded to Section 13. Source
evidence is specific to Trial3, not an arbitrary-video input interface.

Outputs are saved under `outputs/trial3/contact_region_plans/` using input,
code and configuration hashes. Complete cached runs verify every artifact hash.
Partial runs are preserved and rejected as completed caches. Old outputs are
never overwritten. The main deliverables are `selected_plan.mp4`,
`comparison.mp4`, `viewer.html`, `candidate_comparison.csv`,
`contact_episodes.json`, per-candidate trajectories and limb/frame audits,
`phase_explanations.csv`, and `angle_sensitivity.csv`. Videos retain all native
frames at 30 FPS. Purple connectors distinguish surface points from joint
centers; red indicates failure and amber indicates unmeasured assumptions.

## Measured 180 cm results

The final selected candidate is **region_offset_native**. Both the
[five-candidate parent study](../outputs/trial3/contact_region_plans/target_180cm_ea6fec26c0a9/)
and the [six-candidate final comparison](../outputs/trial3/contact_region_plans/target_180cm_9cd2cf3084ba_native/)
are preserved. The final [video](../outputs/trial3/contact_region_plans/target_180cm_9cd2cf3084ba_native/selected_plan.mp4)
and [offline viewer](../outputs/trial3/contact_region_plans/target_180cm_9cd2cf3084ba_native/viewer.html)
contain every native frame from 0 through 502.

| Candidate | Contact-failure frames | Maximum gap (cm) | Continuity-warning frames | Conditional balance frames, 15° | Pre-release failure frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| center | 133 | 16.075 | 5 | 110 / 503 | 72 |
| evidence_point | 129 | 17.999 | 3 | 113 / 503 | 72 |
| region | 46 | 12.588 | 3 | 140 / 503 | 72 |
| region_offset_half | 43 | 12.567 | 3 | 139 / 503 | 70 |
| region_offset | 39 | 12.635 | 3 | 156 / 503 | 68 |
| region_offset_native | 26 | 12.627 | 2 | 162 / 503 | 69 |

The selected candidate has 26 geometric/contact-failure frames, versus 39 for
its coarser region-offset parent and 133 for the new continuous center baseline.
This comparison is against this experiment's baseline, not the older 2D
98-failure run. Native refinement adds DOFs and computation. The whole-plan
ranking first distinguishes the selected candidate by its geometric-failure
count; local per-move winners were never concatenated.

**All six candidates remain numerically unconverged.** The selected candidate's
fresh normalized projected gradient is 0.355902, above the 1e-5 tolerance.
It is exported as an unresolved candidate, not relabeled as converged. Nominal
conditional balance passes 162/503 frames at the assumed 15° tilt; 69 of 75
pre-release check frames fail. These windows overlap and are not independent
climbing attempts. All 503 frames retain geometry/contact-region/offset uncertainty.

At frame 486, the numerical contact gap decreases from 11.1304 cm to
0.000254 cm. That submillimeter residual is optimizer agreement, not measurement
accuracy. Conditional force balance still fails at that frame. Across the
selected full trajectory, exact limb segment-length errors remain below 1e-10 cm.
The largest remaining gap is 12.6275 cm, so the late-frame example must not stand
in for complete success.

Remaining contact failures occur at **122–129, 255–264 and 375–382**. The first
and third intervals overlap direct known-hold switches. The middle interval
retains a combined pose/contact fit failure; this experiment does not uniquely
separate bad evidence, a restrictive region/body assumption and local solver
failure. Joint-step warnings remain at frames **126 and 379** (maximum 9.002 cm
against the 8 cm warning limit). No labels were silently changed to erase them.

For the selected episode contacts and fixed connectors, the triangle inequality
gives these conditional timing bounds. If two consecutive endpoints must stay
within 8 cm, at least one must miss its assigned target by (target distance − 8)/2:

| Switch frame | Limb | Consecutive target distance (cm) | Minimum one-frame gap at the 8 cm limit (cm) |
| --- | --- | ---: | ---: |
| 379 | left_hand | 29.536 | 10.768 |
| 126 | right_hand | 34.256 | 13.128 |
| 355 | right_hand | 2.324 | 0.000 |

These bounds concern the chosen model targets, not human capability. Reviewing
release/arrival timing and grip positions is more defensible than deleting
contact constraints merely to obtain an all-green video.

For exactly the same selected geometry, changing assumed tilt to 0°/15°/30°
gives 302/162/17 conditional balance frames, respectively. Eligible-evidence
counts are 179/82/17. This sensitivity prevents selection of a true angle from
these results; surface normals, grip capacity and wall geometry remain unmeasured.

The executed parent study plus native refinement took about **26.9 minutes**
on this machine; exploratory development runs are excluded. An intact cached
run loads the stored results. The first complete rerun after a relevant change
can therefore take tens of minutes. No additional environment or pretrained
weight download was required.

Validation includes 28 targeted tests, fresh-kernel execution of the added
notebook section, all native-frame/label/confidence invariants, shared episode
contacts, mask-region bounds, independent reconstruction and gradients, exact
bone lengths, decoded 503-frame/30-FPS videos, and offline viewer/GLB structure.
Previous notebook cells, videos, source evidence and historical results remain
preserved. No commit or push was performed.
