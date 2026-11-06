# Trial2 Stage 1: anatomical-chain verification and consistent contacts

This experiment uses a **180 cm** body and the existing Trial2 wall hypothesis
(middle panel 15°, upper panel 0°). It first compares the previously identified
right-elbow anomaly, frames **550–595** at 30 fps. Earlier notebooks, videos and
numerical experiments remain available. This is a short-segment experiment;
it does not establish full-sequence or human feasibility.

## What changed

### Source-checked rotational chain

The source is the [pinned MoBL-derived OpenSim snapshot](https://github.com/aikkala/O2MConverter/blob/ad2599d5ecb2c18b44570b4d0b53e062ac9b82ae/models/opensim/MoBL_ARMS_OpenSim_tutorial_33/MoBL_ARMS_model_for_mujoco.osim).
Its SHA-256 is `2dc242259e0f5004d935646a424af7f1fd4ba99d79ff05b3a7b1f2e605949dad`.
[The extracted numeric reference](../data/trial2_stage1_chain_reference.json)
records axes, parent/child joints, rest transforms and coordinate couplings.
The source has been checked against that reference, rather than assuming the
previous constants fully reproduce the source hand orientation.

The old shoulder and elbow rotation composition agrees with the selected
rotational source chain. Its wrist was a reduction: all flexion was assigned
to one axis and the distal wrist couplings were omitted. Stage 1 restores:

```text
shoulder plane → elevation → axial coordinate
→ signed anatomical elbow hinge
→ radioulnar pronation/supination
→ radiocarpal deviation and partial flexion
→ coupled distal wrist deviation and remaining flexion → hand frame
```

The two/three-point source splines are evaluated with the corresponding
linear/quadratic special cases and endpoint-tangent extrapolation. These cases
were checked against [OpenSim 4.5 SimmSpline's coefficient rules](https://github.com/opensim-org/opensim-core/blob/4.5/OpenSim/Common/SimmSpline.cpp).
Unsupported larger spline tables raise an error.

An independent SciPy matrix composition tests 128 deterministic poses against
the extracted source, including rest-frame checks. The maximum matrix entry
error is about `1.6e-15`. The old hand orientation differs by up to **25.41°**
on this test set; that is a model-comparison result, not a measured video error.
Additional tests check both elbow hinge axes, reflection, fixed lengths,
distal-joint isolation and derivatives.

**Remaining reduction:** the project retains its existing segment lengths and
joint centers. Source translational bone offsets, shoulder-girdle motion and
scapular rhythm are recorded as omissions. This is not a full MoBL import or
an OpenSim-engine cross-validation. Existing ROM and sampled temporal caps are
retained; posture-dependent shoulder envelopes remain Stage 2 work.

### One contact definition

Previously, the wrist was strongly fitted to `surface + fixed_offset`, while
a separately articulated palm point was weakly fitted to the surface. The
two targets could disagree, yet support eligibility used the wrist target.

Stage 1 uses the same physical proxy throughout:

```text
patch(q) = wrist_or_ankle(q) + segment_frame(q) @ local_patch
contact_residual = patch(q) - selected_surface_point
equivalent_joint_target = selected_surface_point - (patch(q) - joint(q))
```

The equivalent joint target is derived from that pose, not an additional
fixed-offset objective. Optimization, gap/force coupling, independent support
audits, saved metrics and video connectors all use the same patch. The
independent tolerance remains **0.1 cm**. A wrist-target pass cannot create
support when its physical hand patch misses the surface.

Hands retain the assumed 0.035 × height tangential and 0.012 × height normal
connector dimensions, now oriented by the corrected wrist chain. Detailed
finger closure, hold grip direction and measured palm orientation remain absent.

### Explicit conditional foot frames

Each stance episode has a right-handed frame: toe direction, transverse
direction and outward surface normal. Toe direction uses the existing
ankle/contact evidence; ground proposals use visible heel/toe image evidence.
Sparse evidence produces an explicitly reported tangent fallback.

- Wall contacts use a local forefoot vector `(0.045H, 0, −0.025H)`.
- Ground contacts use a sole-under-ankle vector `(0, 0, −0.035H)`.
- The frame is shared throughout the episode. No independently optimized
  offset can move the foot contact away from its displayed geometry.
- Unknown/free contacts retain exactly zero optimized forces.

These dimensions and surface-aligned orientations are **hypotheses**, not
measured shoe geometry or a validated ankle model. Ankle/subtalar ROM,
non-point sole pressure and full shoe collision are not checked. Free feet
are not assigned support merely because they are off the wall. Existing
ground proposals still require the earlier positive floor evidence.

Direct hold-to-hold label changes without a free frame are retained and
flagged for timing review. No contact labels, target-body lengths or old
results are silently changed to obtain a better score.

## Controlled comparison and interpretation

`legacy_control`, `stage1_balanced` and `stage1_steady` start from identical
body coordinates and force initialization. They use the same observations,
contact episodes, assumed wall geometry and iteration budget. The two Stage 1
candidates differ only in their declared smoothness weight.

The report preserves each model's original audit and adds a **common physical
patch audit**. For the control, the actual old articulated palm and explicit
foot proxy are evaluated at the same 0.1 cm threshold. This avoids comparing
an old wrist-target score with a new palm-target score as though they were
the same measurement. Support is independently recomputed using only the
patches that reach their surfaces.

All anatomical, rate, acceleration, contact, support, release and convergence
failures remain visible. Candidate selection retains a feasibility gate; an
unsuccessful solver cannot yield a feasible label. A diagnostic winner is
only the least-violating tested Stage 1 candidate under the documented policy.
Unknown ankle mechanics, geometry and wrist orientation remain explicit even
when numerical contact checks pass.

## Run and inspect

Open **Section 19 of Trial2_Analysis.ipynb** to see the saved source verification,
comparison table, selection reasons, contact plots and 3D video. Recomputing
is opt-in. The new section can run independently from the repository root.

```sh
.venv/bin/python scripts/run_trial2_stage1.py --height-cm 180 --clip right_elbow --iterations 250
```

Other predefined clip names are `hinge_transition` and `left_elbow`. Each run
gets a content-derived folder under `outputs/trial2/stage1_contacts/`; existing
files are verified before reuse and incomplete experiments are not overwritten.
No new environment or package is required.

To verify a downloaded original snapshot against the stored numeric reference:

```sh
.venv/bin/python scripts/verify_trial2_chain_source.py /path/to/MoBL_source.osim
```

The source snapshot is not required to view results or run the optimizer.
Its reference values and source URL/hash are already stored in the repository.

The next anatomical step is a calibrated ankle/foot chain, coupled shoulder
ROM and collision coverage. Consistent contact bookkeeping does not identify
the true wall geometry, fix incorrect contact timing or certify the movement.
