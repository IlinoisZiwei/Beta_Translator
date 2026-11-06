# Personalized shared body and active mechanics

Trial1, Trial2 and Trial3 use one `BodyModel` and the editable
[`trial_user_175.json`](../configs/users/trial_user_175.json) profile. The default
demonstration person is **175 cm tall with a 176 cm fingertip-to-fingertip arm span**.
These are requested example values, not measurements of the filmed climber.
No previous result is overwritten.

## User inputs

| Dimension | Default | Meaning |
| --- | ---: | --- |
| Height / arm span | 175 / 176 cm | Standing stature / bilateral fingertip span |
| Shoulder / hip width | 40 / 30 cm | Joint-center separation |
| Upper arm / forearm | 30 / 24 cm | Shoulder–elbow / elbow–wrist |
| Hand length | 14 cm | Wrist to fingertip |
| Hand patch | 7 cm distal, 2 cm normal offset | Separate assumed grip point, not the fingertip |
| Thigh / shank | 43 / 42 cm | Nominal rigid bone lengths; knee translation remains coupled |
| Torso | 49 cm | Hip midpoint to shoulder midpoint |
| Foot length / width / sole depth | 25 / 9 / 3 cm | Shoe geometry proxies |
| Mass | 70 kg | Explicit demonstration assumption |

Arm span must agree within 0.1 cm with
`shoulder_width + 2 * (upper_arm + forearm + hand_length)`. Inconsistent entries
raise an error; the solver does not stretch a bone or silently change a measurement.
Lengths are symmetric on both sides. Height is not the sum of this reduced chain's
vertical dimensions. Body mass, segment fractions, COM, inertia and capability
parameters are separate inputs. Capacity triplets are X/Y/Z components in the parent-segment frame; they do not denote the three anatomical coordinate axes. Update the evidence strings when measurements
replace assumptions; this does not automatically certify physiological validity.

Copy the template to another JSON file and edit its values. Use either:

```python
from beta_translator.body_model import BodyModel
model = BodyModel(user_profile="configs/users/trial_user_175.json")
result = model.fit("trial2", frames="verification", iterations=150)
```

```sh
source .venv/bin/activate
python scripts/run_body_model.py --trial all \
  --user-profile configs/users/trial_user_175.json --frames verification
```

`--frames all` requests new full-sequence optimization. An explicit `--height-cm`
must agree with an explicit profile. Without a profile file, a height override
scales the demonstration geometry and is labeled an assumption. Old ratios remain
accepted through the Python API; their implied span is recorded. New notebooks
include a separate personalized-model section and retain all historical stages.
For partial measurements, start with the template and keep the remaining values
explicitly marked as assumptions; do not label the entire profile measured.

## What now participates in optimization and selection

| Mechanism | Active computation | Important boundary |
| --- | --- | --- |
| Pose-dependent shoulder range | Bilinear axial-rotation envelope over shoulder plane/elevation | Default table is a project screening assumption, not a MoBL physiological envelope |
| Self-collision | Closest distance between 14 segment capsules, testing every non-adjacent pair | Native-frame samples; exempt adjacent segments overlap by construction; no skin or swept-volume certificate |
| Joint moment capacity | Distal-segment Newton–Euler moment balance at 12 shoulder/elbow/wrist/hip/knee/ankle locations; directional ellipsoidal capacity and speed reduction | Net moments include passive-axis reactions; these are not 26 generalized actuator torques or a muscle model |
| Fatigue/capacity coupling | Causal active/resting/fatigued occupancy states; fatigued fraction reduces available joint capacity | Three-compartment-inspired actuator proxy; custom recruitment controller and uncalibrated rates, not measured endurance |
| Inertial balance | Segment translational acceleration, rotational inertia, angular acceleration and gyroscopic moment; sum external forces and moments about COM | Reduced rigid bodies with assumed mass/inertia; no impact simulation or forward replay |

All five contribute residuals to the **same trajectory optimization**, have
per-frame audits, and can reject a candidate. They are soft during fitting; a
failure is not repaired by post-processing. Existing hard coordinate bounds,
rate constraints, persistent contact regions and zero free-contact force remain.
The complete specification still reports unknown mechanisms and calibration.

The dynamic mode replaces the static force/moment objective. Its audit uses the
**actual optimized forces**, not a fresh LP allocation. Friction, matched patches,
force bounds, departing forces and dynamic balance must agree. The previous static
LP and ±2 cm reserve are retained as separately labeled diagnostics; the latter is
not a dynamic robustness certificate. Short clips use one-sided derivatives at
their boundaries and an explicit initial fatigue state; unobserved pre-clip effort
is not inferred. No continuous-time or subframe feasibility is claimed.

COM, inertia, joint moments, fatigue and rendering share the personalized segment
and hand/foot geometry. Dynamics use SI units; conversion from cm and body-weight
forces is explicit. Limb hand/foot masses are split from the former distal
aggregates, so mass is not counted twice. Uniform-cylinder inertia and a spherical
head are approximations. The torso/pelvis remains rigid.

`anatomical_control` uses the **same user profile and anatomical chain**, with the
previous fitting objectives. It is audited against the new checks for comparison
but cannot win selection. `balanced` and `steady` both optimize the new checks.
Ranking considers feasibility and failures before effort or smoothness; a best
diagnostic candidate is still labeled infeasible when appropriate.

## Ownership and evidence

- `body_user.py`: profile loading, validation and geometry.
- `body_constraints.py`: shared constraint inventory and enforcement declarations.
- `body_mechanics.py`: trial-independent collision, rigid-segment equations and fatigue.
- `body_personalized.py`: optimizer integration, sparse derivatives and audits.
- `body_pipeline.py`: common trial entry, immutable result paths and profile snapshots.

Source anatomical chains retain their existing pinned MoBL/Gait2392 provenance.
The fatigue design is informed by [Xia and Frey Law (2008)](https://pubmed.ncbi.nlm.nih.gov/18789445/);
the recruitment controller and numerical defaults here are explicitly project
choices, not parameters copied from that paper. The separation of centroidal and
joint dynamics follows the distinction documented in
[Drake's centroidal momentum constraint](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_centroidal_momentum_constraint.html)
and [inverse dynamics](https://drake.mit.edu/doxygen_cxx/classdrake_1_1systems_1_1controllers_1_1_inverse_dynamics.html).
Drake and OpenSim engines are not runtime dependencies or independent validators.

Joint-level muscle paths, loaded personal ROM, scapular motion, generalized
joint-dynamics residuals, hand grip wrench capacity, impacts and swept collision
remain outside the implemented reduction. `physical_validation` remains false.
Activating a numerical proxy does not manufacture calibrated human capabilities.

## Verification

Tests cover profile consistency, independent capsule distances, pose-dependent
limits, fatigue conservation/recovery/causality, static and free-fall equations,
mass scaling, all-trial residual derivatives, physical contact identity and saved
profile/audit exports. Numerical verification is separate from human validation.
The new comparison index is `configs/personalized_results.json`; its recorded
short-clip results are not full-video optimization results. See the accompanying
`docs/validation/personalized_verification.json` for executed checks and outcomes.

### Recorded short comparisons

All comparisons use the same 175 cm / 176 cm-span geometry and 35 iterations per candidate.

| Trial / frames | Dynamic-equation failure frames, control → active | Peak moment / assumed capacity, control → active | Contact-failure frames, control → active |
| --- | ---: | ---: | ---: |
| Trial1 / 113–136 | 24 → 0 | 8.02 → 1.69 | 16 → 21 |
| Trial2 / 550–595 | 43 → 0 | 7.53 → 1.83 | 11 → 43 |
| Trial3 / 296–331 | 32 → 0 | 7.96 → 2.15 | 9 → 28 |

Zero dynamic-equation failures does **not** mean successful support: contact,
friction, capacity and release gates still apply. All active clips exceed the
assumed moment capacities, retain other failures, and did not converge within
35 iterations. Trial2 collision failures increased from 1 to 8 frames. These
results demonstrate the active tradeoffs, not universal motion improvement.
All selected outputs remain diagnostic. The current checks passed 263 automated
tests; all 69 notebook code cells executed without errors.
