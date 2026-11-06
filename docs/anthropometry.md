# Mass and center-of-mass assumptions

These are population-based priors and explicit geometry proxies, not personal
measurements or validated climbing biomechanics. Height scales segment lengths;
it does not identify an individual's mass distribution. This change does not add
joint-range constraints, fix elbow swivel, or establish physical safety.

## Evidence and coordinate definitions

The numerical source checked here is [Xu & Zhou (2017), Table 1](https://journals.sagepub.com/doi/10.1177/1550147717703257),
which attributes its table to Dempster. The original report is cataloged as
[Dempster (1955), WADC TR 55-159](https://contrails.library.iit.edu/item/154630).
We have not revalidated these values against the original report or this climber.
[de Leva (1996)](https://doi.org/10.1016/0021-9290(95)00178-6) explains why anatomical
landmark definitions matter when applying segment parameters to joint centers;
this project is **not** implementing de Leva's parameter tables.

All positions use cm in one shared 2D or 3D coordinate frame. Limb masses are
per side. There are two arms and two legs; their counts must be included in
the whole-body mass sum.

| Segment in code | Fraction of body mass | COM rule and evidence |
| --- | --- | --- |
| `head` | .081 | Shoulder center + .40 times the hip-to-shoulder vector. **Project proxy**; .40 is not a published head COM coefficient. |
| `torso` | .497 = .355 + .142 | Midpoint of hip and shoulder centers. **Project proxy for the combined trunk/pelvis**; merging masses does not justify this COM position. |
| `upper_arm` | .028 each | Shoulder to elbow, fraction .436 from shoulder; coefficient matches the published table, mapped approximately to our joints. |
| `forearm` (includes hand) | .022 = .016 + .006 each | Legacy fraction .69 from elbow to wrist is an unverified project approximation. The alternative below explicitly merges component COMs. |
| `thigh` | .100 each | Hip to knee, fraction .433 from hip; coefficient matches the table, with the same landmark caveat. |
| `shank` (includes foot) | .061 = .0465 + .0145 each | Legacy fraction .57 from knee to ankle is an unverified project approximation. The alternative explicitly merges component COMs. |

## Versioned profiles

`anthropometry_profile()` returns parameters, source links, coordinate definitions,
and uncertainty notes as a JSON-serializable record. Returned dictionaries are
independent; modifying one does not change global defaults.

- `legacy_v1` remains the default for existing solvers and video pipelines. It
  preserves the previous numeric assumptions for reproducibility.
- `dempster_endpoint_proxy_v1` uses the published forearm/shank coefficients
  (.430/.433) and component mass fractions. With missing hand/foot orientation,
  it explicitly places hand COM at the wrist and foot COM at the ankle. This is
  an **alternative sensitivity assumption**, not a validated improvement.

For component positions `r_i` and masses `m_i`, the merged COM is
`sum(m_i * r_i) / sum(m_i)`. The endpoint assumption therefore gives:

```text
forearm+hand: (.016 * .430 + .006 * 1) / .022 = .585454545...
shank+foot:   (.0465 * .433 + .0145 * 1) / .061 = .567778689...
```

If actual hand or foot COM coordinates become available,
`combined_distal_center_of_mass(elbow, wrist, hand_com, "forearm")` and its
`"shank"` equivalent perform the weighted combination directly in 2D or 3D.
These are **COM coordinates**, not hold centers, finger contacts, or shoe contacts.
The helper does not infer orientation and does not automatically inject external
positions into a sequence solver. Such integration would also need consistent
pose derivatives and audit calculations.

## Use and validation

```python
from beta_translator.inverse_kinematics import (
    anthropometry_profile, solve_inverse_kinematics_with_gravity,
)

profile = anthropometry_profile("dempster_endpoint_proxy_v1")
result = solve_inverse_kinematics_with_gravity(
    body_lengths, contact_targets, reference_pose, mass_kg=70,
    mass_fractions=profile["mass_fractions"],
    com_fractions=profile["com_fractions"],
)
# Save the complete `profile` record with any result produced this way.
```

For a fixed-pose sensitivity comparison, both
`gravity_sequence.batch_com_and_jacobian(poses, lengths, com_fractions=...)` and
`ik3d.kinematics(poses, lengths, ..., com_fractions=...)` accept the same COM
coefficients. The 2D analytic Jacobian and the 3D complex-step derivatives remain
consistent with the selected coefficients.

Existing full-sequence optimizers and candidate rankings still use `legacy_v1`.
Reevaluating a saved pose's COM does not reoptimize contact forces or trajectories.
Before switching a production sequence to another profile, thread that profile
through **every** optimization, force audit, ranking metric, and export; persist
the profile in run provenance. Do not mix old forces/rankings with a changed COM
and present them as a new validated result. Existing code-hash cache checks will
reject reuse when the source changes; historical files are not overwritten.

Masses must be finite and positive; whole-body fractions (including both sides)
must sum to one. Joint-to-joint COM fractions must lie in [0, 1]. The head
coefficient is a different representation: a finite nonnegative torso-length
offset. Its validation is not a physiological range claim. Invalid/missing values
raise errors rather than silently being repaired. Weighted COM calculations retain
complex-valued perturbations used for differentiation.

Comparing two proxies gives a sensitivity result, not an uncertainty bound. It
cannot prove that candidate rankings are robust to every plausible body shape,
hand orientation, or torso/pelvis distribution.

## Fixed-pose Trial2 check

The existing 180 cm native-refinement trajectory under
`outputs/trial2/spatial_plans/target_180cm_5f9e7fcd448b_native/native_refinement/`
contains 976 frames. Recomputing its COM with the endpoint proxy, without changing
poses, moved whole-body COM by 0.09752 cm on average and at most 0.12174 cm
(frame 244). The default reproduced its stored 3D joints and COM to within
1e-10 cm. No video, force solution, candidate ranking, or historical file was
changed. This comparison does not test the head/torso approximations shared by
both profiles and does not resolve missing hand/foot orientation.
