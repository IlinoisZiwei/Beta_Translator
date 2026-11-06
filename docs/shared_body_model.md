# Shared body model

**Personalized-model update (November 5, 2026):** The shared entry now reads an editable 175 cm / 176 cm arm-span user profile and actively fits/audits shoulder envelopes, capsule collision, joint-moment capacity, fatigue coupling and inertial balance. Defaults are uncalibrated assumptions. Earlier descriptions and results below remain historical; see [current implementation and boundaries](personalized_body.md).

**November 5 update:** The default shared body now has 32 coordinates and an anatomical hip/knee/ankle/subtalar chain. See [the lower-limb guide](anatomical_lower_limbs.md). The 180 cm results and legacy-model descriptions below are preserved as historical records; current notebooks load the separate 175 cm comparison index.

Trial1, Trial2 and Trial3 now call the same reduced anatomical body and constraint implementation. Scene geometry, reviewed observations, contact timing and historical initialization are supplied by separate JSON configs. Historical results are read-only inputs; a new run creates its own content-addressed directory.

This implements a reusable pipeline and a complete **coverage inventory** for [the specification](body_constraint_specification.md). It does **not** complete the specification's proposed full physiological model. Every run records the missing mechanisms as `unknown`; parameterized audit equations alone do not supply those mechanisms.

## Run

Activate the repository's existing environment, then run:

```sh
source .venv/bin/activate
python scripts/run_body_model.py --trial all --height-cm 180 --frames verification
```

For one full native sequence, use `--trial trial2 --frames all`. Adjust `--iterations` for the optimization budget. Iteration-limit exits remain failures to converge. Use `--output-root` for an alternate destination. `--no-render` skips presentation only.

The same public API works in a notebook:

```python
from beta_translator.body_model import BodyModel
model = BodyModel(height_cm=180)
result = model.fit("trial2", project=ROOT, frames="verification")
# Other options: frames="all", frames=[550, 595], ratios={...} on BodyModel.
```

The active trial notebooks now restore source skeleton tracking, wall/hold review, BEFORE/AFTER annotation and historical experiments before the current shared-model sections. Trial1 has 20 sections, Trial2 24, and Trial3 22. Their default flow displays saved results; two explicit run flags request new shared-model fitting. See [complete workflow instructions](complete_trial_notebooks.md).

## Code ownership

| File | Responsibility |
| --- | --- |
| `beta_translator/body_model.py` | Public `BodyModel` entry point and height/proportion settings. |
| `beta_translator/body_constraints.py` | Shared ROM, rate caps, numerical tolerances, enforcement types and complete coverage inventory. |
| `beta_translator/body_kinematics.py`, `body_reference.py` | Fixed-length body, anatomical arm axes, pinned-source rotational coupling and verification. |
| `beta_translator/body_contacts.py`, `body_regions.py` | Persistent contact regions, physical hand/foot patch geometry and conditional foot frames. |
| `beta_translator/body_optimizer.py`, `body_articulation.py` | Shared trajectory objective, hard sampled bounds and consistent physical contact formulation. |
| `beta_translator/body_support.py`, `body_report.py` | Independent quasi-static support, coverage, release checks and candidate ordering. |
| `beta_translator/body_checks.py` | Parameter-dependent shoulder-envelope, SO(3) rate, capsule, strength, fatigue and dynamics audits. These are not enabled physiological models. |
| `beta_translator/trials.py`, `body_scenes.py` | Evidence/geometry adapters; no duplicated anatomical constraints. |
| `beta_translator/body_pipeline.py`, `body_render.py`, `body_notebook.py` | Common fitting/export, fixed-camera presentation and concise notebook displays. |
| `configs/trials/trial1.json`, `trial2.json`, `trial3.json` | Separate trial input paths, scene hypotheses, baseline and verification interval. |
| `scripts/run_body_model.py` | Single command-line entry point for one or all trials. |

The historical `anatomical_arms`, `trial2_anatomy_optimizer` and `trial2_stage1` modules are thin compatibility reexports. Generic support, contact-region and drawing functions were also extracted from Trial2 modules. Existing scripts keep working without duplicate implementations. Trial-specific legacy 2D/3D solvers remain because their historical experiments and notebooks still depend on them.

## Implemented scope

- One fixed-bone 28-coordinate body: floating hip position, rigid torso orientation, seven coordinates per arm, and legacy direction-based legs.
- Source-checked shoulder/elbow/forearm/wrist rotations; scalar arm ROM and sampled coordinate speed/acceleration bounds. The full MoBL translations and shoulder girdle are not imported.
- One contact definition throughout optimization, auditing and rendering: joint origin plus oriented patch offset equals the selected surface point. Hand patches use the wrist chain; foot frames are conditional surface/image hypotheses.
- Shared episode contact coordinates, zero force for free/unknown contacts, bounded force components, assumed foot friction, force/moment penalties and independent reached-contact LP.
- Remaining-support release checks plus an explicit 0.01 BW departing-force audit on the final pre-release frame; trusted-support and four-perturbation reserve reports.
- Sampled penetration/motion checks, continuous-interval fitting, per-frame failure markings and deterministic candidate reasons.

The fitted model uses cm, radians, seconds and body-weight-normalized forces. `body_checks.py` uses SI quantities where its names/docstrings specify them. A caller must supply calibrated capacity, fatigue, inertia and external-wrench inputs; the module does not manufacture them from height.

## What remains unknown

Full anatomical shoulder-girdle and lower-body chains; calibrated posture-dependent shoulder/wrist envelopes; measured grip/sole orientation and mode-specific wrench capacity; complete body/environment and swept collision; personalized strength and muscle paths; coupled muscle-fatigue capacity; inertial dynamics, impacts and forward replay remain unvalidated or unimplemented. The generic checks in `body_checks.py` test supplied quantities only. They are not wired into the fitting objective and do not turn a partial skeleton into a full dynamic model.

Trial1's depth is unobserved and its floor is not a calibrated metric plane. Trial2 uses joined-panel hypotheses. Trial3 uses a single-plane tilt hypothesis and an arbitrary vertical origin, not a measured floor. Raw contact observations remain separate from effective conditional ground proposals. None of these assumptions may be reported as recovered physical truth.

## Selection and verification

Each candidate covers the whole requested interval. Numerical feasibility and anatomy/physical failure counts precede trusted-support/reserve comparisons, convergence and motion preferences. Contact errors already below 0.1 cm share the same contact-ranking band. Hand-load preference applies only among numerically feasible candidates; it is not a fatigue estimate. No local winners are stitched and no Cartesian filter is applied after fitting.

`constraints.csv` contains a result for every declared specification category. `fail` means an implemented required check failed; `unknown` means needed evidence/model coverage is missing; `conditional_pass` is possible only when all required entries have results. No software-only result is labeled `validated_in_scope`.

Verification includes legacy compatibility tests, independent finite-difference Jacobian checks through all three adapters, fixed-source chain tests, contact-identity tests and analytic static/free-flight/fatigue/collision cases. Actual 180 cm short runs are recorded in `configs/verification_results.json`; full native input counts are checked separately. A short-run result does not certify an optimized full video.

## Preserved history

`notebooks/history/pre_shared_body_20261104.zip` contains exact pre-refactor notebook bytes, under renamed filenames that will not overwrite active notebooks. The active notebooks now restore the review widgets and stage displays. Extract the archive into the repository root for exact historical implementation/settings. All old videos, trajectories, exports and numerical results stay at their original paths. `Beta_Translator.ipynb` is untouched.

## Recorded 180 cm verification

All 242 regression tests passed. Each active notebook executed its nine code cells in order with zero errors. The same CLI successfully reloads all three verified runs. Full native geometry/contact consistency was checked for 417 + 976 + 503 = 1,896 frames; the numerical fits below are short intervals, not new full-sequence optimizations.

| Trial | Short interval | Selected | Contact failures | Anatomy/rate failures | Support failures | Release failures | Converged |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| trial1 | 113–136 | steady | 0 | 0 | 23/24 | 0/0 | No |
| trial2 | 550–595 | steady | 0 | 0 | 22/46 | 12/12 | No |
| trial3 | 296–331 | steady | 0 | 0 | 20/36 | 6/6 | No |

Release 0/0 means no applicable release frames. All selected candidates remain **diagnostic only**, and 24 specification categories remain unknown in these runs. Elbow-swivel diagnostics must be read with `elbow_swivel_observable`; legacy zero placeholders on masked/initial intervals are not evidence of zero physical rotation. True segment-rate coverage remains separately unknown without justified caps.

All 1,558 pre-existing output files and `Beta_Translator.ipynb` passed hash-preservation checks. Exact copies of the three old notebooks are retained in the history archive. See [machine-readable verification](validation/shared_body_verification.json) and [native-frame invariant checks](validation/shared_body_native_frames.json).

The public `BodyModel()` default now loads the 175 cm / 176 cm-span demonstration profile. Explicit height-only construction scales that profile. Use an explicit JSON for personal dimensions and capabilities. Historical commands/results above keep their recorded geometry; their numerical outcomes must not be attributed to the new profile.
