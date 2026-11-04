# Trial1: convergent whole-sequence refinement

Open [Trial1_Analysis.ipynb](../Trial1_Analysis.ipynb) with **Python (Beta Translator)**.
Set `TARGET_HEIGHT_CM` in Section 1 and run through Section 8 in order.
The default is **180 cm, assumed 70 kg, foot friction 0.8**, covering all
**417 native frames (0–416)**. Alternatively, run from the repository:

```sh
.venv/bin/python scripts/run_trial1_refinement.py --height-cm 180 --mass-kg 70
```

Results are saved separately under `outputs/trial1/refined_support/`. Existing
annotations, predictions and videos are preserved. The default run
uses the previous 180 cm ground-support poses as an initialization and jointly
refits every frame. It does not concatenate old clips or choose frames from
different solutions. A changed body, force model or ground schedule triggers a
fresh gravity initialization; the review page's previous video remains clearly
labeled as the saved 180 cm ground run.

## What changed

The previous solve reached the evaluation limit in each frozen IRLS fit, with
substantial remaining gradient. This workflow uses a sparse-direct damped
Gauss–Newton step with active bounds and actual cost-decrease checks. It solves
the full temporal system, including acceleration regularization; the sparse
linear solve is not a set of independent frame solutions. Column normalization
reduces the effect of mixed centimeter, radian and body-weight force units.

Convergence requires a normalized projected gradient at most `1e-5` and relative
cost stabilization at `1e-9` (or exact projected stationarity). The projected
gradient respects active force/joint bounds and explicitly fixed coordinates.
The outer IRLS loop additionally requires relative weight change at most `1e-5`
and stationarity of the **refreshed**, not merely frozen, objective. Evaluation
limits and stalled nonstationary solves remain failures. Full histories and
termination reasons are retained. This is a first-order convergence check, not
proof of a global optimum.

Some remaining contacts occupy the same point in the 2D model, notably both
hands on H44. With those contacts alone, exact nominal equilibrium requires the
COM horizontal coordinate to equal that point. Soft interval penalties leave a
small discrepancy, while the independent force LP correctly rejects it.

The new solver therefore uses **COM-x as an optimization coordinate** and derives
hip-x from it and the joint angles. True singleton nominal support intervals
are enforced in this coordinate throughout optimization. All contact, force and
temporal derivatives include the coordinate transformation. This is not a
post-export adjustment of the reported COM. Finite narrow intervals whose robust
targets were collapsed are not mistaken for exact equalities. Conflicting point
equalities are recorded and left unresolved.

Exact point balance is fragile: it cannot tolerate the unchanged ±2 cm COM
perturbation. The audit still flags **low margin**, uncertain contact and uncertain
release timing. Eliminating nominal failure does not turn such frames into robust
or physically validated success.

A soft contact-restoration term starts at 80% of the existing 0.1 cm contact
tolerance, encouraging a small buffer. It does not relax the tolerance or remove
unreachable targets. Source labels, confidence scores, ground evidence, contact
timing, segment lengths, force limits, friction assumptions and independent audit
thresholds remain unchanged. Unknown supports still get exactly zero force.

## Controls and output checks

`REFINEMENT_SETTINGS` controls this solver's maximum inner/outer iterations and
convergence thresholds. The defaults allow up to 6 outer iterations and 2,000
inner iterations each. The legacy support objective supplies its weights; its
old `rounds`/`max_nfev` fields do not control this separate solver.

- `comparison.csv`: initialization versus refinement with identical evidence and
  audit settings. For the default configuration, initialization is the previous
  published ground-support trajectory.
- `review.html`: synchronized source, refined video and preserved previous video.
  Its table always describes the new run.
- `convergence.json` and `solver_history.csv`: convergence criteria, independent
  fresh gradient evaluation, accepted-step costs and all outer-loop outcomes.
- `singleton_com_constraints.csv`: each exact nominal COM constraint, supporting
  limbs, confidence group, event and robustness limitation.
- `ground_evidence.csv`, `trajectory.npz`, frame/limb diagnostics and pre-unload
  tables: complete evidence and numerical checks, including continuing warnings.

The existing ground model remains an uncalibrated projected 2D approximation.
It uses assistant visual interval assessment, approximate heel/toe source points,
an assumed ankle-to-sole offset, image-up gravity and an assumed friction model.
No new floor support is inferred from `free`. Inertia, depth, slipping, collision,
foot rotation and real grip strength remain outside this model.

## Saved 180 cm run (2026-11-03)

| Check | Previous ground run | Refined run |
| --- | ---: | ---: |
| Final numerical convergence | No | Yes |
| Contact-error frames (>0.1 cm) | 5 | 0 |
| Pre-unload failure frames | 6 | 0 |
| Nominal support-balance failure frames | 18 | 0 |
| Low support-margin frames | 0 | 40 |
| Uncertain-support frames | 96 | 74 |
| Uncertain pre-unload frames | 37 | 43 |
| Max contact gap (cm) | 0.175493 | 0.080104 |
| Max hip step (cm/frame) | 4.75267 | 4.03358 |
| Max angle step (degrees/frame) | 14.2898 | 11.7247 |

The final run has 303 nominal/robust model-check passes, 74 uncertain-support
frames and 40 low-margin frames. The latter are frames **12–46 and 48–52**,
where singleton COM constraints are enforced. In particular, the six former
pre-unload failures (12–17) become **uncertain timing with fragile nominal
support**, not robust unloading success. The 18 former balance failures (18–35)
become **low margin**, not robust balance passes. No source evidence is upgraded.

All 417 frames retain geometry/observation uncertainty. Combined numerical checks
pass on 287 frames, which does not certify any frame as physically validated.
The existing ground evidence still supplies only 48 conditional limb-frames
across 45 video frames, with zero geometrically eligible ground contacts.

Three outer iterations use 2,701 objective evaluations in total. The first inner
fit reaches its iteration limit; later fits satisfy the checks, and outer
convergence is established only at the end. A fresh final evaluation yields
normalized projected gradient **2.48981e-7**, relative IRLS weight change
**3.79729e-7**, and final relative cost change **8.83416e-12**. All are below
the explicit tolerances. Maximum exact COM-coordinate residual is 5.69e-14 cm.
Do not compare this normalized constrained gradient numerically with the old
solver's unnormalized gradient; they use different coordinates and conventions.

Validation includes 76 related tests, finite-difference checks of the global
Jacobian and final objective gradient (including a step-size study near penalty
hinges), independent recomputation of 760 feasible force-audit solutions, fixed
segment lengths, unchanged evidence/thresholds and zero unknown-support forces.
The complete video decodes to 417 frames at 30 FPS. Review controls use a mock-DOM
test; browser rendering has not been tested.
