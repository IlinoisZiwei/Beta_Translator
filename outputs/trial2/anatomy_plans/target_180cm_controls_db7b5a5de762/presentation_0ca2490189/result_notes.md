# Trial2 anatomical experiment: how to read this result
Selected candidate: **balanced**. Status: **diagnostic_only_not_feasible**.
The selected video is a complete independent 976-frame fit, not a concatenation of the short clips.
## Full-run comparison
| Check | Previous model | Selected constrained model |
|---|---:|---:|
| Contact-failure frames | 91.000 | 534.000 |
| Support-failure frames | 748.000 | 872.000 |
| Pre-release failure frames | 156.000 | 156.000 |
| Hip acceleration RMS (cm/s²) | 919.152 | 194.015 |

A lower motion metric does not compensate for a contact or support failure. Both are reported; contact tolerance remains 0.1 cm.
## Why this candidate was selected
All tested plans checked; no feasible plan; least-violating diagnostic under the declared anatomy/physics ordering.
- Compared with steady: First differing criterion: worst_failure_fraction (0.9129098360655737 vs 0.8934426229508197).

Solver converged: **False**. Maximum constrained iterations reached.
ROM / coordinate speed / coordinate acceleration failure frames: **0**.
## Remaining uncertainty
Necessary target-span conflicts occur in **53** frames. These concern the chosen model targets, not the filmed climber or every point in the hold regions.
Wall angles, ground geometry and wrist orientation are inferred. The reduced arm chain does not include scapular rhythm, full self-collision, muscles, or dynamic validation.
Elbow swivel is omitted near straight arms. Use the common-observability CSVs and their sample counts for like-for-like comparisons; undefined rotation is not zero rotation.
The full constrained model can lose contact accuracy even where a short isolated fit succeeds. Short-clip success does not certify the full sequence.
