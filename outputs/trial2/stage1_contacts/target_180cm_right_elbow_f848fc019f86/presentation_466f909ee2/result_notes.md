# Trial2 Stage 1 comparison

Frames 550–595, 180 cm, 30 fps.
Selected: **stage1_balanced**. Status: **diagnostic_only_not_feasible**.
All tested plans checked; no feasible plan; least-violating diagnostic under the declared anatomy/physics ordering.

The control and Stage 1 candidates use the same initial body coordinates, contact labels, wall hypothesis and iteration budget. The common audit checks actual hand/foot patch gaps at 0.1 cm in BOTH models. The old joint-target score is retained separately.

The source wrist rotation coupling is restored. Shoulder-girdle translations and bone offsets remain omitted. Foot frames are surface/image hypotheses, not validated ankle anatomy. No contact label was deleted or changed.

Short-segment convergence and accuracy do not establish full-sequence feasibility. Solver failures remain visible in candidate_comparison.csv.
