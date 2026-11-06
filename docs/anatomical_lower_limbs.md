# Anatomical lower limbs in the shared model

The shared `BodyModel` now uses 32 pose coordinates: the existing six root/torso and fourteen arm coordinates, plus six per leg. Trial1, Trial2 and Trial3 use the same implementation in `beta_translator/body_legs.py`, with bounds in `body_constraints.py` and source checks in `body_leg_reference.py`.

## Chain and source

The hierarchy is pelvis → three-axis hip → femur → coupled knee → tibia → ankle → talus → subtalar joint → calcaneus/shoe frame → contact patch. Pelvis orientation remains tied to the rigid torso. Toe motion is fixed, and the shoe landmarks are geometric proxies.

Joint axes, ordering, signs, translations and curve knots come from the [official OpenSim Gait2392 model](https://github.com/opensim-org/opensim-models/blob/d9b05d470b1a481c222372c85b75772faf8f7792/Models/Gait2392_Simbody/gait2392_thelen2003muscle.osim). The [OpenSim documentation](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53086215/Gait%2B2392%2Band%2B2354%2BModels) describes its model family. Extracted parameters and source hashes are in `data/lower_limb_source.json`; no OpenSim installation is needed at runtime.

The knee's source angle is negative in flexion. The shared API uses **positive knee flexion** and converts its sign at the source boundary. Knee translation uses the source SIMM cubic curves, including their endpoint rules. The tibial joint origin therefore moves relative to the femur's distal frame. Bone-length auditing checks the rigid femur and tibia separately; it does not mistakenly call that joint translation bone stretching. The renderer shows the short connection between the two frames.

Project coordinates remain X right, Y up, Z away from the wall, with cm and radians. Both source sides are imported explicitly. Hip spacing and segment lengths retain the project's height-scaled proportions. The femoral curve is scaled to the selected nominal thigh length, the tibial shaft to shank length, and distal source offsets by height/180. This is declared project scaling, not subject-specific OpenSim scaling or a complete musculoskeletal import.

## Adopted limits

The source's broad ranges are coordinate metadata, with `clamped=false`; they are not physiological ROM guarantees. This experiment deliberately enforces narrower **assumed screening bounds**, with no knee hyperextension. These bounds can reject a real climbing pose and should be personalized before interpreting a failure anatomically.

| Coordinate | Adopted range (degrees) | Speed cap (deg/s) | Acceleration cap (deg/s²) |
| --- | ---: | ---: | ---: |
| Hip flexion | −30 to 120 | 150 | 750 |
| Hip adduction | −60 to 35 | 120 | 600 |
| Hip axial rotation | −50 to 50 | 120 | 600 |
| Knee flexion | 0 to 120 | 180 | 900 |
| Ankle source dorsiflexion coordinate | −50 to 30 | 120 | 600 |
| Subtalar source coordinate | −20 to 25 | 90 | 450 |

All speed/acceleration caps are project assumptions, checked at native samples. Scalar limits do not establish coupled personal ROM, ligament loading, continuous-time validity, strength or dynamic balance.

## Feet and contact

The ankle and subtalar joints generate each foot frame. A patch is computed as `foot_origin + foot_frame @ local_patch`; optimization, support auditing and video use that same point. Foot rotation can no longer change independently of the shank/ankle chain.

Forefoot position uses the source MTP offset. A ground-sole proxy lies 0.03 height forward of the calcaneal origin. Sole depth is 0.008 height; the displayed distal toe extends 0.035 height beyond the MTP offset. These dimensions and contact modes are assumptions, not shoe measurements. Existing ground proposals remain conditional; absent wall contact never creates ground support.

Surface orientation is a weak, explicitly uncertain prior. Visible source heel/toe points also contribute reprojection evidence; missing or identity-rejected observations have zero weight. Neither a point-contact match nor a fitted foot frame establishes a measured sole orientation, grip wrench, no-slip condition, or ankle safety. The old aggregate COM fractions remain in use; foot mass is not added twice.

## Compare and run

The three active notebooks now default to **175 cm** and load `configs/leg_verification_results.json`. The previous 180 cm index and outputs are preserved. The current-model sections are now included after the restored evidence/review and historical experiment stages; see [the complete notebook guide](complete_trial_notebooks.md).

```sh
source .venv/bin/activate
python scripts/run_body_model.py --trial all --height-cm 175 --frames verification --iterations 150
```

Each run fits a legacy control and two anatomical candidates. The control is displayed for comparison but cannot win the anatomical-model selection. Candidates share input observations and nominal proportions; initialization and foot geometry differ because the representations differ. No local winners are stitched and no post-fit pose smoothing is applied. Set `--frames all` for a full optimization; short-run verification does not establish full-sequence convergence.

## Verification boundaries

Tests compare both sides against independent source-parameter transforms and a separately constructed cubic interpolator. They check signed knee behavior, ankle/subtalar isolation, symmetry, rigid segment geometry, contact identity and numerical derivatives through all three trial adapters. This is a source-parameter cross-check, not an OpenSim engine execution or experimental human validation.

New results include source verification, full per-frame metrics, adopted-limit failures, knee translation magnitude, foot-frame uncertainty, candidate reasons and 3D videos. Full-frame initialization checks and short optimization outcomes are recorded separately. Shoulder-girdle detail, personalized coupled limits, collision, muscles/fatigue, inertial dynamics and impact validation remain outside this change.

## Recorded 175 cm results

Verification on November 5, 2026: **248 tests passed**, and each then-current analysis notebook executed all nine code cells in order without errors. Both source-side chains agreed with independent source-parameter transforms to floating-point precision (maximum position error below 4 × 10⁻¹⁶ m).

The following counts are **failed frames**, for the selected `steady` candidate. They describe the tested short clips, not complete-video optimization.

| Trial | Frames (inclusive) | Adopted ROM/rate failures | Contact failures | Support failures | Pre-unload failures |
| --- | --- | ---: | ---: | ---: | ---: |
| Trial1 | 113–136 (24) | 0 | 0 | 23 | N/A (no applicable frames) |
| Trial2 | 550–595 (46) | 0 | 0 | 22 | 12 of 12 applicable |
| Trial3 | 296–331 (36) | 0 | 0 | 20 | 6 of 6 applicable |

Each candidate received 150 iterations. None converged within that budget; all selected plans remain **diagnostic, not feasible solutions**. Trial3's selected anatomical fit reduces contact failures from 4 to 0 and support failures from 24 to 20 versus its same-height legacy control. Trial1/Trial2 retain the same support failure counts as their controls. Motion improvement is not universal: Trial2's selected joint-acceleration RMS is 34.6 cm/s² versus 31.3 cm/s² for its legacy control. Full candidate tables and reasons are saved with each run.

Separately, all **1,896 native frames** (417 + 976 + 503) passed initialization/geometry checks: rigid-segment length error below 8 × 10⁻¹⁴ cm, matching contact-point definitions, retained source/identity masks, and zero free-contact forces. This is a forward-kinematics and adapter check; it does not assert that all initialized frames meet contact/support/rate constraints. All six short comparison/selected videos decode at the expected frame count and 30 fps.

The [machine-readable verification record](validation/lower_limb_verification_175cm.json) contains measurements, media checks and preservation results. The [full-frame initialization record](validation/lower_limb_native_175cm.json) is separate. New outputs are indexed in `configs/leg_verification_results.json`; the old 180 cm index, outputs, source videos and `Beta_Translator.ipynb` are unchanged. Exact previous trial notebook snapshots are in `notebooks/history/pre_anatomical_legs_20261105.zip`.
