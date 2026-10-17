# Automatic contact predictions

Sections 17–18 of `Beta_Translator.ipynb` predict limb-to-hold contacts and show them over the video for review. They use existing pose landmarks, selected hold masks, and per-frame homographies. They do not overwrite the manual contact labels or IK inputs.

## Run in Colab

Open this branch's notebook in Colab. Run the Colab-only setup cell immediately below the title before Section 1. It installs dependencies and requests the real `experiment_config.yml`, `model_final.pth`, and `trial1.mp4` resources if missing. Opening a GitHub notebook does not fetch those files. The original macOS setup is preserved; the Colab cell skips its setup outside Colab.

Download `data/reviewed_contact_data.zip` from this branch and upload it when the hold-map helper requests it. Keep the filename unchanged. The bundle contains the reviewed map, existing pose coordinates, verified alignment, contact tables, settings, and sample images. Colab runtime files are temporary, so retain the original resources and bundle outside the runtime.

In a local checkout, the helper reads `data/reviewed_contact_data.zip` directly. The export cell saves a new bundle under `outputs/`; in Colab it also offers a download.

## Stable hold IDs

The earlier saved detector output contained 66 holds. The validation run retained 65. IDs are assigned by position within each prediction run, so reusing the old selection could refer to different physical holds.

The bundle preserves all 65 masks and the 21 reviewed route IDs together. Section 9 loads this map by default, verifies the video's SHA-256, and avoids rerunning detection and assigning new IDs. Section 10 restores its selected route. To review a new detector result, disable `USE_SAVED_HOLD_MAP` in Section 9 and enter a newly reviewed selection in Section 10 before saving it; the old selection is not automatically transferred.

The older Section 16 calibration note calls its endpoints H45/H41; these physical holds are H44/H40 in this saved map. The numerical calibration points and scale are unchanged.

## Prediction rules and validation

The matcher warps hold masks into the current frame, checks landmark visibility and distance to the mask, and requires a unique candidate for at least 0.5 consecutive seconds. Initial margins are 20 pixels for wrists and 10 for foot-index landmarks at 720px video width. Ambiguous, low-visibility, and unreliable-alignment observations remain unresolved. These are image-based predictions, not proof of gripping or supporting weight.

The Colab validation used the existing 417-row pose CSV and previously verified 417-frame alignment with zero fallbacks. It produced 1,668 limb observations and 10 draft intervals. Visibility and ambiguity split intervals, so this is not a count of moves. Five full-frame samples and six close-ups were inspected; this was a visual sanity check, not a labeled accuracy benchmark. The complete pose/IK pipeline was not rerun for this validation.

Sections 17–18 save tables and images under `outputs/contact_matching/`. The additional close-up and export cells follow Section 18. Pose extraction, homography estimation, macOS setup, and IK logic remain unchanged.
