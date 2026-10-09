# CIS 5810 Final Project - Beta Translator

Authors:

- Ziwei Wang
- Justin Zou
- Pengfan Zhang

## Overview

Beta Translator explores how computer vision can help analyze climbing movements and eventually adapt a demonstrated climbing sequence to another climber’s body proportions.

The current notebook focuses on pose estimation, climbing hold segmentation, manual route selection, and contact annotation. Personalized movement recommendations remain a project goal, rather than a validated capability of this prototype.

## Current Workflow

1. Load a climbing video.
2. Estimate body landmarks using MediaPipe and generate pose outputs.
3. Extract the current video’s first frame as the reference for hold detection.
4. Detect candidate climbing holds using a pretrained Detectron2 model.
5. Manually select the holds relevant to the route.
6. Use an interactive frame viewer to select BEFORE and AFTER frames for a movement.
7. Annotate each hand and foot’s contact state at both endpoints.

The annotation interface supports hold IDs and the states `unknown`, `free`, and `wall_contact`. Contact assignments are currently manual.

## Main Files

* `Beta_Translator.ipynb`: Main analysis notebook.
* `Codespaces_Setup_Guide.tex`: LaTeX guide for environment setup and troubleshooting.
* `experiment_config.yml`: Configuration for the pretrained hold segmentation model.
* `model_final.pth`: Matching model weights.
* `pose_landmarker.task`: MediaPipe pose model asset.
* `trial1.mp4`: Current input video.
* `pose_coordinates.csv`: Generated pose coordinate output.
* `pose_output.mp4`: Generated pose visualization.
* `wall_reference.jpg`: Saved reference image from the earlier workflow.
* `detectron2_install.log`: Detectron2 installation output.

The updated hold detection cell reads the first frame directly from the current video. The saved `wall_reference.jpg` may therefore differ from the reference used by that cell.

## GitHub Codespaces Setup

Open this repository in GitHub Codespaces and select **Terminal > New Terminal**. Run the following commands inside Codespaces.

### 1. Retrieve Git LFS assets

Model weights and videos are managed with Git Large File Storage.

```bash
sudo apt-get update
sudo apt-get install -y git-lfs build-essential libgl1 libegl1 libgles2

git lfs install
git lfs pull
```

Run `git lfs pull` before opening model weights or videos. A small Git LFS pointer file is not the actual asset.

### 2. Create a Python environment

Run these commands from the repository root:

```bash
python --version
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

For an existing working environment, activate that environment instead of creating another one.

### 3. Install Python dependencies

```bash
python -m pip install torch torchvision numpy pandas ninja
python -m pip install opencv-python matplotlib scipy pillow pyyaml
python -m pip install mediapipe ipywidgets ipykernel
```

These commands do not specify a tested set of package versions. Compatibility must be checked in the selected Python environment.

### 4. Install Detectron2

Install PyTorch before running this command:

```bash
MAX_JOBS=1 CFLAGS="-O0 -g0" CXXFLAGS="-O0 -g0" \
python -m pip install --no-build-isolation -v \
"git+https://github.com/facebookresearch/detectron2.git" \
> detectron2_install.log 2>&1

echo "Detectron2 installation exit code: $?"
```

Wait for installation to finish. An exit code of `0` indicates success.

The compiler settings reduce optimization, debug information, and parallel build activity to limit resource demand. They do not guarantee that compilation will succeed.

If installation fails, inspect the log:

```bash
tail -n 40 detectron2_install.log
```

### 5. Select the notebook kernel

Register the environment:

```bash
python -m ipykernel install --user \
--name climbing-env \
--display-name "Python (climbing-env)"
```

Ensure Microsoft’s Python and Jupyter extensions are enabled in Codespaces.

Open `Beta_Translator.ipynb`, click **Select Kernel**, and choose `Python (climbing-env)` or the repository’s `.venv/bin/python`.

Restart the kernel after installing packages.

### 6. Verify installation

Run this in a notebook cell:

```python
import sys
import torch
import torchvision
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import ipywidgets as widgets
from detectron2 import _C

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Required imports succeeded.")
```

The current hold segmentation configuration uses the CPU. A `False` CUDA result does not prevent that step from running.

Successful imports do not establish that the complete notebook or model inference will succeed.

## Running the Notebook

1. Confirm that the video path refers to the intended input video.
2. Confirm that `MODEL_DIR` contains `experiment_config.yml` and `model_final.pth`.
3. Run the notebook cells in order.
4. Review the predicted hold image.
5. Enter the desired IDs in `SELECTED_HOLD_IDS`, without duplicates.
6. Run the hold review cell before opening the contact annotation interface.
7. Check the overlay alignment before assigning contacts.

Hold IDs belong to a particular prediction run. Changing the reference frame, model, or threshold may change their numbering. Review the new prediction rather than reusing an old ID list automatically.

Restarting the kernel clears variables, including the prediction and review identifiers. Rerun the prerequisite cells after a restart.

## Recent Updates

1. Adapted file loading for files already present in Codespaces.
2. Added environment setup documentation.
3. Added the matching hold model configuration and weights.
4. Changed the hold detection reference to the current video’s first frame.
5. Updated the manually selected route hold IDs.
6. Added generated pose outputs and supporting files.

## Current Limitations

The hold rendering function applies fixed reference masks to video frames. It does not currently compensate for camera movement.

Using the first video frame as the reference addresses an inconsistent starting reference, but later frames can still become misaligned. Equal image dimensions do not guarantee equal hold coordinates.

Inspect overlays at several points in the video. If the camera moves, frame registration or video stabilization is needed before relying on spatial contact comparisons.

Hold predictions can contain missed detections or incorrect boundaries. Manual review remains necessary. The prototype does not yet establish whether an adapted movement is physically feasible for another climber.

## Troubleshooting

### Package imports fail in the notebook

Check `sys.executable` and confirm that the notebook uses the environment where the packages were installed.

### Widgets do not display

Test a basic widget:

```python
import ipywidgets as widgets
from IPython.display import display

display(widgets.IntSlider(description="Test"))
```

If it fails, confirm that Microsoft’s Jupyter and Jupyter Notebook Renderers extensions are enabled. Save the notebook, run **Developer: Reload Window** from the Command Palette, reselect the kernel, and rerun the test.

Old widget outputs may need to be cleared and recreated.

### Video cannot be read

Check the input path and file size. Confirm that Git LFS downloaded the actual video using `git lfs pull`.

A valid path alone does not guarantee that OpenCV can decode the video.

### Hold review variables are undefined

Run hold detection, then the hold selection and review cell, before running contact annotation.

### Detectron2 compilation terminates

Inspect `detectron2_install.log`. A terminated compiler does not by itself prove a memory shortage or Python compatibility problem.

To inspect container memory information:

```bash
free -h
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
```

The cgroup paths depend on the container configuration.

## Model Source

The hold segmentation experiment uses the authors’ `Basic_Augmented_P2P3_DoubleTopK_NMS_2KIT` model.

Dataset and model source:
https://www.kaggle.com/datasets/tomasslama/indoor-climbing-gym-hold-segmentation

Detectron2:
https://github.com/facebookresearch/detectron2

MediaPipe:
https://github.com/google-ai-edge/mediapipe
