#!/usr/bin/env bash
set -euo pipefail

# Run once in a terminal; the notebook does not install packages.
beta_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$beta_project_dir"
brew install python@3.12 ffmpeg git-lfs
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
MAX_JOBS=1 CFLAGS="-O0 -g0" CXXFLAGS="-O0 -g0" \
  .venv/bin/python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git"
mkdir -p models outputs videos
if [ ! -f models/pose_landmarker.task ]; then
  .venv/bin/python - <<'PYMODEL'
from pathlib import Path
from urllib.request import urlretrieve
path = Path('models/pose_landmarker.task')
temporary = path.with_suffix('.download')
urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
            'pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task', temporary)
temporary.replace(path)
PYMODEL
fi
printf 'Setup complete. Select .venv/bin/python as the notebook kernel.\n'
