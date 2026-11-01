#!/usr/bin/env bash
set -euo pipefail
beta_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$beta_project_dir"
case "${1:-}" in
  ""|--skip-system) ;;
  *) echo "Usage: bash setup.sh [--skip-system]" >&2; exit 2 ;;
esac

# --skip-system is for an existing Python 3.12 + FFmpeg installation (and CI).
if [[ "${1:-}" != "--skip-system" ]]; then
  if [[ "$(uname -s)" != Darwin ]]; then
    echo 'Automatic system setup supports macOS. Install Python 3.12 and FFmpeg, then run bash setup.sh --skip-system.' >&2
    exit 1
  fi
  if ! command -v brew >/dev/null; then
    for beta_brew in /opt/homebrew/bin/brew /usr/local/bin/brew; do
      if [[ -x "$beta_brew" ]]; then eval "$("$beta_brew" shellenv)"; break; fi
    done
  fi
  if ! command -v brew >/dev/null; then
    echo 'Installing Homebrew. macOS may ask for your password and Command Line Tools.'
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [[ -x /opt/homebrew/bin/brew ]]; then eval "$(/opt/homebrew/bin/brew shellenv)";
    else eval "$(/usr/local/bin/brew shellenv)"; fi
  fi
  for beta_formula in python@3.12 ffmpeg; do
    if ! brew list --versions "$beta_formula" >/dev/null 2>&1; then brew install "$beta_formula"; fi
  done
  beta_python="$(brew --prefix python@3.12)/bin/python3.12"
else
  beta_python="${BETA_PYTHON:-python3.12}"
fi
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || {
  echo 'FFmpeg and ffprobe are required.' >&2; exit 1;
}
if [[ ! -x .venv/bin/python ]]; then "$beta_python" -m venv .venv; fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12; rename the old .venv folder and run setup again."'
beta_signature="$(.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
h = hashlib.sha256()
for name in ('setup.sh', 'requirements.txt', 'scripts/check_environment.py'):
    h.update(Path(name).read_bytes())
print(h.hexdigest())
PY
)"
if [[ ! -f .venv/.beta-setup ]] || [[ "$(cat .venv/.beta-setup)" != "$beta_signature" ]] \
   || ! .venv/bin/python scripts/check_environment.py >/dev/null 2>&1; then
  echo 'Installing the shared environment (first run can take several minutes)...'
  .venv/bin/python -m pip install -r requirements.txt
  MAX_JOBS=1 CFLAGS="-O0 -g0" CXXFLAGS="-O0 -g0" \
    .venv/bin/python -m pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git@1e3e13bbf607b54f62205c4c33922521822fb298'
  .venv/bin/python -m pip check
  .venv/bin/python scripts/check_environment.py
  printf '%s\n' "$beta_signature" > .venv/.beta-setup
fi
.venv/bin/python -m ipykernel install --prefix "$beta_project_dir/.venv" \
  --name beta-translator --display-name 'Python (Beta Translator)'
.venv/bin/python scripts/prepare_assets.py
echo 'Ready. All five notebooks use Python (Beta Translator).'
echo 'Double-click Start.command to open JupyterLab, or select .venv/bin/python in VS Code.'
