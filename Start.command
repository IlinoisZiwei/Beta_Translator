#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! bash setup.sh; then
  echo 'Setup stopped. The error above explains what to fix; run this file again to retry.'
  read -r -p 'Press Return to close.' _beta_answer
  exit 1
fi
exec .venv/bin/python -m jupyterlab --ServerApp.root_dir="$PWD"
