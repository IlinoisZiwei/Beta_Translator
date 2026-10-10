#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

rm -f models/pose_landmarker.task
rm -rf outputs && mkdir outputs
rm -f .setup_done
rm -rf .venv