# Exit the script if any command fails
set -e

# Install necessary global libraries
brew install python@3.12 wget ffmpeg git-lfs

# Create the Python 3.12 virtual environment and enter it
python3.12 -m venv .venv
source .venv/bin/activate

# Install the python libraries/dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Set some environment variables
MAX_JOBS=1 CFLAGS="-O0 -g0" CXXFLAGS="-O0 -g0" \

# Create directories if not already existing
mkdir -p models
mkdir -p outputs

# Install other dependencies/files
cd models
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
if [ ! -f pose_landmarker.task ]; then
  wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
fi
cd ../


touch .setup_done