#!/usr/bin/env bash
# Builds the C++ guided matching of #087 (kiss_slam/cpp/guided_match.cpp) into kiss_slam/_guided_match<EXT_SUFFIX>.
# Run inside the kiss-slam-main environment.  Without it, kiss_slam.intensity_deskew falls back to the Python version.
set -e
cd "$(dirname "$0")/.."
SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O3 -march=native -fopenmp -shared -fPIC -std=c++17 $(python -m pybind11 --includes) kiss_slam/cpp/guided_match.cpp -o kiss_slam/_guided_match$SUFFIX
echo "built kiss_slam/_guided_match$SUFFIX"
