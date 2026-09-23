#!/bin/zsh
# A/B run: baseline vs intensity, ίδιο config, έξοδος σε runs/ (μόνιμο, gitignored).
# Χρήση:  ./scripts/run_ab.sh [config] [extra kiss_slam_pipeline args...]
set -e
CFG=${1:-configs/indoor_fast.yaml}; shift 2>/dev/null || true
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT
TAG=$(basename $CFG .yaml)

echo "########## A: BASELINE  [$TAG]  $(date +%T) ##########"
KISS_SLAM_OUT_DIR=$ROOT/runs/${TAG}_base \
  kiss_slam_pipeline data/church_02_cut.bag -t /hesai/pandar --config $CFG --no-use-intensity "$@"

echo "########## B: INTENSITY [$TAG]  $(date +%T) ##########"
KISS_SLAM_OUT_DIR=$ROOT/runs/${TAG}_int \
  kiss_slam_pipeline data/church_02_cut.bag -t /hesai/pandar --config $CFG --use-intensity "$@"

echo "########## DONE $(date +%T) ##########"
echo "Αξιολόγηση:"
echo "  python scripts/evaluate_gt.py gt/church_02_gt-tum.txt runs/${TAG}_base runs/${TAG}_int"
