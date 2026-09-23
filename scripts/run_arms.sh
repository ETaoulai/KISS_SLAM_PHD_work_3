#!/bin/zsh
# Τρέχει επιλεγμένους βραχίονες με το ίδιο config, έξοδος στο runs/ (μόνιμο, gitignored).
#
# Χρήση:  ./scripts/run_arms.sh <config> <arm> [<arm> ...]
#   arm ∈ { base, refine, replace }
#
#   base     → --no-use-intensity                       (vanilla KISS-SLAM)
#   refine   → --use-intensity --intensity-mode refine  (αρχική μέθοδος: 2ος ICP)
#   replace  → --use-intensity --intensity-mode replace (ένας ICP μόνο με τα επιλεγμένα)
#
# Παράδειγμα:  ./scripts/run_arms.sh configs/indoor_detail.yaml refine replace
set -e
CFG=$1; shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT
TAG=$(basename $CFG .yaml)
DIRS=()

for ARM in "$@"; do
  case $ARM in
    base)    FLAGS=(--no-use-intensity) ;;
    refine)  FLAGS=(--use-intensity --intensity-mode refine) ;;
    replace) FLAGS=(--use-intensity --intensity-mode replace) ;;
    *) echo "άγνωστος βραχίονας: $ARM"; exit 1 ;;
  esac
  OUT=$ROOT/runs/${TAG}_${ARM}
  rm -rf $OUT
  echo "########## $ARM [$TAG]  $(date +%T) ##########"
  KISS_SLAM_OUT_DIR=$OUT kiss_slam_pipeline data/church_02_cut.bag -t /hesai/pandar \
    --config $CFG $FLAGS
  DIRS+=(runs/${TAG}_${ARM})
done

echo "########## DONE $(date +%T) ##########"
echo "Αξιολόγηση:"
echo "  python scripts/evaluate_gt.py gt/church_02_gt-tum.txt ${DIRS[*]}"
