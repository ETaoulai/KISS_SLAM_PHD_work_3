#!/bin/zsh
# Deskew σε περισσότερα περάσματα (#017): ICP → deskew ξανά με την κίνηση της ίδιας της σάρωσης → ICP.
# Βάση: configs/indoor_detail.yaml (deskew: true)· αλλάζει μόνο το deskew_refine.passes.
# Όρια σύγκρισης: deskew KISS 0.339 m · χωρίς deskew 0.283 m · deskew από GT 0.101 m (#012).
#     scripts/run_deskew_refine.sh [passes ...]      (default: 2 3)
# Αξιολόγηση (η θέση αντιστοιχεί στο τέλος της σάρωσης όταν το deskew είναι σωστό → δοκίμασε −10 και −70 ms):
#     python scripts/evaluate_gt.py gt/church_02_gt-tum.txt runs/indoor_detail_deskew_p* --offset-ms=-10
export PATH=/opt/homebrew/Caskroom/miniforge/base/envs/kissslam/bin:$PATH
source "$(dirname "$0")/env_threads.sh"
cd "$(dirname "$0")/.."
for P in ${@:-2 3}; do
  NAME=indoor_detail_deskew_p$P
  python - "$NAME" "$P" <<'PY'
import sys, yaml
name, p = sys.argv[1], int(sys.argv[2])
c = yaml.safe_load(open("configs/indoor_detail.yaml"))
c["deskew_refine"] = {"passes": p}
yaml.safe_dump(c, open(f"runs/{name}.yaml", "w"), sort_keys=False)
PY
  rm -rf "runs/$NAME"
  echo "### $NAME έναρξη $(date +%T)"
  KISS_SLAM_OUT_DIR=$PWD/runs/$NAME kiss_slam_pipeline data/church_02_cut.bag -t /hesai/pandar \
    --config "runs/$NAME.yaml" --no-use-intensity > "runs/$NAME.log" 2>&1
  echo "### $NAME τέλος $(date +%T)"
done
echo "### DONE $(date +%T)"
