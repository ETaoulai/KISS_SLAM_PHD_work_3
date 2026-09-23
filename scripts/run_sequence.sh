#!/bin/zsh
# Επιβεβαίωση σε άλλη ακολουθία (#028): οι ίδιες ακριβώς ρυθμίσεις με το church_02.
#   1. KISS (configs/indoor_detail.yaml) · 2. χωρίς deskew · 3. κίνηση από την εικόνα (car, παρεμβολή, «ίδιο σημείο» 5 cm
#   μόνο στο κοντινό δάπεδο — #027) · 4. SLAM με την κίνηση της εικόνας ως deskew ΚΑΙ αρχική θέση του ICP, σ σταθερό 2.0
#   (#031 — η ρύθμιση που δουλεύει σε church_02 και christ-church-03).
#     scripts/run_sequence.sh christ-church-03
# Αξιολόγηση: python scripts/evaluate_gt.py gt/<seq>_gt-tum.txt runs/<seq>_* --offset-ms=-10  (και −55 για το «χωρίς deskew»)
export PATH=/opt/homebrew/Caskroom/miniforge/base/envs/kissslam/bin:$PATH
source "$(dirname "$0")/env_threads.sh"
cd "$(dirname "$0")/.."
S=$1; BAG=data/$S.bag; GT=gt/${S}_gt-tum.txt
python - "$S" <<'PY'
import sys, yaml
c = yaml.safe_load(open("configs/indoor_detail.yaml")); c["odometry"]["preprocessing"]["deskew"] = False
yaml.safe_dump(c, open(f"runs/{sys.argv[1]}_nodeskew.yaml", "w"), sort_keys=False)
PY
echo "### $S KISS $(date +%T)"
nice -n 10 env KISS_SLAM_OUT_DIR=$PWD/runs/${S}_kiss kiss_slam_pipeline $BAG -t /hesai/pandar --config configs/indoor_detail.yaml --no-use-intensity > runs/${S}_kiss.log 2>&1
echo "### $S χωρίς deskew $(date +%T)"
nice -n 10 env KISS_SLAM_OUT_DIR=$PWD/runs/${S}_nodeskew kiss_slam_pipeline $BAG -t /hesai/pandar --config runs/${S}_nodeskew.yaml --no-use-intensity > runs/${S}_nodeskew.log 2>&1
echo "### $S κίνηση εικόνας $(date +%T)"
python scripts/precompute_i3_motion.py --model=car --subpixel --stuck=0.05 --floor-only --bag=$BAG --gt=$GT --run=runs/${S}_kiss > runs/${S}_i3_motion.log 2>&1
echo "### $S SLAM με deskew+init εικόνας, σ 2.0 $(date +%T)"
python scripts/run_i3_deskew.py configs/indoor_detail.yaml runs/${S}_i3 runs/${S}_i3_motion_car_sp_st0.05_floor.npz $BAG 0 init fixed > runs/${S}_i3.log 2>&1
echo "### DONE $S $(date +%T)"
