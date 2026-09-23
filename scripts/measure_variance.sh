#!/usr/bin/env bash
# Μέτρηση διασποράς της μεθόδου (#037): πόσο αλλάζει το ATE όταν αλλάζει ΜΟΝΟ ο σπόρος του RANSAC;
# Χωρίς αυτήν, διαφορές ±0.02–0.05 m ανάμεσα σε παραλλαγές δεν κρίνονται.
#
#   bash scripts/measure_variance.sh [σπόροι="0 1 2 3"] [τρόποι="splat raycast"]
#
# Ένα run τη φορά, 8 νήματα (env_threads.sh). Για κάθε (τρόπος, σπόρος): κίνηση από την εικόνα → SLAM με αυτήν →
# αξιολόγηση έναντι GT. Τα αποτελέσματα μαζεύονται στο τέλος με το evaluate_gt.py σε έναν πίνακα.
set -u
cd "$(dirname "$0")/.."
source scripts/env_threads.sh
SEEDS=${1:-"0 1 2 3"}
RENDERS=${2:-"splat raycast"}
BAG=data/church_02_cut.bag
GT=gt/church_02_gt-tum.txt
CFG=configs/indoor_detail.yaml
RUNS=""
for R in $RENDERS; do
  for S in $SEEDS; do
    SUF=""; [ "$R" = raycast ] && SUF="_ray"
    [ "$S" != 0 ] && SUF="${SUF}_seed${S}"
    NPZ="runs/i3_motion_car_sp_st0.05_floor${SUF}.npz"
    OUT="runs/var_${R}_s${S}"
    if [ ! -f "$NPZ" ]; then
      echo "=== κίνηση: $R seed $S → $NPZ"
      nice -n 10 python scripts/precompute_i3_motion.py --model=car --subpixel --stuck=0.05 --floor-only \
           --render="$R" --seed="$S" > "runs/var_${R}_s${S}_motion.log" 2>&1
      tr '\r' '\n' < "runs/var_${R}_s${S}_motion.log" | grep -v "scans/s" | tail -2
    else
      echo "=== κίνηση: $R seed $S — υπάρχει ήδη ($NPZ)"
    fi
    if [ ! -d "$OUT" ]; then
      echo "--- SLAM: $OUT"
      KISS_SLAM_OUT_DIR="$OUT" nice -n 10 kiss_slam_pipeline "$BAG" -t /hesai/pandar --config "$CFG" \
           --image-deskew --motion-file "$NPZ" > "${OUT}.log" 2>&1
      grep -h "closures found" "${OUT}.log" | tail -1
    fi
    RUNS="$RUNS $OUT"
  done
done
echo; echo "=== Αξιολόγηση έναντι GT (offset −10 ms)"
nice -n 10 python scripts/evaluate_gt.py "$GT" $RUNS --offset-ms=-10 2>&1 | grep -v Warn
