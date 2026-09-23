#!/usr/bin/env bash
# End-to-end της διόρθωσης κοντινού πεδίου με το πρωτόκολλο του #037 (4 σπόροι), στο church_02.
# Βραχίονας Α: σημερινή μέθοδος (runs/var_raycast_s*, υπάρχουν ήδη). Βραχίονας Β: με --trans-mode=auto.
set -u
cd "$(dirname "$0")/.."
source scripts/env_threads.sh
RUNS=""
for S in 0 1 2 3; do
  SUF=""; [ "$S" != 0 ] && SUF="_seed${S}"
  NPZ="runs/i3_motion_car_sp_st0.05_floor_ray_tr8${SUF}.npz"
  OUT="runs/auto_s${S}"
  [ -f "$NPZ" ] || nice -n 10 python scripts/precompute_i3_motion.py --model=car --subpixel --stuck=0.05 \
       --floor-only --render=raycast --trans-min=8 --trans-mode=auto --seed="$S" > "runs/auto_motion_s${S}.log" 2>&1
  [ -d "$OUT" ] || KISS_SLAM_OUT_DIR="$OUT" nice -n 10 kiss_slam_pipeline data/church_02_cut.bag -t /hesai/pandar \
       --config configs/indoor_detail.yaml --image-deskew --motion-file "$NPZ" > "${OUT}.log" 2>&1
  RUNS="$RUNS $OUT"
  echo "ok $OUT"
done
echo "=== Αξιολόγηση"
nice -n 10 python scripts/evaluate_gt.py gt/church_02_gt-tum.txt runs/var_raycast_s0 runs/var_raycast_s1 runs/var_raycast_s2 runs/var_raycast_s3 $RUNS --offset-ms=-10 2>&1 | grep -v Warn
