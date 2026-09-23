#!/bin/zsh
# Βελτιώσεις της συνένωσης χαρτών (loop closure) σε κτήριο με ορόφους — #014.
# Βάση: configs/indoor_detail.yaml (ίδια με το runs/indoor_detail_base_overlapfix του #013).
# Αλλάζει μόνο:
#   topk5      loop_closer.top_k = 5            (επαλήθευση έως 5 υποψηφίων, όχι μόνο του πρώτου)
#   h<H>       local_mapper.splitting_height = H (νέος χάρτης όταν το ύψος αλλάξει > H m)
#   topk5_h<H> και τα δύο
# Οι βραχίονες τρέχουν διαδοχικά (παράλληλα μοιράζονται τους πυρήνες και αργούν συνολικά). Αξιολόγηση:
#   python scripts/evaluate_gt.py gt/church_02_gt-tum.txt runs/indoor_detail_lc_* --offset-ms=-70
#   python scripts/analyze_loop_closures.py runs/indoor_detail_lc_<arm>
#
#     scripts/run_lc_variants.sh [H, default 1.5] [nodeskew]
#   με «nodeskew»: ίδιοι βραχίονες με deskew: false (runs/indoor_detail_nodeskew_lc_*) — #015.
#   Αξιολόγηση τότε με --offset-ms=-55 (χωρίς deskew η θέση αντιστοιχεί στη μέση της σάρωσης, #012).
#     scripts/run_lc_variants.sh 1.5 {deskew|nodeskew} V
#   με τρίτο όρισμα V: ΜΟΝΟ ο βραχίονας top-5 + ύψος, με loop_closer.max_height_disagreement = V m
#   (έλεγχος ύψους στις ενώσεις, #016) → runs/<prefix>_topk5_h<H>_vc<V>.
export PATH=/opt/homebrew/Caskroom/miniforge/base/envs/kissslam/bin:$PATH
source "$(dirname "$0")/env_threads.sh"
cd "$(dirname "$0")/.."
H=${1:-1.5}
DESKEW=true; PREFIX=indoor_detail_lc
if [ "$2" = "nodeskew" ]; then DESKEW=false; PREFIX=indoor_detail_nodeskew_lc; fi
VC=${3:-null}

make_cfg() {   # $1 = όνομα, $2 = top_k, $3 = splitting_height ή null
  python - "$1" "$2" "$3" "$DESKEW" "$VC" <<'EOF'
import sys, yaml
name, k, h, deskew, vc = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] == "true", sys.argv[5]
c = yaml.safe_load(open("configs/indoor_detail.yaml"))
c["odometry"]["preprocessing"]["deskew"] = deskew
c["loop_closer"]["top_k"] = k
c["local_mapper"]["splitting_height"] = None if h == "null" else float(h)
c["loop_closer"]["max_height_disagreement"] = None if vc == "null" else float(vc)
yaml.safe_dump(c, open(f"runs/{name}.yaml", "w"), sort_keys=False)
EOF
}

run_arm() {    # $1 = όνομα
  rm -rf "runs/$1"
  KISS_SLAM_OUT_DIR=$PWD/runs/$1 kiss_slam_pipeline data/church_02_cut.bag -t /hesai/pandar \
    --config "runs/$1.yaml" --no-use-intensity > "runs/$1.log" 2>&1
  echo "### $1 τέλος $(date +%T)"
}

if [ "$VC" != "null" ]; then
  make_cfg ${PREFIX}_topk5_h${H}_vc$VC 5 $H
  echo "### έναρξη $(date +%T)"; run_arm ${PREFIX}_topk5_h${H}_vc$VC; echo "### DONE $(date +%T)"; exit 0
fi
make_cfg ${PREFIX}_topk5 5 null
make_cfg ${PREFIX}_h$H 1 $H
make_cfg ${PREFIX}_topk5_h$H 5 $H
echo "### έναρξη $(date +%T)"
run_arm ${PREFIX}_h$H
run_arm ${PREFIX}_topk5
run_arm ${PREFIX}_topk5_h$H
wait
echo "### DONE $(date +%T)"
