#!/bin/zsh
# Λήψη μιας ακολουθίας Oxford Spires (δημόσιο, χωρίς λογαριασμό): raw rosbag + GT (TUM). Συνεχίζει αν διακοπεί.
#     scripts/download_sequence.sh 2024-03-18-christ-church-03
set -e
cd "$(dirname "$0")/.."
SEQ=$1
BASE=https://huggingface.co/datasets/ori-drs/oxford_spires_dataset/resolve/main/sequences/$SEQ
API=https://huggingface.co/api/datasets/ori-drs/oxford_spires_dataset/tree/main/sequences/$SEQ/raw/rosbag
mkdir -p data gt
curl -sfL "$BASE/processed/trajectory/gt-tum.txt" -o "gt/${SEQ#2024-??-??-}_gt-tum.txt"
for f in $(curl -s "$API" | python3 -c "import sys,json; print(' '.join(x['path'].split('/')[-1] for x in json.load(sys.stdin)))"); do
  echo "### $SEQ/$f $(date +%T)"
  curl -fL -C - --retry 5 "$BASE/raw/rosbag/$f" -o "data/${SEQ#2024-??-??-}.bag"
done
echo "### DONE $SEQ $(date +%T)"
