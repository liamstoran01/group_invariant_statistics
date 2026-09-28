#!/bin/bash
# run_all_batteries.sh -- run sky_battery.py for every model with PCA128
# activations in PCA128/ (or only the models given as arguments).
# Writes degree-purity heatmaps to
# model_sky_heatmaps/sky_heatmap_{model}.pdf/.png.
#
# Usage: ./run_all_batteries.sh [model ...] [-- extra sky_battery.py args]
#   e.g. ./run_all_batteries.sh
#        ./run_all_batteries.sh qwen32b llama33_70b
#        ./run_all_batteries.sh -- --lmax 6
set -o pipefail
cd "$(dirname "$0")"
HMDIR=model_sky_heatmaps
mkdir -p "$HMDIR"

models=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  models+=("$1")
  shift
done
[ "$1" = "--" ] && shift
if [ ${#models[@]} -eq 0 ]; then
  for f in PCA128/*_pca128.npz; do
    m=$(basename "$f" _pca128.npz)
    models+=("$m")
  done
fi

failed=()
for m in "${models[@]}"; do
  echo "===== $m ====="
  python3 sky_battery.py --model "$m" \
    --heatmap-out "$HMDIR/sky_heatmap_$m" "$@" \
    || failed+=("$m")
done

[ ${#failed[@]} -gt 0 ] && echo "FAILED: ${failed[*]}"
exit 0