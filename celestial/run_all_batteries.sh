#!/bin/bash
# run_all_batteries.sh -- run sky_battery.py for every model with PCA128
# activations in PCA128/ (or only the models given as arguments).
# Figures go to model_sky_batteries/sky_battery_{model}.pdf/.png; each
# model's printed numbers go to model_sky_batteries/sky_battery_{model}.log;
# standalone degree-purity heatmaps go to
# model_sky_heatmaps/sky_heatmap_{model}.pdf/.png.
#
# Usage: ./run_all_batteries.sh [model ...] [-- extra sky_battery.py args]
#   e.g. ./run_all_batteries.sh
#        ./run_all_batteries.sh qwen32b llama33_70b
#        ./run_all_batteries.sh -- --lmax 6
set -o pipefail
cd "$(dirname "$0")"
OUTDIR=model_sky_batteries
HMDIR=model_sky_heatmaps
mkdir -p "$OUTDIR" "$HMDIR"

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
  python3 sky_battery.py --model "$m" --out "$OUTDIR/sky_battery_$m" \
    --heatmap-out "$HMDIR/sky_heatmap_$m" "$@" \
    2>&1 | tee "$OUTDIR/sky_battery_$m.log" || failed+=("$m")
done

echo
echo "===== summary (best layer) ====="
for m in "${models[@]}"; do
  log="$OUTDIR/sky_battery_$m.log"
  printf '%-24s %s | %s | %s\n' "$m" \
    "$(grep -m1 'best layer' "$log" | sed 's/.*: //;s/^/layer /')" \
    "$(grep -m1 'zonality' "$log" | sed 's/.*R^2 of activation Gram): */zonality /')" \
    "$(grep -m1 -F 'decode R^2 (weighted)' "$log" | sed 's/.*: */decode /')"
done
[ ${#failed[@]} -gt 0 ] && echo "FAILED: ${failed[*]}"
exit 0