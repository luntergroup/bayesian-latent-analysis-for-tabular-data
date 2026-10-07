#!/usr/bin/env bash
# The whole example analysis, one step after another:
#
#   1  make_example_data.sh              simulate and convert the example table
#   2  run_model.sh K    for every K     bare chains and evidence   (Figs 1, 2, 3)
#   3  run_final.sh K_FINAL              calibrated final model, explain, imputation,
#                                        and the notebooks          (Figs 4-6; all figures)
#
# The settings come from one of two files, sourced first:
#
#   source quick.sh         # about half an hour
#   source thorough.sh      # the paper's settings; days if run sequentially
#
# Execute one of the above, then do
#
#   ./run_all.sh
#
# Steps with finished output are skipped, so rerunning after an interruption is safe.
#
# The run_all.sh script runs every step in sequence.  On a cluster the work can be spread
# out: each `blat` call is a single-core job that communicates only through files, so the
# calls can be submitted in six steps, all jobs in a step at once, each step starting when
# the previous one has finished.  Each step is a section marked "Step N" in run_model.sh
# (Steps 1-2, once per K) or run_final.sh (Steps 3-6):
#
#   1. Bare models: `blat run` for every K in K_GRID and every chain (GRID x CHAINS
#      jobs), together with the forward TI paths, which need only the data.
#   2. Backward TI paths, TI_REPLICATES per K; each needs only chain 1 at its K.
#   3. Calibration at K_FINAL, starting from chain 1 at K_FINAL.
#   4. The final runs at K_FINAL, with the calibrated declaration: FINAL_CHAINS
#      independent chains, of which chain 1 is the final model.
#   5. In parallel: explain and imputation of the new records, both from final chain 1,
#      and for each fold a masked run followed by its imputation.  (Step 5 needs only
#      chain 1 of Step 4, and the fold runs only the calibration, so they may also start
#      alongside Step 4.)
#   6. Pooling the imputed folds into cross_validation.tsv, a few seconds of Python.
#
# The notebooks (the last section of run_final.sh) only read these outputs, so each can
# run once its inputs exist: figure1-convergence and figure3-tsne after Step 1,
# figure2-evidence after Step 2, figure4-loadings and figure6-chains after Step 4, and
# figure5-types after Step 5.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${BLAT_SETTINGS:-}" ]; then
  cat >&2 <<EOF
run_all.sh: no settings.  Source one of the settings files first:

  source quick.sh        # quick run, about half an hour
  source thorough.sh     # the paper's settings (about 75 hours sequentially)

then run ./run_all.sh again.
EOF
  exit 2
fi

K_GRID="${K_GRID:?is not set; see quick.sh}"
K_FINAL="${K_FINAL:?is not set; see quick.sh}"
DATA="${DATA:-example/data}"
NAME="${NAME:-example}"
case "$DATA" in /*) ;; *) DATA="$HERE/$DATA" ;; esac
export DATA NAME

echo "run_all.sh: $BLAT_SETTINGS settings; K = $K_GRID; final K = $K_FINAL"
if [ ! -f "$DATA/$NAME.yaml" ]; then
  "$HERE/make_example_data.sh"
fi
for k in $K_GRID; do
  "$HERE/run_model.sh" "$k"
done
"$HERE/run_final.sh" "$K_FINAL"
