#!/usr/bin/env bash
# The final model for one K, and everything built on it, then the notebooks.
#
#   ./run_final.sh K
#
#   Steps 1-2  the bare model for K, via run_model.sh, if it has not been run
#   Step 3     calibrate the latent words N_f per continuous variable, from bare chain 1
#   Step 4     the final runs at K with the calibrated words: $FINAL_CHAINS chains, of
#              which chain 1 is the final model (-> Figs 4, 5) and the others check
#              that it is reproducible (-> Fig 6)
#   Step 5     explain (-> Fig 5), imputation of the new records, and cross-validated
#              imputation over the mask's folds
#   Step 6     pool the folds' scores
#   then the notebooks, which draw Figs 1-6 into $RESULTS/figures
# Steps whose output exists are skipped, so an interrupted run resumes.
#
# Every `blat` call is one single-core process that communicates only through files, so
# calls that do not depend on each other can run at the same time, e.g. as cluster jobs.
# The steps are those of the recipe in run_all.sh; within Step 5 everything is
# independent, except that each fold's imputation follows that fold's run.  Step 5 needs
# only chain 1 of Step 4, and the fold runs need only Step 3, so they may also start
# alongside Step 4.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- settings (edit here; each can also be set in the environment) --------------------
# The paper used ALPHA_FROZEN=2000, BURNIN=4000, ITERS=30000, THIN=10 for the final run,
# and CAL_ROUNDS=4, CAL_ITERS=1000, CAL_BURN=200, CAL_DRAWS=200 for the calibration.
# Each final chain and each cross-validation fold is fitted with the same run settings.
PYTHON="${PYTHON:-python3}"
DATA="${DATA:-$HERE/example/data}"
NAME="${NAME:-example}"
RESULTS="${RESULTS:-$HERE/example/results}"
FINAL_CHAINS="${FINAL_CHAINS:-${CHAINS:-3}}"    # final chains; as many as the bare chains
ALPHA_FROZEN="${ALPHA_FROZEN:-500}"             # final runs and each fold, as in run_model.sh
BURNIN="${BURNIN:-1000}"
ITERS="${ITERS:-5000}"
THIN="${THIN:-10}"
CAL_ROUNDS="${CAL_ROUNDS:-2}"                   # calibration rounds: short chain + estimate
CAL_ITERS="${CAL_ITERS:-1000}"                  # iterations of each short chain
CAL_BURN="${CAL_BURN:-100}"                     # per-record chain: burn-in ...
CAL_DRAWS="${CAL_DRAWS:-100}"                   # ... and draws
EXPLAIN_BURN="${EXPLAIN_BURN:-200}"             # explain's per-record chains
EXPLAIN_DRAWS="${EXPLAIN_DRAWS:-100}"
IMPUTE_PHI_DRAWS="${IMPUTE_PHI_DRAWS:-20}"      # stored draws of phi used ...
IMPUTE_THETA_DRAWS="${IMPUTE_THETA_DRAWS:-10}"  # ... and theta draws per record per phi
IMPUTE_BURN="${IMPUTE_BURN:-100}"
NOTEBOOKS="${NOTEBOOKS:-1}"                     # 0 to skip the notebooks
# ----------------------------------------------------------------------------------------
# relative paths are relative to this directory
absolute() { case "$1" in /*) echo "$1" ;; *) echo "$HERE/$1" ;; esac; }
DATA="$(absolute "$DATA")"
RESULTS="$(absolute "$RESULTS")"

K="${1:?usage: run_final.sh K}"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
blat() { "$PYTHON" -m blat "$@"; }
TABLE="$DATA/$NAME.yaml"
MASK="$DATA/$NAME.mask.tsv"
[ -f "$TABLE" ] || { echo "no $TABLE: run make_example_data.sh first" >&2; exit 2; }
mkdir -p "$RESULTS/final" "$RESULTS/explain" "$RESULTS/impute" "$RESULTS/figures"

# ==== Steps 1-2: the bare model, if chain 1 is missing ==================================
if [ ! -f "$RESULTS/runs/K$K-c1/summary.yaml" ]; then
  "$HERE/run_model.sh" "$K"
fi

# ==== Step 3: calibration ===============================================================
# needs: chain 1 at K
CALIBRATION="$RESULTS/final/calibration-K$K"
CALIBRATED="$CALIBRATION/declaration.yaml"
if [ ! -f "$CALIBRATED" ]; then
  echo "== calibrate K=$K =="
  blat calibrate --from-run "$RESULTS/runs/K$K-c1" --rounds "$CAL_ROUNDS" \
    --iters "$CAL_ITERS" --burn "$CAL_BURN" --draws "$CAL_DRAWS" --out "$CALIBRATION"
fi

# ==== Step 4: the final runs ============================================================
# needs: Step 3.  The chains are independent of each other; chain 1 is the final model.
for c in $(seq 1 "$FINAL_CHAINS"); do
  out="$RESULTS/final/K$K-c$c"
  if [ -f "$out/summary.yaml" ]; then echo "exists: $out"; continue; fi
  echo "== final run K=$K chain $c =="
  blat run --decl "$CALIBRATED" --k "$K" --seed "$((6 + c))" --alpha-frozen "$ALPHA_FROZEN" \
    --burnin "$BURNIN" --iters "$ITERS" --thin "$THIN" --out "$out"
done
FINAL="$RESULTS/final/K$K-c1"

# ==== Step 5: explain, impute the new records, impute the folds =========================
# The three parts are independent of each other; within the folds, each fold's
# imputation follows its own run, and the folds are independent.

# explain -- needs: chain 1 of Step 4
if [ ! -f "$RESULTS/explain/final/summary.yaml" ]; then
  echo "== explain =="
  blat explain --run "$FINAL" --burn "$EXPLAIN_BURN" --draws "$EXPLAIN_DRAWS" \
    --apparent --out "$RESULTS/explain/final"
fi

# impute the new records -- needs: chain 1 of Step 4
IMPUTE="--phi-draws $IMPUTE_PHI_DRAWS --theta-draws $IMPUTE_THETA_DRAWS --burn $IMPUTE_BURN"
if [ -f "$DATA/new.preproc.tsv" ] && [ ! -f "$RESULTS/impute/new/summary.yaml" ]; then
  echo "== impute the new records =="
  blat impute --run "$FINAL" --data "$DATA/new.preproc.tsv" $IMPUTE --intervals \
    --out "$RESULTS/impute/new"
fi

# impute the folds -- each fold's run needs only Step 3
FOLDS="$("$PYTHON" -c "import pandas as p, sys; print(int(p.read_csv(sys.argv[1], sep='\t', index_col=0).to_numpy().max()))" "$MASK")"
for n in $(seq 1 "$FOLDS"); do
  run="$RESULTS/impute/run-fold$n"
  if [ ! -f "$run/summary.yaml" ]; then
    echo "== cross-validation fold $n of $FOLDS: run =="
    blat run --decl "$CALIBRATED" --k "$K" --seed "$((50 + n))" --mask "$MASK" --fold "$n" \
      --alpha-frozen "$ALPHA_FROZEN" --burnin "$BURNIN" --iters "$ITERS" \
      --thin "$THIN" --out "$run"
  fi
  if [ ! -f "$RESULTS/impute/fold$n/summary.yaml" ]; then
    echo "== cross-validation fold $n of $FOLDS: impute =="
    blat impute --run "$run" --mask "$MASK" --fold "$n" $IMPUTE --cells \
      --out "$RESULTS/impute/fold$n"
  fi
done

# ==== Step 6: pool the folds ============================================================
# needs: every imputed fold of Step 5 (a few seconds; no blat call)
echo "== cross-validation over $FOLDS folds =="
"$PYTHON" - "$RESULTS/impute" "$FOLDS" <<'EOF'
import sys
from pathlib import Path
from blat import scores
d, folds = Path(sys.argv[1]), int(sys.argv[2])
pooled = scores.combine_scores([d / f"fold{n}" for n in range(1, folds + 1)])
scores.write(pooled, d / "cross_validation.tsv")
scores.write(scores.by_kind(pooled), d / "cross_validation_by_kind.tsv")
for row in pooled.itertuples():
    print(f"  {row.variable:12s} {row.kind:12s} R^2 = {row.r2:6.3f} +/- {row.r2_se:.3f}"
          f"  (n = {row.n})")
EOF

# ==== Notebooks =========================================================================
# Each needs only its own inputs: figure1 and figure3 Step 1, figure2 Step 2, figure4 and
# figure6 Step 4, figure5 Step 5.  They are independent of each other.
[ "$NOTEBOOKS" = 1 ] || exit 0
echo "== notebooks =="
"$PYTHON" -c "import nbconvert, ipykernel, matplotlib, sklearn" 2>/dev/null || {
  echo "the notebooks need nbconvert, ipykernel, matplotlib and scikit-learn in $PYTHON" >&2
  exit 2; }
# A private kernel for this interpreter, so that a user-level "python3" kernel pointing
# at some other environment is not used instead.
KERNELS="$RESULTS/notebooks/.jupyter"
mkdir -p "$KERNELS/kernels/blat-example"
"$PYTHON" - "$KERNELS/kernels/blat-example/kernel.json" <<'EOF'
import json, sys
json.dump({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
           "display_name": "BLAT example", "language": "python"}, open(sys.argv[1], "w"))
EOF
for nb in "$HERE"/notebooks/figure*.ipynb; do
  echo "executing $(basename "$nb")"
  BLAT_RESULTS="$RESULTS" JUPYTER_PATH="$KERNELS" "$PYTHON" -m nbconvert --to notebook \
    --execute --ExecutePreprocessor.kernel_name=blat-example \
    --ExecutePreprocessor.timeout=3600 --output-dir "$RESULTS/notebooks" "$nb"
done
ls "$RESULTS/figures"
