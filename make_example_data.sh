#!/usr/bin/env bash
# Make the example data: simulate a table from the BLAT model, then convert it.
#
#   ./make_example_data.sh
#
# Writes, in $DATA:
#   $NAME.raw.csv              the raw table, as a user would have it
#   $NAME.new.raw.csv          new records with extra holes, for imputation
#   $NAME.declaration.yaml     the variable declaration (types, levels)
#   $NAME.truth.yaml/.tsv/.npz kappa, the drawn alpha, theta and phi, per-variable settings
#   $NAME.yaml                 the fitted declaration (with the continuous transforms)
#   $NAME.report.tsv           per-variable diagnostics of the conversion
#   $NAME.preproc.tsv          the preprocessed table the model reads
#   $NAME.mask.tsv             a $FOLDS-fold mask for cross-validated imputation
#   new.preproc.tsv            the new records, with the same transforms
#
# Every `blat` call is one single-core process that communicates only through files, so
# calls that do not depend on each other can run at the same time, e.g. as cluster jobs.
#
# Dependencies: the three calls run in order, each needing the previous one's output:
#   simulate -> convert (fits the transforms) -> convert --apply (new records).
# Together they take seconds; nothing here is worth parallelising.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- settings (edit here; each can also be set in the environment) --------------------
PYTHON="${PYTHON:-python3}"
DATA="${DATA:-$HERE/example/data}"           # where the data go
NAME="${NAME:-example}"                      # file name stem
K_TRUE="${K_TRUE:-5}"                        # number of types the data are drawn from
RECORDS="${RECORDS:-2500}"                   # records in the table
CATEGORICAL="${CATEGORICAL:-16}"             # number of categorical variables
ORDINAL="${ORDINAL:-7}"                      # ... ordinal variables
CONTINUOUS="${CONTINUOUS:-10}"               # ... continuous variables
KAPPA_TRUE="${KAPPA_TRUE:-$K_TRUE}"          # alpha_k ~ Exp(kappa); kappa = K makes
                                             # E[sum alpha] = 1, and
                                             # alpha / sum(alpha) ~ Dirichlet(1, ..., 1)
BETA_TRUE="${BETA_TRUE:-0.5}"                # phi_kf ~ Dirichlet(beta, ..., beta)
RESIDUAL_SD_TRUE="${RESIDUAL_SD_TRUE:-0.2}"  # sets the latent words per continuous cell
MAX_MISSING="${MAX_MISSING:-0.3}"            # each variable is missing at rate U(0, this)
NEW_RECORDS="${NEW_RECORDS:-200}"            # new records for imputation
NEW_MISSING="${NEW_MISSING:-0.2}"            # extra missingness in the new records
FOLDS="${FOLDS:-5}"                          # folds of the cross-validation mask
SEED="${SEED:-6}"                            # 6: the first seed whose alpha / sum(alpha)
                                             # is at least 0.06 in every type (K_TRUE = 5)
# ----------------------------------------------------------------------------------------
# relative paths are relative to this directory
absolute() { case "$1" in /*) echo "$1" ;; *) echo "$HERE/$1" ;; esac; }
DATA="$(absolute "$DATA")"

export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
blat() { "$PYTHON" -m blat "$@"; }
mkdir -p "$DATA"

# needs: nothing
echo "== simulate: $RECORDS records, K = $K_TRUE =="
blat simulate --k "$K_TRUE" --records "$RECORDS" --categorical "$CATEGORICAL" \
  --ordinal "$ORDINAL" --continuous "$CONTINUOUS" --kappa "$KAPPA_TRUE" --beta "$BETA_TRUE" \
  --residual-sd "$RESIDUAL_SD_TRUE" --max-missing "$MAX_MISSING" \
  --new-records "$NEW_RECORDS" --new-missing "$NEW_MISSING" --seed "$SEED" \
  --name "$NAME" --out "$DATA"

# needs: simulate (the raw table and its declaration)
echo "== convert, with a $FOLDS-fold mask =="
blat convert "$DATA/$NAME.raw.csv" --decl "$DATA/$NAME.declaration.yaml" \
  --out "$DATA/$NAME" --make-mask "$FOLDS" --seed "$SEED"
# needs: convert above (the fitted declaration $NAME.yaml)
blat convert "$DATA/$NAME.new.raw.csv" --decl "$DATA/$NAME.yaml" --apply --out "$DATA/new"
