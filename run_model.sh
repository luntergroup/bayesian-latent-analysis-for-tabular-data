#!/usr/bin/env bash
# Fit the bare (uncalibrated) model for one K, and estimate its evidence.
#
#   ./run_model.sh K
#
# Runs $CHAINS chains one after another, then thermodynamic integration forward (from the
# prior) and backward (from chain 1), $TI_REPLICATES times each.  Outputs go to
#   $RESULTS/runs/K<K>-c<chain>/               (Figs 1, 3)
#   $RESULTS/ti/K<K>-<direction>-<n>/          (Fig 2)
# Steps whose output exists are skipped, so an interrupted run resumes.
#
# Every `blat` call is one single-core process that communicates only through files, so
# calls that do not depend on each other can run at the same time, e.g. as cluster jobs.
#
# This script holds Steps 1 and 2 of the recipe in run_all.sh:
#   Step 1  the chains, and the forward TI paths: need only the data, all independent
#   Step 2  the backward TI paths: each needs chain 1 at this K
# Different K are entirely independent: run_model.sh 2, 3, ... can run side by side.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- settings (edit here; each can also be set in the environment) --------------------
# The defaults take a few minutes per chain on 2,500 records.  The paper used
# CHAINS=20, ALPHA_FROZEN=2000, BURNIN=4000, ITERS=30000, THIN=10.  A TI path takes
# at least as many iterations as a chain: TI_TEMPERATURES x (TI_BURN + TI_ITERS)
# >= BURNIN + ITERS; the defaults give 30 x (60 + 140) = 6,000.
PYTHON="${PYTHON:-python3}"
DATA="${DATA:-$HERE/example/data}"   # made by make_example_data.sh
NAME="${NAME:-example}"
RESULTS="${RESULTS:-$HERE/example/results}"
CHAINS="${CHAINS:-3}"                # independent chains
ALPHA_FROZEN="${ALPHA_FROZEN:-500}"  # iterations with alpha held at its initial value
BURNIN="${BURNIN:-1000}"             # burn-in iterations in total, the frozen ones included
ITERS="${ITERS:-5000}"               # sampling iterations
THIN="${THIN:-10}"                   # keep every THIN-th sampling iteration
TI_TEMPERATURES="${TI_TEMPERATURES:-30}"
TI_BURN="${TI_BURN:-60}"             # iterations discarded at each temperature
TI_ITERS="${TI_ITERS:-140}"          # iterations averaged at each temperature
TI_INITIAL_BURN="${TI_INITIAL_BURN:-200}"
TI_REPLICATES="${TI_REPLICATES:-1}"  # paths per direction
# ----------------------------------------------------------------------------------------
# relative paths are relative to this directory
absolute() { case "$1" in /*) echo "$1" ;; *) echo "$HERE/$1" ;; esac; }
DATA="$(absolute "$DATA")"
RESULTS="$(absolute "$RESULTS")"

K="${1:?usage: run_model.sh K}"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
blat() { "$PYTHON" -m blat "$@"; }
TABLE="$DATA/$NAME.yaml"
[ -f "$TABLE" ] || { echo "no $TABLE: run make_example_data.sh first" >&2; exit 2; }
mkdir -p "$RESULTS/runs" "$RESULTS/ti"

# ==== Step 1: bare models -- the chains and the forward TI paths ========================
# All jobs in this step need only the data and are independent of each other (and of
# the other K).
for c in $(seq 1 "$CHAINS"); do
  out="$RESULTS/runs/K$K-c$c"
  if [ -f "$out/summary.yaml" ]; then echo "exists: $out"; continue; fi
  echo "== run K=$K chain $c =="
  blat run --decl "$TABLE" --k "$K" --seed "$((1000 * K + c))" \
    --alpha-frozen "$ALPHA_FROZEN" --burnin "$BURNIN" --iters "$ITERS" --thin "$THIN" \
    --out "$out"
done

TI="--temperatures $TI_TEMPERATURES --burn-per-temperature $TI_BURN
    --iters-per-temperature $TI_ITERS --initial-burn $TI_INITIAL_BURN --per-temperature"
for r in $(seq 1 "$TI_REPLICATES"); do
  out="$RESULTS/ti/K$K-forward-$r"
  if [ -f "$out/summary.yaml" ]; then echo "exists: $out"; continue; fi
  echo "== ti K=$K forward $r =="
  blat ti --decl "$TABLE" --k "$K" --direction forward --seed "$r" $TI --out "$out"
done

# ==== Step 2: the backward TI paths =====================================================
# Each needs chain 1 at this K (it starts from that run's last draw); the paths are
# independent of each other.
for r in $(seq 1 "$TI_REPLICATES"); do
  out="$RESULTS/ti/K$K-backward-$r"
  if [ -f "$out/summary.yaml" ]; then echo "exists: $out"; continue; fi
  echo "== ti K=$K backward $r =="
  blat ti --from-run "$RESULTS/runs/K$K-c1" --direction backward --seed "$r" $TI \
    --out "$out"
done
