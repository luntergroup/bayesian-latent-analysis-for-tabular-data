# Settings for a quick example run (about half an hour).  Use:
#
#   source quick.sh
#   ./run_all.sh
#
# Relative paths are relative to the directory holding this file.

export BLAT_SETTINGS=quick
export RESULTS=example/results-quick

# bare models (run_model.sh): 2 chains of 3,000 iterations for K = 2..6
export K_GRID="2 3 4 5 6"
export CHAINS=2
export ALPHA_FROZEN=200       # iterations with alpha held at its initial value
export BURNIN=500             # burn-in iterations in total, the frozen ones included
export ITERS=2500             # sampling iterations
export THIN=5
# thermodynamic integration: 30 temperatures x (30 + 70) = 3,000 iterations per path,
# as many as one chain
export TI_TEMPERATURES=30
export TI_BURN=30
export TI_ITERS=70
export TI_INITIAL_BURN=50
export TI_REPLICATES=1

# final model (run_final.sh); the final chains and each cross-validation fold use the
# chain settings above
export K_FINAL=5
export FINAL_CHAINS=$CHAINS   # final chains, as many as the bare chains per K
export CAL_ROUNDS=2
export CAL_ITERS=300
export CAL_BURN=50
export CAL_DRAWS=50
export EXPLAIN_BURN=100
export EXPLAIN_DRAWS=50
export IMPUTE_PHI_DRAWS=10
export IMPUTE_THETA_DRAWS=5
export IMPUTE_BURN=50
