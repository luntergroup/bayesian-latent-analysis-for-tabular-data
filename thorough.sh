# Settings for the thorough example run: the paper's settings.  Use:
#
#   source thorough.sh
#   ./run_all.sh
#
# Sequentially this takes about 75 hours, almost all of it the 160 chains.  To use several
# cores, source this file in several terminals and run `./run_model.sh K` for different K
# in each; then `./run_all.sh` skips what is done and runs the final model and notebooks.
#
# Relative paths are relative to the directory holding this file.

export BLAT_SETTINGS=thorough
export RESULTS=example/results-thorough

# bare models (run_model.sh): 20 chains of 34,000 iterations for K = 2..8
export K_GRID="2 3 4 5 6 7 8"
export CHAINS=20
export ALPHA_FROZEN=2000      # iterations with alpha held at its initial value
export BURNIN=4000            # burn-in iterations in total, the frozen ones included
export ITERS=30000            # sampling iterations
export THIN=10
# thermodynamic integration: 50 temperatures x (200 + 480) = 34,000 iterations per path,
# as many as one chain
export TI_TEMPERATURES=50
export TI_BURN=200
export TI_ITERS=480
export TI_INITIAL_BURN=200
export TI_REPLICATES=3

# final model (run_final.sh); the final chains and each cross-validation fold use the
# chain settings above
export K_FINAL=5
export FINAL_CHAINS=$CHAINS   # final chains, as many as the bare chains per K
export CAL_ROUNDS=4
export CAL_ITERS=1000
export CAL_BURN=200
export CAL_DRAWS=200
export EXPLAIN_BURN=500
export EXPLAIN_DRAWS=200
export IMPUTE_PHI_DRAWS=20
export IMPUTE_THETA_DRAWS=10
export IMPUTE_BURN=100
