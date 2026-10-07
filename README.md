# BLAT: Bayesian Latent Analysis of Tabular data

A standalone implementation of the model in the accompanying paper: every record of a
table is a mixture theta_d over K latent *types*, and every type a distribution phi_k over
the values of every variable.  Categorical, ordinal and continuous variables are modelled
together; missing cells are marginalised.  Inference is by collapsed Gibbs sampling, with
the Dirichlet concentration alpha sampled under an exponential hyperprior.

For details please refer to: 

    D. Neijzen, H.C. Donker, J.M. Vonk and G. Lunter (2026), Unsupervised learning in 
    heterogeneous tabular data: Application to a respiratory disease cohort.  Journal of
    Biomedical Informatics.

## Install

    pip install -e .                 # numpy, scipy, pandas, numba
    pip install -e ".[figures,test]" # + matplotlib, scikit-learn, jupyter, pytest

Without installing, `PYTHONPATH=/path/to/this/folder python -m blat ...` works too.

## Testing

Optional.  The tests need the full installation (`pip install -e ".[figures,test]"`):

    pytest                  # runs all tests except the mini runthrough (half a minute)
    pytest --all            # runs all tests (about 2 minutes)

## The example analysis

    source quick.sh         # or thorough.sh -- use source, do not execute
    ./run_all.sh

`quick.sh` fits K = 2..6 with 2 chains of 3,000 iterations each and takes about
half an hour; `thorough.sh` uses the paper's settings -- K = 2..8, 20 chains of
34,000 iterations, a longer temperature ladder -- and takes about 75 hours run
sequentially.  (To use several cores, source it in several terminals and run
`./run_model.sh K` for different K in each; `./run_all.sh` then skips what is done.)
`run_all.sh` refuses to start without one of them.  Results go to
`example/results-quick/` or `example/results-thorough/`, figures to its `figures/`.
The notebooks need the `figures` extras (see Install); set `NOTEBOOKS=0` to skip
them.

To open the notebooks yourself, start Jupyter from the top directory (`jupyter lab`) and
set `RESULTS` in each notebook's first cell to your run's directory
(`example/results-quick` by default, or `example/results-thorough`).

The steps are four scripts, run one after another:

| script                 | 
|------------------------|---------------------------------------------------------------
| `make_example_data.sh` | simulates a table of 2,500 records and 33 variables from the 
|                        | model (5 true types), converts it, and makes a 5-fold mask; 
|                        | into `example/data/` (already there, so this is only needed to 
|                        | change the data)
| `run_model.sh K`       | the bare model for one K: several chains, then thermodynamic 
|                        | integration forward and backward (Figs 1, 2, 3)
| `run_final.sh K`       | the bare model for K if it has not been run; calibration of the 
|                        | latent words; the final chains (as many as the bare chains;
|                        | chain 1 is the final model); `explain`; cross-validated
|                        | imputation over the mask's folds and imputation of new records;
|                        | then the notebooks, which draw Figs 1-6
| `run_all.sh`           | all of the above: the data if missing, `run_model.sh` for every K 
|                        | in `K_GRID`, `run_final.sh K_FINAL`

Every setting is an environment variable, set by the settings files `quick.sh` and
`thorough.sh`; the step scripts also have defaults at the top, used when they are run on
their own.  Steps whose outputs exist are skipped, so an interrupted run resumes.

The simulated table is not read from any model file: `simulate` draws its own parameters
from the model's priors.

## The steps

    python -m blat <step> --help

| step        | does                               | writes
|-------------|------------------------------------|----------------------------------------
| `convert`   | raw .csv/.tsv -> declaration and   | `P.draft.yaml`, or `P.yaml` +
|             | preprocessed table; optionally a   | `P.preproc.tsv` + `P.report.tsv` (+
|             | mask                               | `P.mask.tsv`)
| `run`       | fits one chain                     | directory: `summary.yaml`, `types.tsv`,
|             |                                    | `phi.tsv`, `theta.tsv`,
|             |                                    | `diagnostics.tsv`, `arrays.npz`
| `calibrate` | estimates the residual sd s_f,     | directory: `declaration.yaml` (with
|             | hence the latent word count N_f,   | `words` filled in), `trace.tsv`,
|             | of each continuous variable        | `summary.yaml`
| `ti`        | log p(x \| K) by thermodynamic     | directory: `summary.yaml` (+
|             | integration, forward from the      | `per_temperature.tsv`, `arrays.npz`)
|             | prior or backward from a run       |
| `explain`   | out-of-sample variance explained   | directory: `summary.yaml`, `scores.tsv`
|             | per variable                       |
| `impute`    | fills missing or masked cells, of  | directory: `summary.yaml`,
|             | the fitted table or of new records | `imputed.tsv` (+ `scores.tsv`,
|             |                                    | `intervals.tsv`, `cells.tsv`)
| `simulate`  | synthetic data from the model      | raw table, declaration, true alpha,
|             |                                    | theta and phi

**Formats.**  Settings and summaries are YAML; tables are tab-separated.  Large
arrays (traces, posterior draws) are in `arrays.npz`.  Numbers are written to 4
significant digits.  Paths inside a `summary.yaml` are relative to its directory.

### convert

Two passes, because the kinds of the variables and the order of ordinal levels cannot
be inferred from the raw data:

    python -m blat convert table.csv --id id --out mydata      # writes mydata.draft.yaml
    
    # edit mydata.draft.yaml:
    # - types, level lists, ordinal order (set confirmed to true), exclusions
    
    python -m blat convert table.csv --decl mydata.draft.yaml --out mydata [--make-mask 5]

The draft takes text columns with few values as categorical, numeric columns with few
integer values as ordinal, other numeric columns as continuous, and excludes anything else.  
The second pass fits the continuous transforms -- log if all values are positive and the 
skew exceeds 1, winsorize at the 0.5% and 99.5% quantiles, map linearly onto [0.005, 0.995]
-- and stores their parameters in `mydata.yaml`.  Choices can be overridden in the 
declaration (`transform`).  Dirichlet concentration (`beta`) parameters, the residual sd 
(`residual_sd`) or the latent word count (`words`) and the model and run settings 
(`model`, `run`) can also be overridden.  Command-line options override the declaration.

In the YAML, `no`, `yes`, `on` and `off` are read as text; level labels that look like 
numbers must be quoted (`'01'`).

The preprocessed table holds level labels for discrete cells and the transformed value
y for continuous ones.  New records are converted with the *stored* transform:

    python -m blat convert new.csv --decl mydata.yaml --apply --out new

`--make-mask 0.1` writes a binary mask (1 = masked; 10% of observed cells); `--make-mask 5`
a 5-fold mask (every observed cell gets a fold 1..5, missing cells 0).  Variables listed
together in the declaration's `mask_groups` are masked together.

### run, and masks

    python -m blat run --decl mydata.yaml --k 5 --seed 1 --out runs/K5-1
    python -m blat run --decl mydata.yaml --k 5 --mask mydata.mask.tsv --fold 2 --out cv/run2

One chain per call.  Masked cells are left out of the fit and scored (log predictive
density) along the way.  `phi.tsv` holds, per variable, level and type, the posterior mean
and 95% interval of the level probability -- or, for a continuous variable, of the value
the type expects, in original units and on the y scale; `theta.tsv` the same for every
record's mixture; `types.tsv` each type's usage and alpha.  `arrays.npz` holds the
per-iteration traces (burn-in included), the per-draw phi and alpha, and the last draw's
counts.

### ti

    python -m blat ti --decl mydata.yaml --k 5 --direction forward --out ti/K5-f
    python -m blat ti --from-run runs/K5-1 --direction backward --out ti/K5-b

A backward path starts from a posterior draw of the run (theta, phi and alpha rebuilt from
its last stored counts, and the assignments drawn exactly given them), so no sampler state
needs storing.  Forward and backward estimates should agree within their errors; each run
reports one estimate, and comparing or combining them is left to the user (the Fig 2
notebook shows both).

### impute

    python -m blat impute --run cv/run2 --mask mydata.mask.tsv --fold 2 --out cv/imp2
    python -m blat impute --run runs/K5-1 --data new.preproc.tsv --out new-imputed

theta_d is re-inferred for every record from its remaining cells with phi and alpha fixed
at stored posterior draws; phi is never updated.  With a mask the truth is known and each
variable is scored: accuracy, R^2 (Brier for categorical, on the level index for ordinal,
on the y scale for continuous), 95% interval coverage, log predictive density.
`scores.tsv` has one row per variable, with the scores followed by the sums they are
computed from, so that folds combine exactly -- by adding the sum columns, or with

    from blat.scores import combine_scores
    combine_scores(["cv/imp1", "cv/imp2", ...])      # a table like scores.tsv

## Diagnostics

* `convert`: per column the kind, missingness, level counts and never-observed levels,
  log decision, skew before and after, winsorization bounds and the number of values
  clipped; the token budget (share of tokens that are continuous); warnings.
* `run`, every iteration: the collapsed log joint log p(x, z | alpha, beta, gamma)
  (`log_joint`) and its three terms (`log_p_types`, `log_p_words`, `log_p_continuous`),
  alpha and sum alpha, the fraction of tokens reassigned (discrete, continuous), type
  occupancy, K_eff, mean entropy of theta_d, the time per iteration.  Every stored draw:
  the log likelihood log p(x | theta, phi) (`log_lik`); with a mask, the held-out log
  predictive density.  End of chain (sampling phase only, on label-invariant quantities):
  ESS, integrated autocorrelation time, MCSE, Geweke z, drift; the matched cosine of the
  types between the two halves of the sampling phase (label switching); warnings for
  drift, low ESS, near-empty types, unstable labels.
* Across chains (in the notebooks): split R-hat of the log joint and sum alpha, the spread
  of chain means, the medoid chain and each chain's type agreement with it; for the final
  chains, a t-SNE of every chain's types, each matched to a type of the medoid (Fig 6).
* `calibrate`: s_f and N_f per variable and iteration, floored estimates, whether the
  last two iterations agree to within one word.
* `ti`: the estimate with its Monte Carlo and quadrature errors; optionally, per
  temperature, E[log L], its standard error and ESS, the difference between the halves of
  the iterations, and sum alpha; for a backward path, the log joint over the initial
  burn-in.
* `explain` / `impute`: see above.

## Tests

    pytest                       # all but the slow test, about half a minute
    pytest --all                 # + run_all.sh end to end, about two minutes in all

The sampler is checked against the exact posterior, and thermodynamic integration against
the exact evidence, by enumerating every assignment of a three-record table.  Further
tests cover the transforms, masks, the per-record inference, the score sums, the
simulator, and every step through the command line; `--all` adds `run_all.sh` with
every setting shrunk.
