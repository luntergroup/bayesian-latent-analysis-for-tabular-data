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

Optional; needs the full installation.

    pytest                  # all tests except the end-to-end run (half a minute)
    pytest --all            # all tests (about 2 minutes)

## The example analysis

    source quick.sh         # or thorough.sh -- use source, do not execute
    ./run_all.sh

`quick.sh` takes about half an hour; `thorough.sh` uses the paper's settings and takes
about 75 hours sequentially (to use several cores, run `./run_model.sh K` for different K
in separate terminals first).  Results go to `example/results-quick/` or
`example/results-thorough/`, the figures to its `figures/`.  Steps whose outputs exist
are skipped, so an interrupted run resumes.  To open the notebooks yourself, start
Jupyter from the top directory and set `RESULTS` in the first cell.

The steps are four scripts, run one after another:

| script                 | does                                                          |
|------------------------|---------------------------------------------------------------|
| `make_example_data.sh` | simulates the example table, converts it, makes a 5-fold mask |
| `run_model.sh K`       | the bare model for one K: chains, TI both ways (Figs 1-3)     |
| `run_final.sh K`       | calibration, final chains, explain, imputation (Figs 1-6)     |
| `run_all.sh`           | all of the above: every K in `K_GRID`, then `K_FINAL`         |

The example data (2,500 records, 33 variables, 5 true types) are already in
`example/data/`.  Of the final chains, chain 1 is the final model; the others check that
it is reproducible (Fig 6).  The settings are environment variables, set by `quick.sh`
and `thorough.sh`.

## The steps

    python -m blat <step> --help

| step        | does                                                                    |
|-------------|-------------------------------------------------------------------------|
| `convert`   | raw .csv/.tsv -> declaration and preprocessed table; optionally a mask  |
| `run`       | fits one chain                                                          |
| `calibrate` | estimates s_f, hence the latent words N_f, of each continuous variable  |
| `ti`        | the log evidence of K by thermodynamic integration, forward or backward |
| `explain`   | out-of-sample variance explained per variable                           |
| `impute`    | fills missing or masked cells, of the fitted table or of new records    |
| `simulate`  | synthetic data from the model                                           |

Every step but `convert` and `simulate` writes a directory holding `summary.yaml`
(settings, results and warnings), tab-separated tables, and large arrays in `arrays.npz`.

### convert

Two passes are needed because the kinds of the variables and the order of ordinal levels 
cannot be inferred from the raw data:

    python -m blat convert table.csv --id id --out mydata      # writes mydata.draft.yaml

    # edit mydata.draft.yaml and check types, level lists, ordinal order (set confirmed to 
    # true), exclusions

    python -m blat convert table.csv --decl mydata.draft.yaml --out mydata [--make-mask 5]

The second pass fits the continuous transforms and stores them in `mydata.yaml`, which
is read by later steps.  The declaration's header comment lists what can be set there
per variable and for the model.  Command-line options can override these.  Level labels 
that look like numbers must be quoted (`'01'`).  New records are converted with the stored
transforms:

    python -m blat convert new.csv --decl mydata.yaml --apply --out new

`--make-mask 0.1` hides 10% of the observed cells; `--make-mask 5` assigns every observed
cell to one of 5 folds.

### run

    python -m blat run --decl mydata.yaml --k 5 --seed 1 --out runs/K5-1
    python -m blat run --decl mydata.yaml --k 5 --mask mydata.mask.tsv --fold 2 --out cv/run2

One chain per call; masked cells are left out of the fit.

### ti

    python -m blat ti --decl mydata.yaml --k 5 --direction forward --out ti/K5-f
    python -m blat ti --from-run runs/K5-1 --direction backward --out ti/K5-b

Forward and backward estimates should agree within their errors (see Fig 2).

### impute

    python -m blat impute --run cv/run2 --mask mydata.mask.tsv --fold 2 --out cv/imp2
    python -m blat impute --run runs/K5-1 --data new.preproc.tsv --out new-imputed

With a mask, the imputed cells are scored against the truth.  `scores.tsv` holds the
scores per variable and the sums they are computed from, so that folds combine exactly:

    from blat.scores import combine_scores
    combine_scores(["cv/imp1", "cv/imp2", ...])      # a table like scores.tsv

## Diagnostics

`convert` reports problems with the columns in `mydata.report.tsv` and the declaration's
`report` section.  A run's `summary.yaml` warns of drift, low effective sample size,
near-empty types and label switching, with the details in `diagnostics.tsv`.  The
notebooks show how well the chains agree (Figs 1 and 6); `ti` reports its Monte Carlo
and integration errors.
