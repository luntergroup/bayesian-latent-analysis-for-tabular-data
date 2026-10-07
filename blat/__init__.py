"""Bayesian Latent Analysis of Tabular data (BLAT).

A mixed-membership model for tables of categorical, ordinal and continuous variables:
every record is a mixture `theta_d` over K latent types, and every type a distribution
`phi_k` over the values of every variable.  Fitted by collapsed Gibbs sampling.

Steps, each a subcommand of `python -m blat`:

    convert     raw .csv/.tsv  ->  .yaml declaration + .preproc.tsv (and optionally a mask)
    run         one chain      ->  a directory: summary.yaml, tables (.tsv), arrays.npz
    calibrate   estimate the residual sd, hence latent words N_f, per continuous variable
    ti          log p(x | K) by thermodynamic integration, forward or backward
    explain     out-of-sample variance explained per variable
    impute      fill missing or masked cells, of the fitted table or of new records
    simulate    synthetic data from the model itself

"""

__version__ = "1.0.0"
