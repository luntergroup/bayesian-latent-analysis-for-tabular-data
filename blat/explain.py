"""`explain`: how much of each variable the types account for, out of sample.

For every observed cell of variable f, theta_d is re-inferred from the record's *other*
cells (phi and alpha fixed at their posterior means), so the prediction has not seen the
cell it is scored on.  The in-sample ("apparent") alternative uses the fitted theta_d,
which did see it, and is optimistic.

  categorical  Brier R^2 against the marginal level frequencies
  ordinal      R^2 on the level index against its mean (Brier R^2 alongside)
  continuous   R^2 = 1 - s_f^2 / var(y_f) on the y scale, with the prediction
               y(mu_df), mu_df = (N_f a_df + gamma_1) / (N_f + gamma), a_df = theta_d . phi_f1;
               s_f is the predictive residual sd, the quantity `calibrate` estimates

The results are written as sums (`scores.py`) as well as scores.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .encoding import EncodedData
from . import scores as S
from .io import header, output_dir, relative, write_summary
from .quantities import expected_y, level_probabilities
from .record import batch, infer_theta
from .runs import Run
from .scores import finish, variable_sums


def theta_without(data: EncodedData, column: str, phi, alpha, gamma, burn, draws, seed):
    """(rows, theta) for the records where `column` is observed, re-inferred without it."""
    layout = data.layout
    subs = layout.subs[column]
    if layout.kinds[column] == "continuous":
        rows = np.unique(data.cell_doc[data.cell_sub == subs[0]])
    else:
        rows = np.unique(data.disc_doc[np.isin(data.disc_sub, subs)])
    tokens = batch(data, rows, exclude_subs=subs)
    theta = infer_theta(tokens, phi, alpha, gamma, seed, burn=burn, draws=draws,
                        rao_blackwell=True).mean(axis=1)
    return rows, theta


def score_variable(data: EncodedData, codes: np.ndarray, decl, cfg, column: str, phi,
                   alpha, theta=None, burn=500, draws=200, seed=0) -> dict:
    """Sums for one variable; `theta` given means apparent (in-sample) scoring."""
    j = decl.names.index(column)
    v = decl[column]
    layout = data.layout
    if theta is None:
        rows, th = theta_without(data, column, phi, alpha, cfg.gamma, burn, draws,
                                 seed + 104729 * j)
    else:
        subs = layout.subs[column]
        rows = (np.unique(data.cell_doc[data.cell_sub == subs[0]])
                if v.kind == "continuous"
                else np.unique(data.disc_doc[np.isin(data.disc_sub, subs)]))
        th = theta[rows]
    truth = codes[rows, j]
    tp = th @ phi
    if v.kind == "continuous":
        base = layout.vocab_offset[layout.subs[column][0]]
        n = data.words[column]
        predicted = expected_y(tp[:, base + 1], n, cfg.gamma)
        out = variable_sums("continuous", truth, predicted, mean=predicted,
                            baseline_mean=float(truth.mean()))
        resid = truth - predicted
        out["s_hat"] = float(np.sqrt(resid @ resid / max(truth.size - 1, 1)))
        out["words"] = n
        return out
    pmf = level_probabilities(tp, layout, column)
    counts = np.bincount(truth.astype(int), minlength=v.n_levels).astype(float)
    marginal = counts / counts.sum()
    return variable_sums(v.kind, truth, pmf.argmax(axis=1), pmf=pmf, baseline_pmf=marginal,
                         baseline_mean=float(np.arange(v.n_levels) @ marginal),
                         log_density=np.log(np.clip(pmf[np.arange(truth.size),
                                                        truth.astype(int)], 1e-300, None)))


def score_all(data, codes, decl, cfg, phi, alpha, columns, theta=None, burn=500,
              draws=200, seed=0, workers=1, progress=True) -> dict:
    t0 = time.time()

    def one(column):
        s = score_variable(data, codes, decl, cfg, column, phi, alpha, theta, burn, draws,
                           seed)
        if progress:
            print(f"  explain {column}: R^2 = {finish(s)['r2']:.3f}"
                  f"  ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        return s

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, columns))
    else:
        results = [one(c) for c in columns]
    return dict(zip(columns, results))


def explain(run_path, out, variables=None, apparent=False, burn=500, draws=200,
            seed=0, workers=1, progress=True) -> str:
    """Write OUT/summary.yaml and OUT/scores.tsv (one row per variable)."""
    run = Run(run_path)
    decl, cfg, data = run.decl, run.cfg, run.data
    codes = np.where(run.inputs.hidden, np.nan, run.inputs.codes)
    columns = list(variables) if variables else decl.names
    unknown = set(columns) - set(decl.names)
    if unknown:
        raise ValueError(f"unknown variables {sorted(unknown)}")
    phi = run.phi_mean
    alpha = run.alpha_draws.mean(axis=0)
    if progress:
        print(f"explain: {len(columns)} variables, burn {burn}, draws {draws}",
              file=sys.stderr, flush=True)
    sums = score_all(data, codes, decl, cfg, phi, alpha, columns, None, burn, draws, seed,
                     workers, progress)
    table = S.table(sums)
    table.insert(3, "s_hat", [sums[c].get("s_hat", np.nan) for c in table["variable"]])
    table.insert(4, "words", [sums[c].get("words", np.nan) for c in table["variable"]])
    if apparent:
        in_sample = S.table(score_all(data, codes, decl, cfg, phi, alpha, columns,
                                      run.theta_mean, progress=False))
        table.insert(3, "r2_apparent", in_sample["r2"].to_numpy())
    out = output_dir(out)
    S.write(table, out / "scores.tsv")
    record = header("explain")
    record.update(run=relative(run.path, out),
                  settings=dict(burn=burn, draws=draws, seed=seed, phi="posterior mean",
                                alpha="posterior mean"),
                  tables={"scores.tsv": "per variable: out-of-sample R^2 (r2) and, if asked, "
                                        "in-sample (r2_apparent); s_hat, the predictive "
                                        "residual sd of continuous variables; the sums"})
    write_summary(out, record)
    if progress:
        print(f"explain: wrote {out}", file=sys.stderr)
    return str(out)
