"""`impute`: fill cells from a fitted run.

Which table and which cells:
  * the run's own table, no mask      -> every missing cell;
  * a table and a mask (--fold n)     -> the masked cells, which are also left out of the
                                         record when inferring theta_d; the truth is known,
                                         so they are scored;
  * new records (--data)              -> their missing cells (or masked cells, with a mask).

In every case theta_d is re-inferred per record with phi and alpha fixed at stored
posterior draws of the run (`record.infer_theta`), from the record's remaining cells; phi
is never updated.  The predictive of a cell pools every (phi draw, theta draw) pair.

Point estimates: the mode for categorical and ordinal cells, the mean for continuous
ones (configurable).  Scores are on the model's scales -- level codes, and y for
continuous variables, before any back-transformation -- and are written as sums
(`scores.py`), so folds combine exactly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import convert as C
from .encoding import encode
from . import scores as S
from .formats import write_table
from .io import Inputs, header, locate, output_dir, relative, write_summary
from .quantities import block_levels, column_slice
from .record import (GRID, Predictive, batch, infer_theta, log_density_y,
                     predict_continuous)
from .runs import Run
from .scores import squared_errors, variable_sums

CHUNK = 512


def predictives(run: Run, codes: np.ndarray, hidden: np.ndarray, targets: np.ndarray,
                phi_draws: int, theta_draws: int, burn: int, seed: int,
                progress: bool = True):
    """{(row, column): Predictive} for every target cell, and the p1 draws of continuous ones."""
    decl, cfg = run.decl, run.cfg
    data = encode(codes, decl, cfg, hidden=hidden)
    layout = data.layout
    rows = np.flatnonzero(targets.any(axis=1))
    picks = run.spread_draws(phi_draws)
    phi = run.phi_draws[picks]
    alpha = run.alpha_draws[picks]
    tokens = batch(data, rows)
    t0 = time.time()
    theta = np.stack([infer_theta(tokens, phi[s], alpha[s], cfg.gamma, seed + 7919 * s,
                                  burn=burn, draws=theta_draws)
                      for s in range(len(picks))])                     # (S, R, J, K)
    if progress:
        print(f"  impute: theta for {rows.size} records x {len(picks)} phi draws "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    out, p1_draws = {}, {}
    for j, v in enumerate(decl.variables):
        in_rows = np.flatnonzero(targets[rows, j])
        if in_rows.size == 0:
            continue
        for start in range(0, in_rows.size, CHUNK):
            part = in_rows[start:start + CHUNK]
            th = theta[:, part]
            if v.kind == "continuous":
                base = layout.vocab_offset[layout.subs[v.name][0]]
                p1 = np.einsum("srjk,sk->rsj", th, phi[:, :, base + 1]).reshape(part.size, -1)
                for r, p in zip(part, p1):
                    density = predict_continuous(p, data.words[v.name], cfg.gamma)
                    out[(rows[r], v.name)] = Predictive("continuous", density=density)
                    p1_draws[(rows[r], v.name)] = p
            else:
                block = np.einsum("srjk,skv->srjv", th,
                                  phi[:, :, column_slice(layout, v.name)])  # (S, R, J, V_f)
                pmf = block_levels(block, v.kind).mean(axis=(0, 2))
                for r, p in zip(part, pmf):
                    out[(rows[r], v.name)] = Predictive(v.kind, pmf=p / p.sum())
    return out, p1_draws, data


def baselines(codes: np.ndarray, usable: np.ndarray, decl) -> dict:
    """The marginal prediction per variable, from the cells the fit could use."""
    out = {}
    for j, v in enumerate(decl.variables):
        values = codes[usable[:, j], j]
        if v.kind == "continuous":
            out[v.name] = {"mean": float(values.mean()) if values.size else 0.5}
        else:
            counts = np.bincount(values.astype(int), minlength=v.n_levels).astype(float)
            pmf = counts / counts.sum() if counts.sum() else np.full(v.n_levels,
                                                                      1 / v.n_levels)
            out[v.name] = {"pmf": pmf, "mean": float(np.arange(v.n_levels) @ pmf)}
    return out


def impute(run_path, out, data_path=None, mask_path=None, fold=None,
           point_discrete="mode", point_continuous="mean", phi_draws=20, theta_draws=10,
           burn=100, seed=0, intervals=False, cells=False, progress=True) -> str:
    """Impute; write OUT/summary.yaml and imputed.tsv, and with a mask scores.tsv
    (plus intervals.tsv and cells.tsv if asked)."""
    run = Run(run_path)
    decl, cfg = run.decl, run.cfg
    own_table = data_path is None
    inputs = Inputs(run.inputs.declaration_path,
                    run.inputs.data_path if own_table else data_path, mask_path, fold)
    codes = inputs.codes
    observed = np.isfinite(codes)
    masked = mask_path is not None
    warnings = []
    if masked:
        targets = inputs.hidden & observed
        hidden = targets
        if own_table:
            ran = run.record["inputs"]
            same = (ran.get("mask") is not None
                    and locate(ran["mask"], run.path).resolve() == Path(mask_path).resolve()
                    and ran.get("fold") == fold)
            if not same:
                warnings.append("the run was not fitted with this mask and fold, so phi has "
                                "seen the masked cells: the scores are optimistic")
    else:
        targets = ~observed
        hidden = np.zeros_like(observed)
    if progress:
        print(f"impute: {int(targets.sum())} cells in {int(targets.any(axis=1).sum())} "
              f"records", file=sys.stderr, flush=True)
    preds, p1_draws, data = predictives(run, codes, hidden, targets, phi_draws, theta_draws,
                                        burn, seed, progress)

    names = decl.names
    col = {n: j for j, n in enumerate(names)}
    index = inputs.frame.index
    # the completed table, in original units / level labels
    completed = C.from_codes(pd.DataFrame(codes, index=index, columns=names), decl)
    for v in decl.of_kind("continuous"):
        completed[v.name] = [_number(x) for x in C.inverse(codes[:, col[v.name]],
                                                              v.transform)]
    detail = []
    for (r, name), pred in preds.items():
        v = decl[name]
        how = point_continuous if v.kind == "continuous" else point_discrete
        point = pred.point(how)
        lo, hi = pred.interval()
        row = dict(id=index[r], variable=name, kind=v.kind)
        if v.kind == "continuous":
            if how == "mean":
                w = pred.density / pred.density.sum()
                shown = float(C.inverse(GRID, v.transform) @ w)
            else:
                shown = float(C.inverse(point, v.transform))
            completed.iat[r, col[name]] = _number(shown)
            row.update(imputed=_number(shown), lo=_number(C.inverse(lo, v.transform)),
                       hi=_number(C.inverse(hi, v.transform)), imputed_y=point, lo_y=lo,
                       hi_y=hi)
        else:
            completed.iat[r, col[name]] = v.levels[int(point)]
            row.update(imputed=v.levels[int(point)], imputed_code=int(point),
                       probabilities=";".join(f"{l}={p:.3g}" for l, p in
                                              zip(v.levels, pred.pmf)))
        if masked:
            truth = codes[r, col[name]]
            if v.kind == "continuous":
                row.update(truth=_number(C.inverse(truth, v.transform)), truth_y=truth,
                           mean_y=pred.mean(),
                           log_density=log_density_y(p1_draws[(r, name)], truth,
                                                     data.words[name], cfg.gamma))
            else:
                row.update(truth=v.levels[int(truth)],
                           log_density=float(np.log(max(pred.pmf[int(truth)], 1e-300))))
        row["_r"] = r
        detail.append(row)

    out = output_dir(out)
    record = header("impute")
    record.update(run=relative(run.path, out), inputs=inputs.describe(out),
                  mode="mask" if masked else ("new records" if not own_table else "missing"),
                  settings=dict(point_discrete=point_discrete,
                                point_continuous=point_continuous, phi_draws=phi_draws,
                                theta_draws=theta_draws, burn=burn, seed=seed),
                  cells=int(targets.sum()), records=int(targets.any(axis=1).sum()),
                  warnings=warnings, tables={"imputed.tsv": "the table with every imputed "
                                             "cell filled in, in original units"})
    frame = pd.DataFrame(detail)
    if masked and len(frame):
        base = baselines(codes, observed & ~hidden, decl)
        sums = {}
        for v in decl.variables:
            sel = frame[frame["variable"] == v.name]
            if sel.empty:
                continue
            rr = sel["_r"].to_numpy()
            truth = codes[rr, col[v.name]]
            fit = dict(baseline_mean=base[v.name]["mean"])
            if v.kind == "continuous":
                fit.update(mean=sel["mean_y"].to_numpy(float))
                point = fit["mean"]
                interval = dict(lo=sel["lo_y"].to_numpy(float),
                                hi=sel["hi_y"].to_numpy(float))
            else:
                fit.update(pmf=np.stack([preds[(r, v.name)].pmf for r in rr]),
                           baseline_pmf=base[v.name]["pmf"])
                point, interval = sel["imputed_code"].to_numpy(int), {}
            sums[v.name] = variable_sums(v.kind, truth, point, **fit, **interval,
                                         log_density=sel["log_density"].to_numpy(float))
            a, b = squared_errors(v.kind, truth, **fit)
            frame.loc[sel.index, "a"] = a
            frame.loc[sel.index, "b"] = b
        scores = S.table(sums)
        S.write(scores, out / "scores.tsv")
        pooled = S.by_kind(scores)
        record["by_kind"] = {row["kind"]: {m: row[m] for m in ("n", "r2", "r2_se",
                                                                "accuracy", "coverage_95",
                                                                "mean_log_density")}
                             for row in pooled.to_dict("records")}
        record["tables"]["scores.tsv"] = ("per variable: R^2 and other scores, and the "
                                          "sums they are computed from (add over folds)")

    write_table(completed.reset_index(), out / "imputed.tsv")
    if intervals and len(frame):
        keep = [c for c in ("id", "variable", "kind", "imputed", "lo", "hi", "imputed_y",
                            "lo_y", "hi_y", "probabilities") if c in frame.columns]
        write_table(frame[keep], out / "intervals.tsv")
        record["tables"]["intervals.tsv"] = "per imputed cell: 95% interval, level probabilities"
    if cells and masked and len(frame):
        keep = [c for c in ("id", "variable", "kind", "truth", "imputed", "truth_y",
                            "imputed_y", "lo_y", "hi_y", "a", "b", "log_density")
                if c in frame.columns]
        table = frame[keep].copy()
        table.insert(2, "fold", fold if fold is not None else "")
        write_table(table, out / "cells.tsv")
        record["tables"]["cells.tsv"] = "per scored cell: truth, prediction, a_i, b_i"
    write_summary(out, record)
    if progress:
        print(f"impute: wrote {out}", file=sys.stderr)
        for w in warnings:
            print(f"  warning: {w}", file=sys.stderr)
    return str(out)


def _number(x) -> str:
    """A value in original units, as text with 4 significant digits."""
    x = float(x)
    return "" if not np.isfinite(x) else f"{x:.4g}"
