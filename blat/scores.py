"""Out-of-sample scores as combinable sums.

Every score is a function of per-variable sums over the scored cells, and the sums are
written next to the scores (one row per variable in scores.tsv), so scores from several
folds combine by adding the sum columns and recomputing.  For cell i, a_i is the model's
squared error and b_i the baseline's, on the variable's R^2 scale:

  categorical  Brier:   a = ||e - p||^2, b = ||e - p0||^2 (e one-hot truth, p predicted
               level probabilities, p0 the marginal level frequencies)
  ordinal      level:   a = (c - E[c])^2, b = (c - mean c)^2 on the level index;
               the Brier sums are kept alongside
  continuous   y scale: a = (y - E[y])^2, b = (y - mean y)^2, y = arccos(1 - 2x)/pi, the
               preprocessed value (no back-transformation)

R^2 = 1 - sum a / sum b.  Its standard error is by the delta method from n, sum a, sum b,
sum a^2, sum b^2 and sum ab, which -- unlike a bootstrap -- can be combined after the fact.
The sums are written at full precision, since they are added and differenced later.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

RATIO = ("sse", "sse_baseline", "sse_sq", "sse_baseline_sq", "sse_cross")
BRIER = tuple(f"brier_{c}" for c in RATIO)
SUMS = ("n",) + RATIO + BRIER + ("correct", "abs_level_error", "covered",
                                 "log_density", "log_density_sq")
METRICS = ("r2", "r2_se", "brier_r2", "brier_r2_se", "accuracy", "mean_abs_level_error",
           "coverage_95", "mse", "mean_log_density")


def ratio_sums(a, b, prefix: str = "") -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    values = (a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum())
    return {prefix + name: float(v) for name, v in zip(RATIO, values)}


def r2(n: float, a: float, b: float, aa: float, bb: float, ab: float) -> tuple[float, float]:
    """R^2 = 1 - sum a / sum b and its delta-method standard error."""
    if not n or n < 2 or not b > 0:
        return math.nan, math.nan
    ma, mb = a / n, b / n
    ratio = ma / mb
    va, vb, cov = aa / n - ma * ma, bb / n - mb * mb, ab / n - ma * mb
    var = (va - 2 * ratio * cov + ratio * ratio * vb) / (n * mb * mb)
    return 1.0 - ratio, math.sqrt(max(var, 0.0))


def squared_errors(kind: str, truth, *, pmf=None, baseline_pmf=None, mean=None,
                   baseline_mean=None) -> tuple[np.ndarray, np.ndarray]:
    """(a, b) per cell, on the variable's R^2 scale (see the module docstring)."""
    truth = np.asarray(truth, float)
    if kind == "categorical":
        onehot = np.eye(pmf.shape[1])[truth.astype(int)]
        return ((onehot - pmf) ** 2).sum(axis=1), ((onehot - baseline_pmf) ** 2).sum(axis=1)
    if kind == "ordinal":
        mean = pmf @ np.arange(pmf.shape[1])
    return (truth - np.asarray(mean, float)) ** 2, (truth - baseline_mean) ** 2


def variable_sums(kind: str, truth, point, *, pmf=None, baseline_pmf=None, mean=None,
                  baseline_mean=None, lo=None, hi=None, log_density=None) -> dict:
    """The sums for one variable's scored cells.

    Discrete: `truth` and `point` are level codes, `pmf` (n, levels) the predictive and
    `baseline_pmf` the marginal frequencies; ordinal also `baseline_mean` (mean level
    index).  Continuous: `truth` is y, `mean` the predictive mean of y, `baseline_mean` the
    marginal mean, `lo`/`hi` the 95% interval (None to skip coverage).
    """
    truth = np.asarray(truth, float)
    row = {"kind": kind, "n": int(truth.size)}
    if log_density is not None:
        ld = np.asarray(log_density, float)
        row.update(log_density=float(ld.sum()), log_density_sq=float((ld * ld).sum()))
    row.update(ratio_sums(*squared_errors(kind, truth, pmf=pmf, baseline_pmf=baseline_pmf,
                                          mean=mean, baseline_mean=baseline_mean)))
    if kind == "continuous":
        if lo is not None:
            row["covered"] = int(np.sum((truth >= lo) & (truth <= hi)))
        return row
    row["correct"] = int(np.sum(np.asarray(point).astype(int) == truth))
    if kind == "ordinal":
        row.update(ratio_sums(*squared_errors("categorical", truth, pmf=pmf,
                                              baseline_pmf=baseline_pmf), prefix="brier_"))
        row["abs_level_error"] = float(np.sum(np.abs(truth - np.asarray(point))))
    return row


def finish(row: dict) -> dict:
    """The scores of one row of sums."""
    get = lambda k: row.get(k, math.nan)
    n = get("n")
    out = {m: math.nan for m in METRICS}
    out["r2"], out["r2_se"] = r2(n, *(get(c) for c in RATIO))
    if row["kind"] == "ordinal":
        out["brier_r2"], out["brier_r2_se"] = r2(n, *(get(c) for c in BRIER))
        out["mean_abs_level_error"] = get("abs_level_error") / n
    if row["kind"] == "continuous":
        out["coverage_95"] = get("covered") / n
        out["mse"] = get("sse") / n
    else:
        out["accuracy"] = get("correct") / n
    out["mean_log_density"] = get("log_density") / n if n else math.nan
    return out


def table(rows: dict) -> pd.DataFrame:
    """{variable: sums} -> one row per variable, scores first, then the sums."""
    frame = pd.DataFrame([{"variable": name, "kind": row["kind"], **finish(row),
                           **{c: row.get(c, math.nan) for c in SUMS}}
                          for name, row in rows.items()])
    return frame[["variable", "kind", "n", *METRICS, *[c for c in SUMS if c != "n"]]]


def write(frame: pd.DataFrame, path) -> None:
    from .formats import write_table

    write_table(frame, path, exact=SUMS)


def recompute(sums: pd.DataFrame) -> pd.DataFrame:
    """Scores from summed rows (e.g. after grouping)."""
    return table({(r["variable"] if "variable" in r else r["kind"]): r
                  for r in sums.to_dict("records")})


def by_kind(frame: pd.DataFrame) -> pd.DataFrame:
    """Sums pooled over the variables of each kind, and their scores."""
    pooled = frame.groupby("kind", sort=False)[list(SUMS)].sum(min_count=1).reset_index()
    pooled["variable"] = "all " + pooled["kind"]
    return recompute(pooled)


def combine_scores(paths) -> pd.DataFrame:
    """Pool the scores of several impute (or explain) outputs -- e.g. the folds of a
    cross-validation -- by adding their sums per variable, and recompute every score.

    Each path is an output directory or its scores.tsv.
    """
    frames = []
    for p in paths:
        p = Path(p)
        frames.append(pd.read_csv(p if p.suffix == ".tsv" else p / "scores.tsv", sep="\t"))
    if not frames:
        raise ValueError("nothing to combine")
    stacked = pd.concat(frames)
    summed = stacked.groupby(["variable", "kind"], sort=False)[list(SUMS)] \
        .sum(min_count=1).reset_index()
    return recompute(summed)


def headline(frame: pd.DataFrame) -> dict:
    """One R^2 per variable: Brier for categorical, level for ordinal, y scale for continuous."""
    return dict(zip(frame["variable"], frame["r2"]))
