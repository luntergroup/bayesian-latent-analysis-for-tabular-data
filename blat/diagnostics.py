"""Convergence diagnostics for scalar traces.

Only for label-invariant quantities (log joint, log lik, sum alpha, K_eff, ...).  Raw
theta_dk or phi_kfw are defined only up to a permutation of the types and give
meaningless values.
Standard errors are autocorrelation-corrected: a naive var/n treats a Gibbs trace as
independent draws.
"""

from __future__ import annotations

import math

import numpy as np


def autocorrelation(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    centred = x - x.mean()
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, size)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:n] / n
    return acov / acov[0] if acov[0] > 0 else np.zeros_like(acov)


def integrated_autocorrelation_time(x: np.ndarray) -> float:
    """Geyer's initial-positive-sequence estimator."""
    rho = autocorrelation(x)
    if rho.size < 3:
        return 1.0
    m = (rho.size - 1) // 2
    pairs = rho[1:2 * m:2] + rho[2:2 * m + 1:2]
    cut = int(np.argmax(pairs < 0)) if np.any(pairs < 0) else pairs.size
    return max(1.0 + 2.0 * float(pairs[:cut].sum()), 1.0)


def effective_sample_size(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 4 or np.allclose(x, x[0]):
        return float(x.size)
    return float(x.size / integrated_autocorrelation_time(x))


def mcmc_standard_error(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return math.nan
    return float(np.sqrt(x.var(ddof=1) / max(effective_sample_size(x), 1.0)))


def geweke_z(x: np.ndarray, first: float = 0.1, last: float = 0.5) -> float:
    """Mean of the first 10% against the last 50%, as an autocorrelation-corrected z."""
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return math.nan
    head = x[:max(int(first * x.size), 2)]
    tail = x[-max(int(last * x.size), 2):]
    denominator = math.hypot(mcmc_standard_error(head), mcmc_standard_error(tail))
    return 0.0 if denominator == 0 else float((head.mean() - tail.mean()) / denominator)


def trend(x: np.ndarray) -> dict:
    """The least-squares drift over the whole trace, against the trace's own spread.

    `drift` is slope * length: how far the level moved from start to end.  Reported next
    to the standard deviation so a reader can see whether it is small.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return {"drift": math.nan, "sd": math.nan}
    slope = np.polyfit(np.arange(x.size), x, 1)[0]
    return {"drift": float(slope * x.size), "sd": float(x.std(ddof=1))}


def split_r_hat(chains: np.ndarray) -> float:
    """Split-R-hat over (n_chains, n_draws); each chain is halved first."""
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    half = chains.shape[1] // 2
    if half < 2:
        return math.nan
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]])
    n = split.shape[1]
    within = split.var(axis=1, ddof=1).mean()
    if within == 0:
        return math.nan
    between = n * split.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def summarise(x: np.ndarray) -> dict:
    """Mean, sd, ESS, IAT, MCSE, Geweke z and drift of one trace."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return {"n": int(x.size)}
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "ess": effective_sample_size(x), "iat": integrated_autocorrelation_time(x),
            "mcse": mcmc_standard_error(x), "geweke_z": geweke_z(x), **trend(x)}


def match_types(a: np.ndarray, b: np.ndarray):
    """Best one-to-one matching of two sets of type vectors by cosine similarity."""
    from scipy.optimize import linear_sum_assignment

    unit = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)
    sim = unit(np.asarray(a, float)) @ unit(np.asarray(b, float)).T
    rows, cols = linear_sum_assignment(-sim)
    return sim, rows, cols


def matched_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine of each type in `a` with its match in `b`."""
    sim, rows, cols = match_types(a, b)
    return sim[rows, cols]


def medoid(phis: list) -> tuple[int, np.ndarray]:
    """The chain whose types agree best, on average, with every other chain's.

    Each candidate is matched one-to-one onto every other set of types and scored by the
    mean matched cosine; an outlying chain is outlying against all the rest and cannot win.
    """
    if len(phis) == 1:
        return 0, np.ones(1)
    scores = np.array([np.mean([matched_cosine(c, o).mean() for o in phis])
                       for c in phis])
    return int(np.argmax(scores)), scores
