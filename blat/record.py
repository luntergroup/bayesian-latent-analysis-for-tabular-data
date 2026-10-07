"""Per-record inference with phi fixed, and the predictive distribution of a cell.

With phi (and alpha) fixed, records are independent, so theta_d can be re-inferred from
any subset of a record's cells by a small Gibbs chain over that record's tokens alone:

    discrete token     p(k) ~ (alpha_k + n_dk) phi_kw
    continuous token   p(k, w) ~ (alpha_k + n_dk) phi_k,w x_w / (n_dfw + gamma_w)

theta is integrated out of the token conditionals and drawn from Dir(alpha + n_d.) at the
end of each kept sweep (or replaced by its conditional mean, `rao_blackwell`).  This is
what imputation, the explained-variance scores and the s_f calibration are built on:
leaving a variable's cells out of the record gives a prediction that has not seen them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .encoding import EncodedData
from .quantities import continuous_log_density, log_jacobian, x_of


@dataclass(frozen=True)
class RecordBatch:
    """The tokens of a set of records, in CSR form (record r owns ptr[r]:ptr[r+1])."""

    records: np.ndarray        # (R,) row numbers in the data
    disc_ptr: np.ndarray
    disc_word: np.ndarray
    cell_ptr: np.ndarray
    cell_base: np.ndarray      # metavocabulary offset of each continuous cell
    cell_x: np.ndarray
    cell_n: np.ndarray


def batch(data: EncodedData, records: np.ndarray, exclude_subs=()) -> RecordBatch:
    """Tokens of `records` from `data`, dropping every sub-variable in `exclude_subs`."""
    records = np.asarray(records, dtype=np.int64)
    excluded = np.asarray(sorted(exclude_subs), dtype=np.int64)
    position = np.full(data.n_records, -1, dtype=np.int64)
    position[records] = np.arange(records.size)

    keep = position[data.disc_doc] >= 0
    if excluded.size:
        keep &= ~np.isin(data.disc_sub, excluded)
    owner = position[data.disc_doc[keep]]
    order = np.argsort(owner, kind="stable")
    disc_word = data.disc_word[keep][order].astype(np.int64)
    disc_ptr = np.concatenate(([0], np.cumsum(np.bincount(owner, minlength=records.size))))

    keep = position[data.cell_doc] >= 0
    if excluded.size:
        keep &= ~np.isin(data.cell_sub, excluded)
    owner = position[data.cell_doc[keep]]
    order = np.argsort(owner, kind="stable")
    cells = np.flatnonzero(keep)[order]
    cell_ptr = np.concatenate(([0], np.cumsum(np.bincount(owner, minlength=records.size))))
    return RecordBatch(records, disc_ptr.astype(np.int64), disc_word,
                       cell_ptr.astype(np.int64),
                       data.vocab_offset[data.cell_sub[cells]].astype(np.int64),
                       data.cell_x[cells].astype(float), data.cell_n[cells].astype(np.int64))


@njit(cache=True, nogil=True, boundscheck=False)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True, nogil=True, boundscheck=False)
def _infer(disc_ptr, disc_word, cell_ptr, cell_base, cell_x, cell_n, phi, alpha,
           g0, g1, burn, n_keep, thin, rao_blackwell, out):
    """Run one fixed-phi chain per record; write kept theta rows into out[r, j, :]."""
    K = phi.shape[0]
    weights = np.empty(2 * K)
    counts = np.empty(K)
    alpha_sum = alpha.sum()
    for r in range(disc_ptr.shape[0] - 1):
        d0, d1 = disc_ptr[r], disc_ptr[r + 1]
        c0, c1 = cell_ptr[r], cell_ptr[r + 1]
        n_disc = d1 - d0
        n_cont = 0
        for c in range(c0, c1):
            n_cont += cell_n[c]
        z_disc = np.empty(n_disc, dtype=np.int64)
        z_cont = np.empty(n_cont, dtype=np.int64)
        w_cont = np.empty(n_cont, dtype=np.int64)
        n1 = np.zeros(c1 - c0, dtype=np.int64)
        for k in range(K):
            counts[k] = 0.0
        for i in range(n_disc):
            z_disc[i] = np.random.randint(0, K)
            counts[z_disc[i]] += 1
        at = 0
        for c in range(c0, c1):
            for j in range(cell_n[c]):
                z_cont[at] = np.random.randint(0, K)
                w_cont[at] = np.random.randint(0, 2)
                counts[z_cont[at]] += 1
                n1[c - c0] += w_cont[at]
                at += 1
        total_tokens = n_disc + n_cont

        kept = 0
        sweep = 0
        while kept < n_keep:
            for i in range(n_disc):
                counts[z_disc[i]] -= 1
                w = disc_word[d0 + i]
                total = 0.0
                for k in range(K):
                    total += (alpha[k] + counts[k]) * phi[k, w]
                    weights[k] = total
                target = np.random.random() * total
                pick = K - 1
                for k in range(K):
                    if weights[k] >= target:
                        pick = k
                        break
                z_disc[i] = pick
                counts[pick] += 1
            at = 0
            for c in range(c0, c1):
                base = cell_base[c]
                x1 = cell_x[c]
                n_c = cell_n[c]
                for j in range(n_c):
                    counts[z_cont[at]] -= 1
                    n1[c - c0] -= w_cont[at]
                    ones = n1[c - c0]
                    zeros = n_c - 1 - ones
                    r0 = (1.0 - x1) / (zeros + g0)
                    r1 = x1 / (ones + g1)
                    total = 0.0
                    for k in range(K):
                        share = alpha[k] + counts[k]
                        total += share * phi[k, base] * r0
                        weights[2 * k] = total
                        total += share * phi[k, base + 1] * r1
                        weights[2 * k + 1] = total
                    target = np.random.random() * total
                    pick = 2 * K - 1
                    for m in range(2 * K):
                        if weights[m] >= target:
                            pick = m
                            break
                    z_cont[at] = pick // 2
                    w_cont[at] = pick % 2
                    counts[pick // 2] += 1
                    n1[c - c0] += pick % 2
                    at += 1
            sweep += 1
            if sweep > burn and (sweep - burn) % thin == 0:
                if rao_blackwell:
                    for k in range(K):
                        out[r, kept, k] = (alpha[k] + counts[k]) / (alpha_sum + total_tokens)
                else:
                    s = 0.0
                    for k in range(K):
                        g = np.random.gamma(alpha[k] + counts[k], 1.0)
                        out[r, kept, k] = g
                        s += g
                    for k in range(K):
                        out[r, kept, k] /= s
                kept += 1


def infer_theta(tokens: RecordBatch, phi: np.ndarray, alpha: np.ndarray, gamma,
                seed: int, burn: int = 100, draws: int = 20, thin: int = 1,
                rao_blackwell: bool = False) -> np.ndarray:
    """(records, draws, K) theta_d given the batch's tokens, phi and alpha fixed.

    A record with no tokens left gets draws from its prior, Dir(alpha).
    """
    out = np.empty((tokens.records.size, draws, phi.shape[0]))
    _seed(int(seed) % (2 ** 31))
    _infer(tokens.disc_ptr, tokens.disc_word, tokens.cell_ptr, tokens.cell_base,
           tokens.cell_x, tokens.cell_n, np.ascontiguousarray(phi, dtype=float),
           np.asarray(alpha, dtype=float), float(gamma[0]), float(gamma[1]),
           int(burn), int(draws), int(thin), bool(rao_blackwell), out)
    return out


# --- the predictive distribution over one cell ----------------------------------------

GRID = np.linspace(5e-4, 1 - 5e-4, 400)


@dataclass
class Predictive:
    """What the model says about one cell, pooled over posterior draws."""

    kind: str
    pmf: np.ndarray | None = None          # discrete: probability per level
    density: np.ndarray | None = None      # continuous: density of y on GRID

    def mean(self) -> float:
        if self.kind == "continuous":
            w = self.density / self.density.sum()
            return float(GRID @ w)
        return float(np.arange(self.pmf.size) @ self.pmf)

    def mode(self) -> float:
        if self.kind == "continuous":
            return float(GRID[int(np.argmax(self.density))])
        return float(int(np.argmax(self.pmf)))

    def median(self) -> float:
        if self.kind == "continuous":
            cdf = np.cumsum(self.density)
            return float(np.interp(0.5, cdf / cdf[-1], GRID))
        return float(int(np.searchsorted(np.cumsum(self.pmf), 0.5)))

    def point(self, how: str) -> float:
        return {"mean": self.mean, "mode": self.mode, "median": self.median}[how]()

    def interval(self, mass: float = 0.95) -> tuple[float, float]:
        tail = (1 - mass) / 2
        if self.kind == "continuous":
            cdf = np.cumsum(self.density)
            cdf = cdf / cdf[-1]
            return float(np.interp(tail, cdf, GRID)), float(np.interp(1 - tail, cdf, GRID))
        cdf = np.cumsum(self.pmf)
        return (float(np.searchsorted(cdf, tail)),
                float(min(np.searchsorted(cdf, 1 - tail), self.pmf.size - 1)))


def predict_continuous(p1: np.ndarray, n_words: int, gamma) -> np.ndarray:
    """Density of y on GRID, averaged over the draws' p1 = theta . phi_f1."""
    # p(y) = sum_j Binom(j | N, p1) Beta(x(y) | j + gamma_1, N - j + gamma_0) dx/dy: the
    # Beta factors do not depend on p1, so they are one (grid, N+1) table, and averaging
    # over draws is a product with the mean binomial pmf.
    p1 = np.clip(np.atleast_1d(p1), 1e-300, 1 - 1e-16)
    return _beta_table(n_words, tuple(gamma)) @ binomial_pmf(p1, n_words).mean(axis=0)


def binomial_pmf(p: np.ndarray, n: int) -> np.ndarray:
    """(len(p), n+1) Binomial(j | n, p)."""
    from scipy.stats import binom

    return binom.pmf(np.arange(n + 1)[None, :], n, np.asarray(p)[:, None])


_TABLES: dict = {}


def _beta_table(n: int, gamma: tuple) -> np.ndarray:
    """(grid, n+1) Beta(x(y) | j + gamma_1, n - j + gamma_0) dx/dy on GRID."""
    key = (n, gamma)
    if key not in _TABLES:
        from scipy.stats import beta

        j = np.arange(n + 1)[None, :]
        x = x_of(GRID)[:, None]
        _TABLES[key] = (beta.pdf(x, j + gamma[1], n - j + gamma[0])
                        * np.exp(log_jacobian(GRID))[:, None])
    return _TABLES[key]


def log_density_y(p1: np.ndarray, y: float, n_words: int, gamma) -> float:
    """log of the pooled predictive density of y at one value, exactly."""
    from scipy.special import logsumexp

    p1 = np.atleast_1d(p1)
    x = x_of(y)
    values = continuous_log_density(p1, np.full(p1.size, np.log(x)),
                                    np.full(p1.size, np.log1p(-x)),
                                    np.full(p1.size, n_words, dtype=np.int64), gamma)
    return float(logsumexp(values) - np.log(p1.size) + log_jacobian(y))
