"""Reported quantities: the collapsed log joint, posterior draws, and cell densities.

Per sweep, from the counts alone (cheap):
    log_p_types      = log p(z | alpha)      Dirichlet-multinomial per record
    log_p_words      = log p(w | z, beta)    Dirichlet-multinomial per (type, sub-variable),
                                             phi integrated out
    log_p_continuous = log p(x | n, gamma)   Beta density per continuous cell
    log_joint        = the sum of the three  the collapsed log joint log p(x, z | alpha,
                                             beta, gamma); Fig 1 plots it.  A convergence
                                             diagnostic, not a model score: not
                                             comparable across K
plus type occupancy, K_eff and the mean entropy of theta_d.

Per stored draw:
    log_lik = log p(x | theta, phi)          the data log likelihood at a draw, with the
                                           latent word counts of continuous cells
                                           marginalised exactly
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.special import gammaln

from .encoding import EncodedData, normalise_blocks


# --- the collapsed log joint --------------------------------------------------------

def _log_rising(base: np.ndarray, n_max: int) -> np.ndarray:
    """table[i, n] = log Gamma(base_i + n) - log Gamma(base_i), for n = 0..n_max."""
    base = np.atleast_1d(np.asarray(base, dtype=float))
    steps = np.log(base[:, None] + np.arange(n_max, dtype=float)[None, :])
    out = np.zeros((base.size, n_max + 1))
    np.cumsum(steps, axis=1, out=out[:, 1:])
    return out


def log_p_z(counts_doc_topic: np.ndarray, alpha: np.ndarray) -> float:
    n_d = counts_doc_topic.sum(axis=1)
    n_max = int(max(n_d.max(initial=0), 1))
    per_type = _log_rising(alpha, n_max)
    total = _log_rising(alpha.sum(), n_max)[0]
    contrib = sum(per_type[k][counts_doc_topic[:, k]].sum() for k in range(alpha.size))
    return float(contrib - total[n_d].sum())


def log_p_w(counts_topic_word, counts_topic_feature, data: EncodedData) -> float:
    K = counts_topic_word.shape[0]
    b_total = data.beta_per_sub * data.vocab_size
    per_word = np.repeat(data.beta_per_sub, data.vocab_size)
    return float(K * np.sum(gammaln(b_total))
                 - np.sum(gammaln(b_total[None, :] + counts_topic_feature))
                 + np.sum(gammaln(per_word[None, :] + counts_topic_word))
                 - K * np.sum(gammaln(per_word)))


@njit(cache=True, nogil=True, boundscheck=False)
def _a3_core(counts_doc_word, cell_doc, col1, cell_n, log_x, log_1mx, log_beta, g0, g1):
    total = 0.0
    for c in range(cell_doc.shape[0]):
        n = cell_n[c]
        n1 = counts_doc_word[cell_doc[c], col1[c]]
        total += ((n1 + g1 - 1.0) * log_x[c] + ((n - n1) + g0 - 1.0) * log_1mx[c]
                  - log_beta[n, n1])
    return total


def _log_beta_table(max_n: int, gamma) -> np.ndarray:
    """table[n, n1] = log B(n1 + gamma_1, n - n1 + gamma_0)."""
    g0, g1 = gamma
    n = np.arange(max_n + 1)[:, None]
    n1 = np.arange(max_n + 1)[None, :]
    with np.errstate(invalid="ignore"):
        out = gammaln(n1 + g1) + gammaln(np.maximum(n - n1, 0) + g0) - gammaln(n + g0 + g1)
    return np.where(n1 <= n, out, np.inf)


def log_p_x(counts_doc_word: np.ndarray, data: EncodedData, gamma) -> float:
    if data.n_cells == 0:
        return 0.0
    col1 = 2 * (data.cell_sub - data.first_continuous) + 1
    return float(_a3_core(counts_doc_word, data.cell_doc, col1, data.cell_n,
                          data.cell_log_x, data.cell_log_1mx,
                          _log_beta_table(data.max_words, gamma),
                          float(gamma[0]), float(gamma[1])))


@njit(cache=True, nogil=True, boundscheck=False)
def _occupancy_stats(counts, alpha):
    n_records, K = counts.shape
    occupancy = np.zeros(K)
    alpha_total = alpha.sum()
    entropy_sum = 0.0
    for d in range(n_records):
        row = alpha_total
        for k in range(K):
            occupancy[k] += counts[d, k]
            row += counts[d, k]
        h = 0.0
        for k in range(K):
            w = (alpha[k] + counts[d, k]) / row
            if w > 0.0:
                h -= w * np.log(w)
        entropy_sum += h
    grand = occupancy.sum()
    h_occ = 0.0
    if grand > 0.0:
        for k in range(K):
            p = occupancy[k] / grand
            if p > 0.0:
                h_occ -= p * np.log(p)
    return occupancy, np.exp(h_occ), entropy_sum / max(n_records, 1)


def sweep_row(state, changed: tuple[int, int]) -> dict:
    """The per-sweep diagnostics, from the counts alone."""
    d = state.data
    occupancy, k_eff, entropy = _occupancy_stats(state.counts_doc_topic, state.alpha)
    types = log_p_z(state.counts_doc_topic, state.alpha)
    words = log_p_w(state.counts_topic_word, state.counts_topic_feature, d)
    continuous = log_p_x(state.counts_doc_word, d, state.cfg.gamma)
    return dict(
        log_p_types=types, log_p_words=words, log_p_continuous=continuous,
        log_joint=types + words + continuous,
        alpha=state.alpha.copy(), alpha_sum=float(state.alpha.sum()),
        change_discrete=changed[0] / max(d.n_discrete_tokens, 1),
        change_continuous=changed[1] / max(d.n_continuous_tokens, 1),
        change_rate=(changed[0] + changed[1]) / max(d.n_tokens, 1),
        occupancy=np.sort(occupancy)[::-1] / max(d.n_tokens, 1),
        k_eff=float(k_eff), theta_entropy=float(entropy))


# --- posterior draws ---------------------------------------------------------------

def draw_theta(counts_doc_topic, alpha, rng) -> np.ndarray:
    g = rng.standard_gamma(alpha[None, :] + counts_doc_topic)
    return g / g.sum(axis=1, keepdims=True)


def draw_phi(counts_topic_word, data: EncodedData, rng) -> np.ndarray:
    per_word = np.repeat(data.beta_per_sub, data.vocab_size)
    return normalise_blocks(rng.standard_gamma(per_word[None, :] + counts_topic_word), data)


def mean_phi(counts_topic_word, data: EncodedData) -> np.ndarray:
    per_word = np.repeat(data.beta_per_sub, data.vocab_size)
    return normalise_blocks(per_word[None, :] + counts_topic_word, data)


# --- densities -----------------------------------------------------------------------

@njit(cache=True, nogil=True, boundscheck=False)
def _marginal_core(p1, log_x, log_1mx, cell_n, log_choose, log_beta, g0, g1):
    """log sum_{n1} Binom(n1 | N, p1) Beta(x | n1 + gamma_1, N - n1 + gamma_0), per cell."""
    out = np.empty(p1.shape[0])
    max_n = log_choose.shape[0] - 1
    terms = np.empty(max_n + 1)
    for c in range(p1.shape[0]):
        n = cell_n[c]
        lp = np.log(p1[c])
        l1p = np.log1p(-p1[c])
        best = -1e308
        for j in range(n + 1):
            v = (log_choose[n, j] + j * lp + (n - j) * l1p
                 + (j + g1 - 1.0) * log_x[c] + (n - j + g0 - 1.0) * log_1mx[c]
                 - log_beta[n, j])
            terms[j] = v
            if v > best:
                best = v
        acc = 0.0
        for j in range(n + 1):
            acc += np.exp(terms[j] - best)
        out[c] = best + np.log(acc)
    return out


def continuous_log_density(p1, log_x, log_1mx, cell_n, gamma) -> np.ndarray:
    """log p(x | a) for continuous cells, the latent counts marginalised exactly.

    A density of the Beta variate x; add `log_jacobian(y)` for a density of y.
    """
    cell_n = np.asarray(cell_n, dtype=np.int64)
    max_n = int(cell_n.max()) if cell_n.size else 1
    n = np.arange(max_n + 1)[:, None]
    j = np.arange(max_n + 1)[None, :]
    with np.errstate(invalid="ignore"):
        log_choose = np.where(j <= n, gammaln(n + 1) - gammaln(j + 1)
                              - gammaln(np.maximum(n - j, 0) + 1), -np.inf)
    p1 = np.clip(np.asarray(p1, dtype=float), 1e-300, 1 - 1e-16)
    return _marginal_core(np.ascontiguousarray(p1), np.ascontiguousarray(log_x),
                          np.ascontiguousarray(log_1mx), np.ascontiguousarray(cell_n),
                          log_choose, _log_beta_table(max_n, gamma),
                          float(gamma[0]), float(gamma[1]))


def log_jacobian(y) -> np.ndarray:
    """log dx/dy for x = (1 - cos(pi y)) / 2."""
    return np.log(0.5 * np.pi * np.sin(np.pi * np.asarray(y, dtype=float)))


def x_of(y):
    return 0.5 - 0.5 * np.cos(np.pi * np.asarray(y, dtype=float))


def y_of(x):
    return np.arccos(np.clip(1.0 - 2.0 * np.asarray(x, dtype=float), -1.0, 1.0)) / np.pi


@njit(cache=True, nogil=True, boundscheck=False)
def _gathered_log(tp, doc, word):
    total = 0.0
    for i in range(doc.shape[0]):
        total += np.log(tp[doc[i], word[i]])
    return total


def data_loglik(data: EncodedData, gamma, theta: np.ndarray, phi: np.ndarray) -> float:
    """log_lik = log p(x | theta, phi) over every observed cell (continuous on the x scale)."""
    tp = theta @ phi
    total = _gathered_log(tp, data.disc_doc, data.disc_word) if data.n_discrete_tokens \
        else 0.0
    if data.n_cells:
        base = data.vocab_offset[data.cell_sub]
        total += float(continuous_log_density(tp[data.cell_doc, base + 1], data.cell_log_x,
                                              data.cell_log_1mx, data.cell_n, gamma).sum())
    return float(total)


# --- level probabilities -------------------------------------------------------------

def column_slice(layout, column: str) -> slice:
    """The metavocabulary columns of a variable: those of its (adjacent) sub-variables."""
    subs = layout.subs[column]
    return slice(int(layout.vocab_offset[subs[0]]), int(layout.vocab_offset[subs[-1] + 1]))


def level_probabilities(tp: np.ndarray, layout, column: str) -> np.ndarray:
    """(..., levels) level probabilities of a discrete variable from theta @ phi rows."""
    return block_levels(tp[..., column_slice(layout, column)], layout.kinds[column])


def block_levels(block: np.ndarray, kind: str) -> np.ndarray:
    """Level probabilities from a variable's own columns of theta @ phi (`column_slice`).

    For an ordinal, whose sub-feature m holds (continue, stop) in columns 2m and 2m+1,
    P(level c) = prod_{m<c} P(continue at m) * P(stop at c), and the top level takes what
    is left, so the levels sum to one exactly.
    """
    if kind == "categorical":
        return block / block.sum(axis=-1, keepdims=True)
    stop = block[..., 1::2] / (block[..., 0::2] + block[..., 1::2])
    survive = np.cumprod(1.0 - stop, axis=-1)                   # past sub-feature m
    before = np.concatenate([np.ones_like(survive[..., :1]), survive[..., :-1]], axis=-1)
    return np.concatenate([before * stop, survive[..., -1:]], axis=-1)


def expected_y(p1, n_words: int, gamma) -> np.ndarray:
    """E[x] = (N p1 + gamma_1) / (N + gamma) mapped to the y scale.

    The value a type (p1 = phi_f1) or a record (p1 = theta_d . phi_f1) expects, on the
    preprocessed scale.
    """
    g0, g1 = gamma
    return y_of((n_words * np.asarray(p1) + g1) / (n_words + g0 + g1))


@dataclass
class DrawSummary:
    """Running mean and a thinned sample of a sequence of arrays, for credible intervals."""

    keep_every: int
    total: np.ndarray = None
    n: int = 0
    kept: list = None

    def add(self, value: np.ndarray, index: int) -> None:
        if self.total is None:
            self.total = np.zeros_like(value, dtype=float)
            self.kept = []
        self.total += value
        self.n += 1
        if index % self.keep_every == 0:
            self.kept.append(value.astype(np.float32))

    def mean(self) -> np.ndarray:
        return self.total / max(self.n, 1)

    def interval(self, mass: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        stack = np.stack(self.kept)
        tail = 100 * (1 - mass) / 2
        return (np.percentile(stack, tail, axis=0), np.percentile(stack, 100 - tail, axis=0))
