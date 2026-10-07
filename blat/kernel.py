"""The compiled inner loops of the collapsed Gibbs sampler.  Paper Sec. 3.4, Algorithm 1.

The per-token draw has a loop-carried dependency through the counts, so it is compiled.
The kernels hold no state and draw no random numbers: uniforms are passed in, so a chain
is reproducible from its seed.  A type of -1 means "not yet assigned", which is how the
first sweep initialises the chain from the prior predictive.

The same loops serve thermodynamic integration.  Tempering the likelihood by t turns the
ratios of counts in the conditionals into ratios of Gamma functions, so the kernels read
those factors from tables built per temperature (`Tables`); at t = 1 the tables hold the
plain ratios and the kernels are the ordinary sampler.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit
from scipy.special import gammaln

_JIT = dict(cache=True, fastmath=False, nogil=True, boundscheck=False)


class Tables(NamedTuple):
    """The factors of the conditionals at temperature t.

    word[f, m]       = Gamma(b_f + t (m+1)) / Gamma(b_f + t m)        t = 1: b_f + m
    feature[f, m]    = Gamma(B_f + t m) / Gamma(B_f + t (m+1))        t = 1: 1 / (B_f + m)
    x0[c], x1[c]     = (1 - x_c)^t, x_c^t
    inv0[n], inv1[n] = (n + gamma_0)^-t, (n + gamma_1)^-t

    with B_f = V_f b_f.  `word` and `feature` serve discrete sub-variables, which have at
    most one token per record, so no count exceeds the number of records.  Only the observation terms are tempered: the latent words of continuous
    cells belong to the prior structure, so their type factors are not tabulated.
    """

    word: np.ndarray
    feature: np.ndarray
    x0: np.ndarray
    x1: np.ndarray
    inv0: np.ndarray
    inv1: np.ndarray


def tables(data, gamma, t: float = 1.0) -> Tables:
    m = np.arange(data.n_records + 2, dtype=float)[None, :]
    b = data.beta_per_sub[:, None]
    bv = (data.beta_per_sub * data.vocab_size)[:, None]
    n = np.arange(data.max_words + 2, dtype=float)
    g0, g1 = gamma
    if t == 1.0:
        word, feature = b + m, 1.0 / (bv + m)
        x0, x1 = 1.0 - data.cell_x, data.cell_x
    else:
        word = np.exp(gammaln(b + t * (m + 1)) - gammaln(b + t * m))
        feature = np.exp(gammaln(bv + t * m) - gammaln(bv + t * (m + 1)))
        x0, x1 = np.exp(t * data.cell_log_1mx), np.exp(t * data.cell_log_x)
    return Tables(*(np.ascontiguousarray(a, dtype=float)
                    for a in (word, feature, x0, x1, (n + g0) ** -t, (n + g1) ** -t)))


@njit(**_JIT)
def sweep_discrete(doc, sub, word, z, ndk, nkw, nkf, nd, alpha, word_ratio, feature_ratio,
                   p_buf, u):
    """Resample the type of every discrete token:

        p(k) ~ (alpha_k + n_dk) word[f, n_kfw] feature[f, n_kf]

    with the token itself removed from the counts.  Returns the number of changed types.
    """
    K = ndk.shape[1]
    n_changed = 0
    for i in range(doc.shape[0]):
        d, f, w, k_old = doc[i], sub[i], word[i], z[i]
        if k_old >= 0:
            ndk[d, k_old] -= 1
            nkw[k_old, w] -= 1
            nkf[k_old, f] -= 1
            nd[d] -= 1
        total = 0.0
        for k in range(K):
            total += ((alpha[k] + ndk[d, k]) * word_ratio[f, nkw[k, w]]
                      * feature_ratio[f, nkf[k, f]])
            p_buf[k] = total
        target = u[i] * total
        k_new = K - 1
        for k in range(K):
            if p_buf[k] >= target:
                k_new = k
                break
        ndk[d, k_new] += 1
        nkw[k_new, w] += 1
        nkf[k_new, f] += 1
        nd[d] += 1
        z[i] = k_new
        if k_new != k_old:
            n_changed += 1
    return n_changed


@njit(**_JIT)
def sweep_continuous(doc, sub, cell, z, ws, ndk, nkw, nkf, nd, ndw, alpha, beta, offset,
                     first_cont, x0, x1, inv0, inv1, p_buf, u):
    """Resample type and latent binary word jointly for every continuous token:

        p(k, w) ~ (alpha_k + n_dk) (b_f + n_kfw) / (2 b_f + n_kf) x_w^t (n_dfw + gamma_w)^-t

    with x_1 = x, x_0 = 1 - x (paper eq. joint_sample).  Returns the number changed.
    """
    K = ndk.shape[1]
    n_changed = 0
    for i in range(doc.shape[0]):
        d, f, c = doc[i], sub[i], cell[i]
        base = offset[f]
        col = 2 * (f - first_cont)
        k_old, w_old = z[i], ws[i]
        if k_old >= 0:
            ndk[d, k_old] -= 1
            nkw[k_old, base + w_old] -= 1
            nkf[k_old, f] -= 1
            nd[d] -= 1
            ndw[d, col + w_old] -= 1
        b = beta[f]
        r0 = x0[c] * inv0[ndw[d, col]]
        r1 = x1[c] * inv1[ndw[d, col + 1]]
        total = 0.0
        for k in range(K):
            shared = (alpha[k] + ndk[d, k]) / (2.0 * b + nkf[k, f])
            total += shared * (b + nkw[k, base]) * r0
            p_buf[2 * k] = total
            total += shared * (b + nkw[k, base + 1]) * r1
            p_buf[2 * k + 1] = total
        target = u[i] * total
        choice = 2 * K - 1
        for j in range(2 * K):
            if p_buf[j] >= target:
                choice = j
                break
        k_new, w_new = choice // 2, choice % 2
        ndk[d, k_new] += 1
        nkw[k_new, base + w_new] += 1
        nkf[k_new, f] += 1
        nd[d] += 1
        ndw[d, col + w_new] += 1
        z[i] = k_new
        ws[i] = w_new
        if k_new != k_old or w_new != w_old:
            n_changed += 1
    return n_changed
