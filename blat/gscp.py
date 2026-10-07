r"""Sampling the Dirichlet parameters alpha via the Gamma-shape conjugate prior.  Paper App. F.

    GSCP(alpha | s, n)  \propto  exp(-s alpha) Gamma(alpha)^(-n)

For n = 0 this is Exp(s), the prior on alpha_k.  Given theta_d ~ Dir(alpha) and
auxiliary a_d ~ Gamma(sum_k alpha_k, 1), the products a_d theta_dk are independent
Gamma(alpha_k, 1) variates, and alpha_k | ... ~ GSCP(kappa - sum_d log(a_d theta_dk), D).
Draws are by Metropolised independence sampling from a matched Gamma proposal.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.special
from numba import njit


def inverse_digamma(x):
    """psi^-1(x), by three Newton steps from Minka's initialisation."""
    digamma, polygamma = scipy.special.digamma, scipy.special.polygamma
    x = np.asarray(x, dtype=float)
    m = x >= -2.22
    y = m * (np.exp(x) + 0.5) + (m - 1) / (x - digamma(1.0))
    for _ in range(3):
        y = y - (digamma(y) - x) / polygamma(1, y)
    return y


def _gamma_matching_mode(s, n):
    """Shape and rate of the Gamma matching the mode and curvature of log GSCP(s, n)."""
    m = inverse_digamma(-s / n)
    tau = n * scipy.special.polygamma(1, m)
    rate = 0.5 * tau * (m + np.sqrt(m * m + 4 / tau))
    return rate * rate / tau, rate


def _proposal(s, n):
    """A heavier-tailed proposal: shape n, rate chosen so its mean is the GSCP mode.

    Heavier tails than the matched Gamma keep the chain from sticking in a state the
    proposal would rarely produce, at the price of more rejections.
    """
    shape, rate = _gamma_matching_mode(s, n)
    return n, n * rate / (shape - 1)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False)
def _metropolis(s, n, x, prop_shape, prop_rate, proposals, uniforms):
    n_steps, k = proposals.shape
    for step in range(n_steps):
        for i in range(k):
            new = proposals[step, i]
            old = x[i]
            log_ratio = ((-s[i] * new - math.lgamma(new) * n)
                         - (-s[i] * old - math.lgamma(old) * n)
                         + ((prop_shape - 1.0) * math.log(old) - prop_rate[i] * old)
                         - ((prop_shape - 1.0) * math.log(new) - prop_rate[i] * new))
            if log_ratio > 5.0 or uniforms[step, i] < math.exp(min(log_ratio, 5.0)):
                x[i] = new
    return x


def sample_gscp(s, n: float, x, rng: np.random.Generator, n_steps: int = 5) -> np.ndarray:
    """Draws from GSCP(s_k, n) given current values x_k; vectorised over k."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    x = np.array(x, dtype=float, copy=True)
    if n == 0:
        return -np.log(rng.uniform(size=s.shape)) / s
    shape, rate = _proposal(s, n)
    shape = float(np.atleast_1d(shape)[0])
    rate = np.atleast_1d(rate).astype(float)
    proposals = rng.standard_gamma(shape, size=(n_steps, s.size)) / rate
    uniforms = rng.random((n_steps, s.size))
    return _metropolis(s, float(n), x, shape, rate, proposals, uniforms)


def sample_alpha(alpha: np.ndarray, counts_doc_topic: np.ndarray, kappa: float,
                 rng: np.random.Generator, n0: float = 0.0, iters: int = 5) -> np.ndarray:
    """A posterior draw of alpha given the type counts.  Paper Algorithm 1, last lines.

    Every iteration redraws theta_d ~ Dir(alpha + n_d.) and a_d from the *current* alpha;
    holding one theta fixed across iterations would not be a valid Gibbs sweep.
    """
    n_records, K = counts_doc_topic.shape
    alpha = np.array(alpha, dtype=float, copy=True)
    tiny = 1e-100
    shapes = np.empty((n_records, K))
    g = np.empty_like(shapes)
    for _ in range(iters):
        np.add(counts_doc_topic, alpha, out=shapes)
        rng.standard_gamma(shapes, out=g)
        theta = g / (g.sum(axis=1, keepdims=True) + tiny)
        a = rng.standard_gamma(alpha.sum(), size=n_records)
        s = kappa - np.log(theta + tiny).sum(axis=0) - np.log(a).sum()
        alpha = sample_gscp(s, n0 + n_records, alpha, rng)
    return alpha
