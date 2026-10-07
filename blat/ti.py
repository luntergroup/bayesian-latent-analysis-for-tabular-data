"""`ti`: log p(x | K) by thermodynamic integration.  Paper App. G.

Define the tempered target p_t(z, phi) ~ p(z | alpha) p(phi | beta) L(z, phi)^t with
normaliser Z(t).  Z(0) = 1 and Z(1) = p(x | K), and d/dt log Z(t) = E_t[log L], so

    log p(x | K) = integral_0^1 E_t[log L] dt,

estimated on a ladder t_i = (i/n)^power by the trapezoid rule; the difference from
Simpson's rule on the same points estimates the quadrature error, and the per-temperature
standard errors (autocorrelation-corrected) give the Monte Carlo error.

phi stays collapsed: tempering multiplies the Dirichlet counts by t, so the conditional
is a ratio of Gamma functions, tabulated per temperature (`kernel.tables`) and swept by
the sampler's own kernels.  Only the observation terms are tempered; the latent binary
words of continuous cells belong to the prior structure.
alpha is sampled along the path with the same update as in `run` (it does not see t), so
the quantity is the evidence of the hierarchical model, log p(x | K, beta, gamma, kappa).

Directions:
  forward   from t = 0 (the prior; the first iteration initialises from the prior predictive)
            up to t = 1;
  backward  from t = 1 down to 0, starting from a posterior draw of a finished run
            (`sampler.state_from_parameters`), after a short burn-in at t = 1.
The two should agree; a gap between them says the ladder is too coarse or too short.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import digamma

from .config import ModelConfig
from .diagnostics import effective_sample_size
from .encoding import EncodedData
from .gscp import sample_alpha
from .kernel import tables
from .quantities import log_p_x, log_p_z, log_p_w
from .sampler import gibbs_sweep, new_state


@dataclass(frozen=True)
class TiConfig:
    temperatures: int = 50
    power: float = 5.0
    burn_per_temperature: int = 150
    iters_per_temperature: int = 250
    initial_burn: int = 200       # extra iterations at the starting temperature
    direction: str = "forward"
    seed: int = 0
    sample_alpha: bool = True
    alpha_iters: int = 2

    def __post_init__(self):
        if self.temperatures < 3 or not self.power > 0:
            raise ValueError("need >= 3 temperatures and a positive ladder power")
        if self.iters_per_temperature < 4 or min(self.burn_per_temperature,
                                                   self.initial_burn) < 0:
            raise ValueError("need >= 4 iterations per temperature and non-negative burn-in")
        if self.direction not in ("forward", "backward"):
            raise ValueError("direction must be forward or backward")


def ladder(n: int, power: float) -> np.ndarray:
    return (np.arange(n, dtype=float) / (n - 1)) ** power


def integrand(state, data: EncodedData, gamma, t: float) -> float:
    """E[log L | z] at temperature t, Rao-Blackwellised over phi.

        discrete:    sum n_kfw [psi(b_f + t n_kfw) - psi(V_f b_f + t n_kf)]
        continuous:  sum_cells log Beta(x | n + gamma)        (log_p_continuous)
    """
    total = 0.0
    discrete = ~data.layout.is_continuous
    if discrete.any():
        cols = np.repeat(discrete, data.vocab_size)
        sub_of_word = np.repeat(np.arange(data.n_sub), data.vocab_size)[cols]
        b = np.repeat(data.beta_per_sub, data.vocab_size)[cols]
        bv = (data.beta_per_sub * data.vocab_size)[sub_of_word]
        nkw = state.counts_topic_word[:, cols]
        nkf = state.counts_topic_feature[:, sub_of_word]
        total += float(np.sum(nkw * (digamma(b[None, :] + t * nkw)
                                     - digamma(bv[None, :] + t * nkf))))
    return total + log_p_x(state.counts_doc_word, data, gamma)


@dataclass
class TiResult:
    log_evidence: float
    mc_error: float
    quadrature_error: float
    ladder: np.ndarray
    means: np.ndarray
    errors: np.ndarray
    ess: np.ndarray
    half_difference: np.ndarray
    alpha_sum: np.ndarray
    samples: np.ndarray
    start_log_joint: np.ndarray
    seconds: float

    @property
    def total_error(self) -> float:
        return float(math.hypot(self.mc_error, self.quadrature_error))


def simpson_uneven(y: np.ndarray, x: np.ndarray) -> float:
    total = 0.0
    for i in range(0, len(x) - 2, 2):
        h1, h2 = x[i + 1] - x[i], x[i + 2] - x[i + 1]
        if h1 <= 0 or h2 <= 0:
            continue
        h = h1 + h2
        total += (h / 6.0) * ((2 - h2 / h1) * y[i] + (h * h / (h1 * h2)) * y[i + 1]
                              + (2 - h1 / h2) * y[i + 2])
    if (len(x) - 1) % 2:
        total += 0.5 * (x[-1] - x[-2]) * (y[-1] + y[-2])
    return float(total)


def trapezoid_weights(t: np.ndarray) -> np.ndarray:
    w = np.zeros_like(t)
    h = np.diff(t)
    w[:-1] += h / 2
    w[1:] += h / 2
    return w


def run_ti(data: EncodedData, cfg: ModelConfig, ti: TiConfig, rng: np.random.Generator,
           start=None, progress: bool = True) -> TiResult:
    """Traverse the ladder; `start` is the initial State for a backward path."""
    t_grid = ladder(ti.temperatures, ti.power)
    state = start if start is not None else new_state(data, cfg)

    n = ti.temperatures
    means, errors, ess = np.empty(n), np.empty(n), np.empty(n)
    half, alpha_sum = np.empty(n), np.empty(n)
    samples = np.empty((n, ti.iters_per_temperature))
    start_log_joint = []
    order = list(range(n)) if ti.direction == "forward" else list(range(n))[::-1]
    t0 = time.time()
    for step, i in enumerate(order):
        t = t_grid[i]
        t_tables = tables(data, cfg.gamma, t)
        burn = ti.burn_per_temperature + (ti.initial_burn if step == 0 else 0)
        values, sums = np.empty(ti.iters_per_temperature), np.empty(ti.iters_per_temperature)
        for sweep in range(burn + ti.iters_per_temperature):
            gibbs_sweep(state, rng, t_tables)
            if ti.sample_alpha:
                state.alpha = sample_alpha(state.alpha, state.counts_doc_topic, cfg.kappa,
                                           rng, n0=cfg.gscp_n0, iters=ti.alpha_iters)
            if step == 0 and ti.direction == "backward":
                start_log_joint.append(log_p_z(state.counts_doc_topic, state.alpha)
                                 + log_p_w(state.counts_topic_word,
                                           state.counts_topic_feature, data)
                                 + log_p_x(state.counts_doc_word, data, cfg.gamma))
            if sweep >= burn:
                values[sweep - burn] = integrand(state, data, cfg.gamma, t)
                sums[sweep - burn] = state.alpha.sum()
        samples[i] = values
        means[i] = values.mean()
        ess[i] = max(effective_sample_size(values), 1.0)
        errors[i] = values.std(ddof=1) / math.sqrt(ess[i])
        h = values.size // 2
        half[i] = values[h:].mean() - values[:h].mean()
        alpha_sum[i] = sums.mean()
        if progress and (step % max(1, n // 10) == 0 or step == n - 1):
            print(f"  ti {ti.direction} {step + 1}/{n}  t={t:.3g}  E[log L]={means[i]:.1f}"
                  f" +/- {errors[i]:.2f}  ({time.time() - t0:.0f}s)", file=sys.stderr,
                  flush=True)

    trap = float(trapezoid(means, t_grid))
    return TiResult(log_evidence=trap,
                    mc_error=float(np.sqrt(np.sum((trapezoid_weights(t_grid) * errors) ** 2))),
                    quadrature_error=abs(trap - simpson_uneven(means, t_grid)),
                    ladder=t_grid, means=means, errors=errors, ess=ess,
                    half_difference=half, alpha_sum=alpha_sum, samples=samples,
                    start_log_joint=np.array(start_log_joint), seconds=time.time() - t0)


def result_record(result: TiResult, ti: TiConfig) -> tuple:
    """(summary entries, per-temperature table)."""
    import pandas as pd

    out = {"log_evidence": result.log_evidence, "mc_error": result.mc_error,
           "quadrature_error": result.quadrature_error, "total_error": result.total_error,
           "direction": ti.direction,
           "target": "hierarchical (alpha integrated)" if ti.sample_alpha
           else "fixed alpha", "ti": asdict(ti), "seconds": result.seconds}
    if result.start_log_joint.size:
        q = result.start_log_joint
        k = max(1, q.size // 4)
        out["start_burn_in"] = {"log_joint_first_quarter": float(q[:k].mean()),
                                "log_joint_last_quarter": float(q[-k:].mean()),
                                "log_joint_sd": float(q.std(ddof=1)) if q.size > 1
                                else math.nan}
    table = pd.DataFrame({"t": result.ladder, "mean_log_l": result.means,
                          "se": result.errors, "ess": result.ess,
                          "half_difference": result.half_difference,
                          "alpha_sum": result.alpha_sum})
    return out, table
