"""Sampler state, sweeps, initialisation and the chain schedule.

The schedule has three phases: `alpha-frozen` (alpha held at its initial value, so the
types can form before the concentration adapts), `burn-in` (alpha sampled too), and
`sampling` (every `thin`-th sweep is kept).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ModelConfig, RunConfig
from .encoding import IDX, EncodedData, make_counts
from .gscp import sample_alpha
from .kernel import Tables, sweep_continuous, sweep_discrete, tables

PHASES = ("alpha-frozen", "burn-in", "sampling")


@dataclass
class State:
    data: EncodedData
    cfg: ModelConfig
    alpha: np.ndarray
    z_disc: np.ndarray
    z_cont: np.ndarray
    w_cont: np.ndarray
    counts_doc_topic: np.ndarray
    counts_topic_word: np.ndarray
    counts_topic_feature: np.ndarray
    counts_doc: np.ndarray
    counts_doc_word: np.ndarray
    sweep: int = 0
    p_buf: np.ndarray = field(default=None, repr=False)
    tables: Tables = field(default=None, repr=False)    # at t = 1

    @property
    def K(self) -> int:
        return self.cfg.K


def new_state(data: EncodedData, cfg: ModelConfig) -> State:
    """Zero counts and unassigned tokens; the first sweep then initialises the chain."""
    return State(data=data, cfg=cfg, alpha=np.full(cfg.K, float(cfg.alpha_init)),
                 z_disc=np.full(data.n_discrete_tokens, -1, dtype=IDX),
                 z_cont=np.full(data.n_continuous_tokens, -1, dtype=IDX),
                 w_cont=np.zeros(data.n_continuous_tokens, dtype=IDX),
                 p_buf=np.zeros(2 * cfg.K), tables=tables(data, cfg.gamma),
                 **make_counts(data, cfg.K))


def gibbs_sweep(state: State, rng: np.random.Generator,
                t_tables: Tables | None = None) -> tuple[int, int]:
    """One sweep over every token, at temperature 1 unless `t_tables` are given (`ti`).
    Returns (discrete changed, continuous changed)."""
    d, tab = state.data, t_tables or state.tables
    n_disc = sweep_discrete(
        d.disc_doc, d.disc_sub, d.disc_word, state.z_disc,
        state.counts_doc_topic, state.counts_topic_word, state.counts_topic_feature,
        state.counts_doc, state.alpha, tab.word, tab.feature,
        state.p_buf, rng.random(d.n_discrete_tokens)) if d.n_discrete_tokens else 0
    n_cont = sweep_continuous(
        d.cont_doc, d.cont_sub, d.cont_cell, state.z_cont, state.w_cont,
        state.counts_doc_topic, state.counts_topic_word, state.counts_topic_feature,
        state.counts_doc, state.counts_doc_word, state.alpha, d.beta_per_sub,
        d.vocab_offset, d.first_continuous, tab.x0, tab.x1, tab.inv0, tab.inv1,
        state.p_buf, rng.random(d.n_continuous_tokens)) if d.n_continuous_tokens else 0
    state.sweep += 1
    return n_disc, n_cont


def update_alpha(state: State, rng: np.random.Generator, iters: int) -> None:
    state.alpha = sample_alpha(state.alpha, state.counts_doc_topic, state.cfg.kappa, rng,
                               n0=state.cfg.gscp_n0, iters=iters)


def recompute_counts(data: EncodedData, K: int, z_disc, z_cont, w_cont) -> dict:
    """Every count array rebuilt from the assignments alone."""
    fresh = make_counts(data, K)
    if data.n_discrete_tokens:
        np.add.at(fresh["counts_doc_topic"], (data.disc_doc, z_disc), 1)
        np.add.at(fresh["counts_topic_word"], (z_disc, data.disc_word), 1)
        np.add.at(fresh["counts_topic_feature"], (z_disc, data.disc_sub), 1)
        np.add.at(fresh["counts_doc"], data.disc_doc, 1)
    if data.n_continuous_tokens:
        base = data.vocab_offset[data.cont_sub] + w_cont
        col = 2 * (data.cont_sub - data.first_continuous) + w_cont
        np.add.at(fresh["counts_doc_topic"], (data.cont_doc, z_cont), 1)
        np.add.at(fresh["counts_topic_word"], (z_cont, base), 1)
        np.add.at(fresh["counts_topic_feature"], (z_cont, data.cont_sub), 1)
        np.add.at(fresh["counts_doc"], data.cont_doc, 1)
        np.add.at(fresh["counts_doc_word"], (data.cont_doc, col), 1)
    return fresh


def check_counts(state: State) -> None:
    """Assert that the maintained counts equal those recomputed from (z, w)."""
    fresh = recompute_counts(state.data, state.K, state.z_disc, state.z_cont, state.w_cont)
    for name, expected in fresh.items():
        if not np.array_equal(getattr(state, name), expected):
            raise AssertionError(f"{name} drifted from the assignments at sweep "
                                 f"{state.sweep}")


def state_from_parameters(data: EncodedData, cfg: ModelConfig, alpha: np.ndarray,
                          theta: np.ndarray, phi: np.ndarray,
                          rng: np.random.Generator) -> State:
    """A state whose (z, w) is drawn exactly from p(z, w | theta, phi, x).

    Given theta and phi the tokens are conditionally independent across records and
    variables.  A discrete token's type has p(k) ~ theta_dk phi_kw.  For a continuous
    cell with N words, the number of 1-words n1 has

        p(n1) ~ Binom(n1 | N, a) Beta(x | n1 + gamma_1, N - n1 + gamma_0),  a = theta_d . phi_f1

    which is drawn first; n1 of the cell's words (chosen at random) are then 1, and each
    word's type follows p(k | w) ~ theta_dk phi_k,w.  If (alpha, theta, phi) is a posterior
    draw, the resulting state is an exact draw from the posterior -- which is what lets a
    backward thermodynamic integration start from a stored draw rather than a stored state.
    """
    from scipy.special import gammaln

    state = new_state(data, cfg)
    state.alpha = np.asarray(alpha, dtype=float).copy()
    K = cfg.K

    def pick(weights: np.ndarray) -> np.ndarray:
        cum = np.cumsum(weights, axis=1)
        u = rng.random(len(weights)) * cum[:, -1]
        return (cum < u[:, None]).sum(axis=1).clip(0, weights.shape[1] - 1)

    if data.n_discrete_tokens:
        w = theta[data.disc_doc] * phi[:, data.disc_word].T
        state.z_disc[:] = pick(w)

    if data.n_cells:
        g0, g1 = cfg.gamma
        base = data.vocab_offset[data.cell_sub]
        a = np.einsum("ck,kc->c", theta[data.cell_doc], phi[:, base + 1])
        a = np.clip(a, 1e-300, 1 - 1e-16)
        n1 = np.empty(data.n_cells, dtype=np.int64)
        for n in np.unique(data.cell_n):
            rows = np.flatnonzero(data.cell_n == n)
            j = np.arange(n + 1)
            log_w = (gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)
                     + j * np.log(a[rows, None]) + (n - j) * np.log1p(-a[rows, None])
                     + (j + g1 - 1) * data.cell_log_x[rows, None]
                     + (n - j + g0 - 1) * data.cell_log_1mx[rows, None]
                     - (gammaln(j + g1) + gammaln(n - j + g0) - gammaln(n + g1 + g0)))
            weights = np.exp(log_w - log_w.max(axis=1, keepdims=True))
            n1[rows] = pick(weights)
        # rank of each token within its cell, in random order; the lowest n1 become 1
        key = rng.random(data.n_continuous_tokens)
        order = np.lexsort((key, data.cont_cell))
        starts = np.concatenate(([0], np.cumsum(data.cell_n)[:-1]))
        rank = np.empty(data.n_continuous_tokens, dtype=np.int64)
        rank[order] = np.arange(data.n_continuous_tokens) - np.repeat(starts, data.cell_n)
        state.w_cont[:] = (rank < n1[data.cont_cell]).astype(IDX)
        tw = data.vocab_offset[data.cont_sub] + state.w_cont
        state.z_cont[:] = pick(theta[data.cont_doc] * phi[:, tw].T)

    fresh = recompute_counts(data, K, state.z_disc, state.z_cont, state.w_cont)
    for name, value in fresh.items():
        getattr(state, name)[...] = value
    return state


def phase_of(i: int, run: RunConfig) -> str:
    if i < run.alpha_frozen:
        return "alpha-frozen"
    return "burn-in" if i < run.burnin else "sampling"


def run_chain(data: EncodedData, cfg: ModelConfig, run: RunConfig,
              rng: np.random.Generator, on_sweep=None, state: State | None = None,
              check_invariants: bool = False) -> State:
    """Run the schedule.  `on_sweep(state, phase, changed)` is called after every sweep."""
    state = state if state is not None else new_state(data, cfg)
    for i in range(run.total):
        phase = phase_of(i, run)
        changed = gibbs_sweep(state, rng)
        if i >= run.alpha_frozen:
            update_alpha(state, rng, run.gscp_iters_sampling if phase == "sampling"
                         else run.gscp_iters_burnin)
        if check_invariants:
            check_counts(state)
        if on_sweep is not None:
            on_sweep(state, phase, changed)
    return state
