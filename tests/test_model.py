"""The sampler and thermodynamic integration against exact enumeration on a tiny table.

With three records, K = 2 and one latent word per continuous cell, every assignment of
(z, w) can be listed.  The collapsed joint exp(log_p_types + log_p_words + log_p_continuous) then gives the exact
posterior over assignments and the exact evidence p(x | K, alpha, beta, gamma).
"""

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from blat.encoding import encode
from blat.quantities import log_p_w, log_p_x, log_p_z
from blat.sampler import (check_counts, gibbs_sweep, new_state, recompute_counts,
                          state_from_parameters)
from blat.ti import TiConfig, run_ti


def enumerate_posterior(data, cfg):
    """(log weights, z_disc, z_cont, w_cont) for every assignment."""
    K = cfg.K
    alpha = np.full(K, cfg.alpha_init)
    nd, nc = data.n_discrete_tokens, data.n_continuous_tokens
    rows = []
    for zd in itertools.product(range(K), repeat=nd):
        for zc in itertools.product(range(2 * K), repeat=nc):
            z_disc = np.array(zd, dtype=np.int32)
            z_cont = np.array([c // 2 for c in zc], dtype=np.int32)
            w_cont = np.array([c % 2 for c in zc], dtype=np.int32)
            counts = recompute_counts(data, K, z_disc, z_cont, w_cont)
            lw = (log_p_z(counts["counts_doc_topic"], alpha)
                  + log_p_w(counts["counts_topic_word"], counts["counts_topic_feature"], data)
                  + log_p_x(counts["counts_doc_word"], data, cfg.gamma))
            rows.append((lw, z_disc, z_cont, w_cont))
    return rows


def label_free_statistics(z_disc, z_cont, w_cont):
    """Co-assignment indicators of every token pair, and the latent words."""
    z = np.concatenate([z_disc, z_cont])
    i, j = np.triu_indices(z.size, 1)
    return np.concatenate([(z[i] == z[j]).astype(float), w_cont.astype(float)])


def test_counts_stay_consistent(tiny):
    decl, codes, cfg = tiny
    data = encode(codes, decl, cfg)
    state = new_state(data, cfg)
    rng = np.random.default_rng(0)
    for _ in range(20):
        gibbs_sweep(state, rng)
        check_counts(state)


def test_encoding_of_ordinals_and_hidden_cells(tiny):
    decl, codes, cfg = tiny
    data = encode(codes, decl, cfg)
    # ordinal at level 0: one "stop" token; level 2 (top): two "continue" tokens; level 1:
    # continue then stop
    o = [data.layout.subs["o"][0], data.layout.subs["o"][1]]
    words = {(int(d), int(s)): int(w - data.vocab_offset[s])
             for d, s, w in zip(data.disc_doc, data.disc_sub, data.disc_word) if s in o}
    assert words == {(0, o[0]): 1, (1, o[0]): 0, (1, o[1]): 0, (2, o[0]): 0, (2, o[1]): 1}
    hidden = np.zeros(codes.shape, dtype=bool)
    hidden[0, 3] = hidden[1, 0] = True
    fewer = encode(codes, decl, cfg, hidden=hidden)
    assert fewer.n_cells == data.n_cells - 1
    assert fewer.n_discrete_tokens == data.n_discrete_tokens - 1


def test_sampler_matches_exact_posterior(tiny):
    decl, codes, cfg = tiny
    data = encode(codes, decl, cfg)
    rows = enumerate_posterior(data, cfg)
    log_w = np.array([r[0] for r in rows])
    weights = np.exp(log_w - logsumexp(log_w))
    exact = sum(w * label_free_statistics(*r[1:]) for w, r in zip(weights, rows))

    state = new_state(data, cfg)
    rng = np.random.default_rng(1)
    total, n = 0.0, 0
    for sweep in range(40000):
        gibbs_sweep(state, rng)
        if sweep >= 200:
            total = total + label_free_statistics(state.z_disc, state.z_cont, state.w_cont)
            n += 1
    assert np.max(np.abs(total / n - exact)) < 0.02


def test_state_from_parameters_is_consistent(tiny):
    decl, codes, cfg = tiny
    data = encode(codes, decl, cfg)
    rng = np.random.default_rng(2)
    theta = rng.dirichlet(np.ones(2), size=data.n_records)
    phi = rng.random((2, data.n_vocab))
    from blat.encoding import normalise_blocks
    phi = normalise_blocks(phi, data)
    state = state_from_parameters(data, cfg, np.full(2, 0.7), theta, phi, rng)
    check_counts(state)
    assert np.all(state.z_disc >= 0) and np.all(state.z_cont >= 0)


def exact_log_evidence(data, cfg):
    return float(logsumexp([r[0] for r in enumerate_posterior(data, cfg)]))


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_ti_matches_exact_evidence(tiny, direction):
    decl, codes, cfg = tiny
    data = encode(codes, decl, cfg)
    exact = exact_log_evidence(data, cfg)
    rng = np.random.default_rng(3)
    start = None
    if direction == "backward":
        start = new_state(data, cfg)
        for _ in range(100):
            gibbs_sweep(start, rng)
    ti = TiConfig(temperatures=30, power=4, burn_per_temperature=50,
                  iters_per_temperature=1500, initial_burn=100, direction=direction,
                  sample_alpha=False)
    result = run_ti(data, cfg, ti, rng, start=start, progress=False)
    assert abs(result.log_evidence - exact) < max(4 * result.total_error, 0.1), \
        (result.log_evidence, exact, result.total_error)
