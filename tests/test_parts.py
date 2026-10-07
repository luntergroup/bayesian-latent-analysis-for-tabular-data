"""Alpha sampling, densities, per-record inference, and the score sums."""

import numpy as np
from scipy.integrate import trapezoid

from blat import scores as S
from blat.config import sd_from_words, words_from_sd
from blat.encoding import encode, normalise_blocks
from blat.gscp import sample_alpha, sample_gscp
from blat.quantities import continuous_log_density, level_probabilities, x_of
from blat.record import GRID, batch, infer_theta, predict_continuous


def test_gscp_with_n0_is_exponential():
    rng = np.random.default_rng(0)
    draws = sample_gscp(np.full(20000, 2.0), 0, np.ones(20000), rng)
    assert abs(draws.mean() - 0.5) < 0.01


def test_alpha_posterior_recovers_truth():
    """With many records the posterior of alpha concentrates near the generating value."""
    rng = np.random.default_rng(1)
    alpha_true = np.array([0.3, 0.8, 1.5])
    theta = rng.dirichlet(alpha_true, size=3000)
    counts = np.array([rng.multinomial(200, t) for t in theta])
    alpha = np.ones(3)
    trace = []
    for i in range(300):
        alpha = sample_alpha(alpha, counts, kappa=1.0, rng=rng, iters=2)
        if i >= 100:
            trace.append(alpha)
    assert np.allclose(np.mean(trace, axis=0), alpha_true, rtol=0.15)


def test_words_and_sd_invert():
    for n in range(1, 20):
        assert words_from_sd(sd_from_words(n, 1.0) * 0.999, 1.0) == n


def test_continuous_density_integrates_to_one():
    # on the y scale: the Beta(., 0.5) singularities at x = 0, 1 are absorbed by dx/dy
    from blat.quantities import log_jacobian

    y = np.linspace(1e-6, 1 - 1e-6, 20001)
    x = x_of(y)
    for n, p in ((1, 0.2), (3, 0.7), (7, 0.5)):
        d = np.exp(continuous_log_density(np.full(x.size, p), np.log(x), np.log1p(-x),
                                          np.full(x.size, n), (0.5, 0.5)) + log_jacobian(y))
        assert abs(trapezoid(d, y) - 1) < 1e-3
        density_y = predict_continuous(np.array([p, p / 2]), n, (0.5, 0.5))
        assert abs(trapezoid(density_y, GRID) - 1) < 5e-3


def test_level_probabilities_sum_to_one(tiny):
    decl, codes, cfg = tiny
    data = encode(codes, decl, cfg)
    tp = normalise_blocks(np.random.default_rng(0).random((5, data.n_vocab)), data)
    for name in ("a", "b", "o"):
        p = level_probabilities(tp, data.layout, name)
        assert np.allclose(p.sum(axis=-1), 1)


def test_record_inference_matches_exact_single_token():
    """One discrete token: E[theta_k] = sum_z p(z|w) (alpha_k + [z=k]) / (sum alpha + 1)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import tiny_declaration
    from blat.config import ModelConfig

    decl = tiny_declaration(continuous_words=None)
    cfg = ModelConfig(K=3)
    codes = np.array([[1, np.nan, np.nan]])
    data = encode(codes, decl, cfg)
    phi = normalise_blocks(np.random.default_rng(3).random((3, data.n_vocab)), data)
    alpha = np.array([0.2, 0.5, 1.0])
    w = data.disc_word[0]
    pz = alpha * phi[:, w] / (alpha * phi[:, w]).sum()
    exact = (alpha + pz) / (alpha.sum() + 1)
    theta = infer_theta(batch(data, np.array([0])), phi, alpha, cfg.gamma, seed=1,
                        burn=10, draws=40000, rao_blackwell=True)
    assert np.allclose(theta[0].mean(axis=0), exact, atol=0.01)


def test_score_sums_combine_exactly():
    rng = np.random.default_rng(0)
    a, b = rng.random(300), rng.random(300) + 0.5
    whole = S.ratio_sums(a, b)
    first, second = S.ratio_sums(a[:100], b[:100]), S.ratio_sums(a[100:], b[100:])
    assert all(np.isclose(whole[k], first[k] + second[k]) for k in S.RATIO)
    r2, se = S.r2(300, *(whole[k] for k in S.RATIO))
    assert np.isclose(r2, 1 - a.sum() / b.sum())
    # the delta-method SE agrees with a bootstrap
    idx = rng.integers(0, 300, size=(3000, 300))
    boot = 1 - a[idx].sum(axis=1) / b[idx].sum(axis=1)
    assert abs(se - boot.std()) < 0.15 * boot.std()


def test_combine_scores_over_folds(tmp_path):
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 2, 200)
    pmf = rng.dirichlet([1, 1], size=200)
    base = np.array([0.5, 0.5])
    for i, part in enumerate((slice(0, 120), slice(120, 200))):
        s = S.variable_sums("categorical", truth[part], pmf[part].argmax(1), pmf=pmf[part],
                            baseline_pmf=base, log_density=np.zeros(truth[part].size))
        (tmp_path / f"fold{i}").mkdir()
        S.write(S.table({"v": s}), tmp_path / f"fold{i}" / "scores.tsv")
    whole = S.finish(S.variable_sums("categorical", truth, pmf.argmax(1), pmf=pmf,
                                     baseline_pmf=base, log_density=np.zeros(200)))
    combined = S.combine_scores([tmp_path / "fold0", tmp_path / "fold1"]).iloc[0]
    assert combined["n"] == 200
    assert np.isclose(combined["r2"], whole["r2"])
    assert np.isclose(combined["accuracy"], whole["accuracy"])
