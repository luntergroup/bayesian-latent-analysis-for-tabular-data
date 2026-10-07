import numpy as np

from blat.quantities import level_probabilities
from blat.simulate import declare, draw_truth, generate


def test_generated_levels_follow_the_model():
    """Empirical level frequencies match mean_d P(level | theta_d, phi)."""
    rng = np.random.default_rng(0)
    decl = declare(3, 3, 2)
    truth = draw_truth(decl, K=3, n_records=20000, kappa=2.0, beta=0.5, rng=rng)
    words = {v.name: 3 for v in decl.of_kind("continuous")}
    values = generate(decl, truth, truth["theta"], words, (0.5, 0.5), rng)
    tp = truth["theta"] @ truth["phi"]
    for v in decl.variables:
        if v.kind == "continuous":
            base = truth["layout"].vocab_offset[truth["layout"].subs[v.name][0]]
            x = 0.5 - 0.5 * np.cos(np.pi * values[v.name].to_numpy())
            n1_mean = 3 * tp[:, base + 1]
            assert abs(x.mean() - np.mean((n1_mean + 0.5) / 4)) < 0.01
            continue
        expected = level_probabilities(tp, truth["layout"], v.name).mean(axis=0)
        observed = np.bincount(values[v.name].astype(int), minlength=v.n_levels) / len(values)
        assert np.max(np.abs(observed - expected)) < 0.015, (v.name, observed, expected)
