"""Type colours stay distinct for large K; the first five never change."""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from matplotlib.colors import to_rgb          # noqa: E402

from blat import figures as F                 # noqa: E402


def test_first_five_colours_are_fixed_and_prefixes_are_stable():
    assert F.type_colours(5) == [to_rgb(c) for c in F.TYPE_COLOURS]
    assert F.type_colours(3) == F.type_colours(5)[:3]
    assert F.type_colours(30)[:20] == F.type_colours(20)


@pytest.mark.parametrize("k", [6, 12, 20, 30])
def test_colours_are_distinguishable(k):
    lab = F._lab(np.array(F.type_colours(k)))
    distance = np.linalg.norm(lab[:, None] - lab[None], axis=2) + np.eye(k) * 1e9
    assert distance.min() > 20          # Delta E; about 2 is just noticeable


def test_theta_plot_for_many_types():
    import matplotlib
    matplotlib.use("Agg")
    theta = np.random.default_rng(0).dirichlet(np.full(20, 0.3), size=200)
    fig = F.plot_theta(theta)
    legend = fig.axes[0].get_legend()
    assert len(legend.get_texts()) == 20
    colours = {tuple(np.round(h.get_facecolor()[:3], 4)) for h in legend.legend_handles}
    assert len(colours) == 20


def test_chains_are_matched_to_the_medoid():
    """Chains holding the same types in different orders match back type for type."""
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    phi = rng.dirichlet(np.full(8, 0.3), size=3)
    theta = rng.dirichlet(np.ones(3), size=50)
    runs = [SimpleNamespace(name=f"c{c}", K=3, phi_mean=phi[p], theta_mean=theta[:, p])
            for c, p in enumerate(([0, 1, 2], [2, 0, 1], [1, 2, 0]))]
    vectors, labels = F.chain_type_vectors(runs)
    assert vectors.shape == (9, 8) and np.allclose(labels["cosine"], 1.0)
    for t, group in labels.groupby("type"):
        assert np.allclose(vectors[group.index], vectors[group.index[0]])
    assert labels["medoid"].sum() == 3
