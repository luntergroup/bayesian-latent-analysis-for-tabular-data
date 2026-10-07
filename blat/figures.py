"""Loading step outputs and drawing the paper's figures.  Used by the notebooks.

    figure1-convergence  log joint traces, one panel per K       `plot_traces`
    figure2-evidence     log p(x | K) by thermodynamic integration `plot_evidence`
    figure3-tsne         t-SNE of the type vectors phi_k across K  `plot_type_embedding`
    figure4-loadings     every record's mixture theta_d            `plot_theta`
    figure5-types        phi: discrete heatmap and continuous      `plot_discrete`,
                         expected values                           `plot_continuous`
    figure6-chains       t-SNE of every final chain's types        `plot_chain_embedding`

Types are exchangeable, so everything here orders them by usage (mean theta), and picks
as the representative chain at each K the medoid: the chain whose types agree best with
all the others'.  Across chains, only label-invariant summaries are compared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import convert as C
from .diagnostics import match_types, matched_cosine, medoid, split_r_hat
from .io import SUMMARY, locate, read_summary
from .quantities import expected_y, level_probabilities
from .runs import Run
from .sampler import PHASES

#: The colours of types 1-5; further types get colours from `type_colours`.
TYPE_COLOURS = ("#0082C2", "#E6BF00", "#00AE53", "#6F4C9B", "#E54E00")
TRACE_COLOUR = "#1f4e79"


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> CIE Lab (D65), in which distance approximates perceived difference."""
    rgb = np.asarray(rgb, dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.2126, 0.0193],
                          [0.3576, 0.7152, 0.1192],
                          [0.1805, 0.0722, 0.9505]]) / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def type_colours(k: int) -> list:
    """k distinguishable colours: TYPE_COLOURS first, then each further one the candidate
    most different (in Lab) from all chosen so far.

    Deterministic, and a prefix of the list for any larger k, so type t has the same
    colour whatever K is.  Candidates are hues on three lightness levels; very light
    ones are excluded, since they vanish against the white background.
    """
    from matplotlib.colors import hsv_to_rgb, to_rgb

    chosen = [to_rgb(c) for c in TYPE_COLOURS[:k]]
    if k <= len(chosen):
        return chosen
    hue = np.linspace(0, 1, 72, endpoint=False)
    levels = [(0.85, 0.75), (0.55, 0.95), (0.9, 0.5), (0.35, 0.65)]   # (saturation, value)
    pool = np.concatenate([hsv_to_rgb(np.stack([hue, np.full_like(hue, s),
                                                np.full_like(hue, v)], axis=1))
                           for s, v in levels] + [np.array([[0.3, 0.3, 0.3]])])
    lab_pool = _lab(pool)
    dark_enough = lab_pool[:, 0] < 85
    pool, lab_pool = pool[dark_enough], lab_pool[dark_enough]
    distance = np.min(np.linalg.norm(lab_pool[:, None] - _lab(np.array(chosen))[None],
                                     axis=2), axis=1)
    while len(chosen) < k:
        i = int(np.argmax(distance))
        chosen.append(tuple(pool[i]))
        distance = np.minimum(distance, np.linalg.norm(lab_pool - lab_pool[i], axis=1))
    return chosen


# --- loading ----------------------------------------------------------------------------

def outputs(directory: str | Path, step: str):
    """The output directories of `step` directly under `directory`, with their summaries."""
    for path in sorted(Path(directory).glob(f"*/{SUMMARY}")):
        record = read_summary(path)
        if record.get("format") == f"blat-{step}/1":
            yield path.parent, record


def load_runs(directory: str | Path) -> dict[int, list[Run]]:
    """Every run under `directory`, grouped by K."""
    out: dict[int, list[Run]] = {}
    for path, _ in outputs(directory, "run"):
        run = Run(path)
        out.setdefault(run.K, []).append(run)
    if not out:
        raise FileNotFoundError(f"no run outputs under {directory}")
    return dict(sorted(out.items()))


def representative(runs: list[Run]) -> tuple[Run, dict]:
    """The medoid chain, and the agreement of every chain's types with it."""
    phis = [r.phi_mean for r in runs]
    best, scores = medoid(phis)
    agreement = [float(matched_cosine(phis[best], p).min()) for p in phis]
    return runs[best], {"index": best, "scores": scores, "min_cosine_to_medoid": agreement}


def usage_order(run: Run) -> np.ndarray:
    return np.argsort(-run.theta_mean.mean(axis=0))


def cross_chain_table(by_k: dict[int, list[Run]]) -> pd.DataFrame:
    """Per K: split R-hat of the log joint and of sum alpha over chains, and type
    agreement."""
    rows = []
    for k, runs in by_k.items():
        sampling = [r.arrays["trace_phase"] == PHASES.index("sampling") for r in runs]
        qa = [r.arrays["trace_log_joint"][s] for r, s in zip(runs, sampling)]
        sa = [r.arrays["trace_alpha_sum"][s] for r, s in zip(runs, sampling)]
        n = min(len(q) for q in qa)
        rep, info = representative(runs)
        rows.append(dict(K=k, chains=len(runs),
                         r_hat_log_joint=split_r_hat(np.array([q[:n] for q in qa])),
                         r_hat_alpha_sum=split_r_hat(np.array([s[:n] for s in sa])),
                         mean_log_joint=float(np.mean([q.mean() for q in qa])),
                         sd_between_chains=float(np.std([q.mean() for q in qa], ddof=1))
                         if len(qa) > 1 else np.nan,
                         medoid=rep.path.name,
                         worst_type_agreement=float(min(info["min_cosine_to_medoid"]))))
    return pd.DataFrame(rows)


def load_evidence(directory: str | Path) -> pd.DataFrame:
    rows = []
    for path, r in outputs(directory, "ti"):
        rows.append(dict(K=r["K"], direction=r["direction"], file=path.name,
                         log_evidence=r["log_evidence"], total_error=r["total_error"],
                         mc_error=r["mc_error"], quadrature_error=r["quadrature_error"],
                         target=r["target"]))
    if not rows:
        raise FileNotFoundError(f"no ti outputs under {directory}")
    return pd.DataFrame(rows).sort_values(["K", "direction", "file"]).reset_index(drop=True)


# --- Fig 1, convergence -----------------------------------------------------------------

def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window)
    return (np.convolve(values, kernel, mode="same")
            / np.convolve(np.ones_like(values), kernel, mode="same"))


def plot_traces(by_k: dict[int, list[Run]], ncols: int = 3, width: float = 9.0,
                panel_height: float = 2.1, window: int = 25, show_burn_in: float = 0.1,
                log_x: bool = True):
    """One panel per K: every chain's log joint over the iterations.

    The x axis is logarithmic so that the burn-in is visible; the dashed line marks where
    alpha starts being sampled and the dotted line the end of burn-in.  Each panel has its
    own y range (the log joint is not comparable across K), set from the sampling phase.
    A faint raw trace is drawn under a running mean over `window` iterations.
    """
    import matplotlib.pyplot as plt

    ks = list(by_k)
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axes = plt.subplots(nrows, min(ncols, len(ks)), squeeze=False,
                             figsize=(width, panel_height * nrows))
    for index, (ax, k) in enumerate(zip(axes.ravel(), ks)):
        lows, highs = [], []
        first = None
        for run in by_k[k]:
            q = run.arrays["trace_log_joint"]
            sweeps = np.arange(1, q.size + 1)
            kept = run.arrays["trace_phase"] == PHASES.index("sampling")
            lows.append(np.percentile(q[kept], 0.5))
            highs.append(np.percentile(q[kept], 99.5))
            first = max(1, int(run.run_cfg.burnin * show_burn_in))
            ax.plot(sweeps, q, lw=0.3, alpha=0.2, color=TRACE_COLOUR)
            ax.plot(sweeps, smooth(q, window), lw=0.8, alpha=0.8, color=TRACE_COLOUR)
        cfg = by_k[k][0].run_cfg
        if cfg.alpha_frozen:
            ax.axvline(cfg.alpha_frozen, color="0.4", ls="--", lw=0.8)
        ax.axvline(cfg.burnin, color="0.4", ls=":", lw=0.9)
        lo, hi = min(lows), max(highs)
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlim(first, cfg.total)
        ax.set_title(f"K = {k}", fontsize=10)
        ax.set_xlabel("iteration")
        if index % ncols == 0:
            ax.set_ylabel("log joint")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(ks):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


# --- Fig 2, evidence --------------------------------------------------------------------

def plot_evidence(paths: pd.DataFrame, directions=("forward", "backward"),
                  width: float = 6.5, height: float = 4.0, group_gap: float = 0.25):
    """One bar per thermodynamic-integration path, grouped by K, with its error.

    A log evidence has no natural zero, so the bars stand on a floor below the lowest
    path: read differences, not heights.  Forward and backward paths are shown side by
    side; where they disagree by more than their errors, the ladder was too short.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colours = {"forward": "#1f4e79", "backward": "#8ab6d6"}
    paths = paths[paths["direction"].isin(directions)]
    ks = sorted(paths["K"].unique())
    floor = (paths["log_evidence"] - paths["total_error"]).min()
    span = paths["log_evidence"].max() - floor
    base = floor - 0.08 * max(span, 1.0)
    per_group = int(paths.groupby("K").size().max())
    slot = (1 - group_gap) / per_group
    fig, ax = plt.subplots(figsize=(width, height))
    for i, k in enumerate(ks):
        group = paths[paths["K"] == k].sort_values(["direction", "file"],
                                                   key=lambda c: c.map(
                                                       {"forward": 0, "backward": 1})
                                                   if c.name == "direction" else c)
        for j, (_, row) in enumerate(group.iterrows()):
            x = i + (j - (len(group) - 1) / 2) * slot
            ax.bar(x, row["log_evidence"] - base, bottom=base, width=slot * 0.9,
                   color=colours[row["direction"]], edgecolor="white", lw=0.4, zorder=3)
            ax.errorbar(x, row["log_evidence"], yerr=row["total_error"], fmt="none",
                        ecolor="0.15", elinewidth=0.9, capsize=2, zorder=4)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_xlabel("number of types $K$")
    ax.set_ylabel(r"$\log p(x \mid K)$")
    ax.set_ylim(base, None)
    ax.grid(True, axis="y", alpha=0.3, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    present = [d for d in directions if d in set(paths["direction"])]
    if len(present) > 1:
        ax.legend(handles=[Patch(facecolor=colours[d], label=d) for d in present],
                  frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


# --- Fig 3, t-SNE ------------------------------------------------------------------------------

def type_vectors(by_k: dict[int, list[Run]]) -> tuple[np.ndarray, pd.DataFrame]:
    """The representative chain's phi rows at every K, with K, type and usage."""
    vectors, rows = [], []
    for k, runs in by_k.items():
        run, _ = representative(runs)
        order = usage_order(run)
        usage = run.theta_mean.mean(axis=0)
        for t, i in enumerate(order):
            vectors.append(run.phi_mean[i])
            rows.append(dict(K=k, type=t + 1, usage=float(usage[i])))
    return np.array(vectors), pd.DataFrame(rows)


def default_perplexity(n: int) -> float:
    """min(10, (n - 1) / 3), at least 2: t-SNE needs a perplexity below the point count."""
    return max(2.0, min(10.0, (n - 1) / 3.0))


def embed(vectors: np.ndarray, seed: int = 0, perplexity: float | None = None):
    """t-SNE on the cosine distance.  With a few dozen points the layout is illustration:
    read the neighbourhoods, not the distances.  `perplexity` None: `default_perplexity`."""
    from sklearn.manifold import TSNE

    n = len(vectors)
    perplexity = perplexity or default_perplexity(n)
    if not perplexity < n:
        raise ValueError(f"perplexity {perplexity} must be below the number of points ({n})")
    return TSNE(n_components=2, metric="cosine", init="pca", random_state=seed,
                perplexity=perplexity).fit_transform(vectors)


def plot_type_embedding(points: np.ndarray, labels: pd.DataFrame, width=6.5, height=5.5,
                        largest: float = 1500.0):
    """Bubbles at the embedded type vectors: colour = K, area = usage."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    ks = sorted(labels["K"].unique())
    cmap = colormaps["viridis"]
    colours = {k: cmap(i / max(len(ks) - 1, 1)) for i, k in enumerate(ks)}
    fig, ax = plt.subplots(figsize=(width, height))
    for k in ks[::-1]:
        sel = (labels["K"] == k).to_numpy()
        ax.scatter(points[sel, 0], points[sel, 1], s=largest * labels.loc[sel, "usage"],
                   color=colours[k], alpha=0.55, edgecolor="white", lw=0.6,
                   label=f"K = {k}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    handles, names = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], names[::-1], frameon=False, fontsize=8, loc="best",
                       markerscale=0.3)
    for handle in legend.legend_handles:
        handle.set_alpha(0.8)
    fig.tight_layout()
    return fig


# --- Fig 4, loadings --------------------------------------------------------------------

def order_records(theta: np.ndarray, optimal: bool = True) -> np.ndarray:
    """Hierarchical clustering (cosine distance, average linkage) so that similar mixtures
    sit next to each other; `optimal` flips the tree's branches to minimise the distance
    between neighbours (slow beyond a few thousand records)."""
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import pdist

    if len(theta) < 3:
        return np.arange(len(theta))
    distance = pdist(theta, metric="cosine")
    tree = linkage(distance, method="average")
    if optimal:
        tree = optimal_leaf_ordering(tree, distance)
    return leaves_list(tree)


def plot_theta(theta: np.ndarray, width=9.0, height=3.2, colours=None):
    """Records across, mixture up: one stacked column per record.

    `colours` defaults to `type_colours(K)`; the legend wraps into columns of 12.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n, k = theta.shape
    colours = type_colours(k) if colours is None else colours
    if len(colours) < k:
        raise ValueError(f"{len(colours)} colours for {k} types")
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(width, height))
    bottom = np.zeros(n)
    for t in range(k):
        ax.fill_between(x, bottom, bottom + theta[:, t], step="mid", lw=0,
                        color=colours[t])
        bottom = bottom + theta[:, t]
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("record (reordered)")
    ax.set_ylabel(r"mixture weight $\theta$")
    ax.set_yticks([0, 0.5, 1])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=colours[t], label=str(t + 1)) for t in range(k)],
              title="type", frameon=False, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              ncol=int(np.ceil(k / 12)), fontsize=8 if k > 12 else None)
    fig.tight_layout()
    return fig


def mixing_summary(theta: np.ndarray, pure_at: float = 0.5) -> pd.Series:
    largest = theta.max(axis=1)
    return pd.Series({"records": len(theta), "mean largest weight": float(largest.mean()),
                      f"fraction with a weight above {pure_at}": float((largest > pure_at)
                                                                       .mean())})


# --- Fig 5, types -----------------------------------------------------------------------

def discrete_table(run: Run, order=None) -> tuple[pd.DataFrame, list[str]]:
    """(rows, K) phi for the discrete variables: one row per level, except that a binary
    categorical shows only its second level (the first is the complement)."""
    order = usage_order(run) if order is None else order
    rows, owner, names = [], [], []
    for v in run.decl.variables:
        if v.kind == "continuous":
            continue
        name, levels = v.name, v.levels
        p = level_probabilities(run.phi_mean, run.data.layout, name)[order]  # (K, levels)
        pick = [1] if (v.kind == "categorical" and len(levels) == 2) else range(len(levels))
        for j in pick:
            rows.append(p[:, j])
            names.append(f"{name}: {levels[j]}")
            owner.append(name)
    return pd.DataFrame(np.array(rows), index=names,
                        columns=[str(t + 1) for t in range(run.K)]), owner


def plot_discrete(table: pd.DataFrame, owner: list[str], usage=None, explained=None,
                  row_height: float = 0.22, width: float = 5.0):
    """Heatmap of phi for the discrete variables, types across.

    `usage` (per type, in column order) is printed above the columns as a percentage;
    `explained` (per variable) to the right of each variable's first row.
    """
    import matplotlib.pyplot as plt

    k = table.shape[1]
    fig, ax = plt.subplots(figsize=(width, max(2.5, row_height * len(table) + 1.2)))
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap="YlOrBr", vmin=0, vmax=1,
                      interpolation="nearest")
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index, fontsize=7)
    ax.set_xticks(range(k))
    ax.set_xticklabels(table.columns, fontsize=8)
    ax.set_xlabel("type")
    for i in range(1, len(owner)):
        if owner[i] != owner[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=1.5)
    if usage is not None:
        for j, u in enumerate(usage):
            ax.text(j, -0.9, f"{100 * u:.0f}", ha="center", va="bottom", fontsize=7,
                    color="0.25")
        ax.text(-0.6, -0.9, "frequency (%)", ha="right", va="bottom", fontsize=7,
                style="italic", color="0.25")
    if explained is not None:
        seen = set()
        for i, name in enumerate(owner):
            if name not in seen and name in explained and np.isfinite(explained[name]):
                ax.text(k - 0.4, i, f"{explained[name]:.2f}", va="center", fontsize=7,
                        color="0.25")
            seen.add(name)
        ax.text(k - 0.4, len(owner) + 0.3, "explained", fontsize=7, style="italic",
                color="0.25", va="top")
    fig.colorbar(image, ax=ax, shrink=0.3, pad=0.12 if explained is not None else 0.03,
                 label=r"$\phi_{kfw}$")
    fig.tight_layout()
    return fig


def continuous_table(run: Run, order=None) -> pd.DataFrame:
    """(variables, K) expected value per type in original units."""
    order = usage_order(run) if order is None else order
    lay = run.data.layout
    out = {}
    for v in run.decl.of_kind("continuous"):
        p1 = run.phi_mean[:, lay.vocab_offset[lay.subs[v.name][0]] + 1]
        ey = expected_y(p1, run.data.words[v.name], run.cfg.gamma)
        out[v.name] = C.inverse(ey, v.transform)[order]
    return pd.DataFrame(out, index=[str(t + 1) for t in range(run.K)]).T


def plot_continuous(run: Run, table: pd.DataFrame, explained=None, ncols: int = 2,
                    width: float = 6.5, panel_height: float = 1.9, inner: float = 0.95):
    """One panel per continuous variable: the value each type expects, in the units of
    the data (left axis) and on the model's y scale (right axis).  Each y axis spans the
    inner 95% of the observed values."""
    import matplotlib.pyplot as plt

    decl = run.decl
    frame = run.inputs.frame
    names = list(table.index)
    k = table.shape[1]
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(width, panel_height * nrows))
    x = np.arange(1, k + 1)
    tail = (1 - inner) / 2
    for ax, name in zip(axes.ravel(), names):
        v = decl[name]
        y_obs = frame[name].dropna().to_numpy(float)
        lo_y, hi_y = np.quantile(y_obs, [tail, 1 - tail])
        lo, hi = C.inverse(np.array([lo_y, hi_y]), v.transform)
        values = table.loc[name].to_numpy(float)
        low, high = min(lo, values.min()), max(hi, values.max())
        pad = 0.05 * (high - low)
        log_axis = v.transform["fitted"]["log"]
        ax.scatter(x, values, s=40, color="crimson", zorder=3)
        if log_axis:
            from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
            ax.set_yscale("log")
            ax.set_ylim(low / 1.1, high * 1.1)
            ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.set_ylim(low - pad, high + pad)
        ax.set_xticks(x)
        ax.set_xlim(0.5, k + 0.5)
        ax.grid(True, alpha=0.3, lw=0.6)
        title = name if explained is None or name not in explained \
            else f"{name}   $R^2$ = {explained[name]:.2f}"
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(v.units or "", fontsize=7)
        ax.tick_params(labelsize=7)
        twin = ax.twinx()
        twin.set_yscale(ax.get_yscale())
        twin.set_ylim(ax.get_ylim())
        ticks = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
        if log_axis:
            twin.yaxis.set_minor_locator(NullLocator())
            ticks = [t for t in ticks if t > 0]
        twin.set_yticks(ticks)
        twin.set_yticklabels([f"{y:.2f}" for y in C.forward(np.array(ticks), v.transform)],
                             fontsize=6)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    axes[-1, 0].set_xlabel("type")
    fig.tight_layout()
    return fig


# --- Fig 6, agreement of the final chains -----------------------------------------------

def chain_type_vectors(runs: list[Run]) -> tuple[np.ndarray, pd.DataFrame]:
    """Every chain's type vectors, each matched one-to-one to a type of the medoid chain.

    Types are numbered as in the medoid, by usage; `cosine` is the similarity of the
    match.  If the chains found the same solution, every type's matches lie close together.
    """
    rep, info = representative(runs)
    number = np.empty(rep.K, dtype=int)
    number[usage_order(rep)] = np.arange(1, rep.K + 1)
    vectors, rows = [], []
    for c, run in enumerate(runs):
        sim, mine, theirs = match_types(rep.phi_mean, run.phi_mean)
        usage = run.theta_mean.mean(axis=0)
        for i, j in zip(mine, theirs):
            vectors.append(run.phi_mean[j])
            rows.append(dict(chain=run.name, medoid=c == info["index"], type=int(number[i]),
                             usage=float(usage[j]), cosine=float(sim[i, j])))
    return np.array(vectors), pd.DataFrame(rows)


def plot_chain_embedding(points: np.ndarray, labels: pd.DataFrame, width=6.5, height=5.5,
                         largest: float = 1500.0):
    """Bubbles at the embedded type vectors of every chain: colour = the medoid type each
    is matched to, area = usage; the medoid chain's bubbles have a dark edge."""
    import matplotlib.pyplot as plt

    colours = type_colours(int(labels["type"].max()))
    fig, ax = plt.subplots(figsize=(width, height))
    for t in sorted(labels["type"].unique()):
        sel = (labels["type"] == t).to_numpy()
        medoid_chain = labels.loc[sel, "medoid"].to_numpy()
        ax.scatter(points[sel, 0], points[sel, 1], s=largest * labels.loc[sel, "usage"],
                   color=colours[t - 1], alpha=0.55, label=f"type {t}",
                   edgecolor=np.where(medoid_chain, "black", "white"),
                   linewidth=np.where(medoid_chain, 1.2, 0.6))
    ax.scatter([], [], facecolor="none", edgecolor="black", linewidth=1.2,
               label="medoid chain")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    legend = ax.legend(frameon=False, fontsize=8, loc="best")
    for handle in legend.legend_handles[:-1]:
        handle.set_sizes([50])
        handle.set_edgecolor("white")
        handle.set_alpha(0.8)
    legend.legend_handles[-1].set_sizes([50])
    fig.tight_layout()
    return fig


def load_scores(path: str | Path) -> pd.DataFrame:
    """scores.tsv of an explain or impute output directory, indexed by variable."""
    path = Path(path)
    return pd.read_csv(path if path.suffix == ".tsv" else path / "scores.tsv",
                       sep="\t").set_index("variable")


def load_explained(path: str | Path) -> dict:
    """{variable: out-of-sample R^2} from an explain output directory."""
    return load_scores(path)["r2"].to_dict()


def resolve(path: str | Path, anchor: str | Path) -> Path:
    return locate(str(path), anchor)
