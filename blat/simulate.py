"""`simulate`: synthetic data from the BLAT generative model itself.

Paper Sec. 3: alpha_k ~ Exp(kappa), theta_d ~ Dir(alpha), phi_kf ~ Dir(beta_f); every
token of a record draws a type from theta_d and a word from that type's phi.  Categorical
cells are one token; ordinal cells walk the continuation sub-features until the first "stop";
continuous cells draw N_f latent binary words, then x ~ Beta(n1 + gamma_1, n0 + gamma_0)
and y = arccos(1 - 2x) / pi.

The output is a *raw* table, as a user would provide: level labels as text, continuous
values in made-up units (some log-normal-like, some linear), missing cells MCAR at a
per-variable rate, and an ID column.  It goes through `convert` like any other table.
The variables have generic names and every parameter is set here by hand; nothing is
estimated from any real dataset.  Alongside: the declaration (so `convert` needs no
editing), the true alpha, theta and phi, and a second file of new records with extra holes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import words_from_sd
from .declaration import Declaration, Variable
from .formats import dump_yaml, write_table
from .encoding import build_layout

ORDINAL_SIZES = (3, 4, 5, 6, 7, 3, 4)


def declare(n_categorical: int, n_ordinal: int, n_continuous: int) -> Declaration:
    variables = []
    for i in range(n_categorical):
        levels = ("A", "B", "C") if i % 5 == 4 else ("no", "yes")
        variables.append(Variable(f"cat{i + 1:02d}", "categorical", levels))
    for i in range(n_ordinal):
        size = ORDINAL_SIZES[i % len(ORDINAL_SIZES)]
        variables.append(Variable(f"ord{i + 1:02d}", "ordinal",
                                  tuple(f"L{c}" for c in range(size)), confirmed=True))
    for i in range(n_continuous):
        variables.append(Variable.from_dict({"name": f"num{i + 1:02d}",
                                             "type": "continuous",
                                             "units": "a.u."}))
    return Declaration(variables=variables, id_column="id")


def scales(n_continuous: int, rng: np.random.Generator) -> list[dict]:
    """Made-up units per continuous variable: odd ones skewed (exponential of y)."""
    out = []
    for i in range(n_continuous):
        if i % 2:
            lo = float(rng.uniform(-1, 2))
            out.append({"log": True, "lo": lo, "hi": lo + float(rng.uniform(2, 5))})
        else:
            lo = float(rng.uniform(0, 100))
            out.append({"log": False, "lo": lo, "hi": lo + float(rng.uniform(20, 200))})
    return out


def draw_truth(decl: Declaration, K: int, n_records: int, kappa: float, beta: float,
               rng: np.random.Generator) -> dict:
    """alpha_k ~ Exp(kappa) independently, the model's prior; then theta and phi.

    With independent exponentials, alpha / sum(alpha) ~ Dir(1, ..., 1) whatever kappa is,
    and sum(alpha) ~ Gamma(K, kappa), whose mean K / kappa is 1 at kappa = K.  alpha is the
    first draw from the generator, so it depends on the seed, K and kappa only.
    """
    layout = build_layout(decl)
    alpha_v = rng.exponential(1.0 / kappa, size=K)
    theta = rng.dirichlet(alpha_v, size=n_records)
    phi = np.zeros((K, layout.n_vocab))
    for f in range(layout.n_sub):
        lo, hi = layout.vocab_offset[f], layout.vocab_offset[f + 1]
        phi[:, lo:hi] = rng.dirichlet(np.full(hi - lo, beta), size=K)
    return {"alpha": alpha_v, "theta": theta, "phi": phi, "layout": layout}


def _tokens(theta, phi, lo, hi, rng):
    """One token per record: a type from theta_d, then a word in [lo, hi) from phi."""
    cum = np.cumsum(theta, axis=1)
    k = (cum < rng.random(len(theta))[:, None] * cum[:, -1:]).sum(axis=1)
    k = np.minimum(k, theta.shape[1] - 1)
    block = phi[k, lo:hi]
    cum = np.cumsum(block, axis=1)
    w = (cum < rng.random(len(theta))[:, None] * cum[:, -1:]).sum(axis=1)
    return np.minimum(w, hi - lo - 1)


def generate(decl: Declaration, truth: dict, theta: np.ndarray, words: dict,
             gamma, rng: np.random.Generator) -> pd.DataFrame:
    """Level codes and y values, complete (no missingness yet)."""
    layout, phi = truth["layout"], truth["phi"]
    n = len(theta)
    out = {}
    for v in decl.variables:
        subs = layout.subs[v.name]
        if v.kind == "categorical":
            f = subs[0]
            out[v.name] = _tokens(theta, phi, layout.vocab_offset[f],
                                  layout.vocab_offset[f + 1], rng).astype(float)
        elif v.kind == "ordinal":
            level = np.full(n, v.n_levels - 1, dtype=float)
            active = np.arange(n)
            for c, f in enumerate(subs):
                base = layout.vocab_offset[f]
                stop = _tokens(theta[active], phi, base, base + 2, rng) == 1
                level[active[stop]] = c
                active = active[~stop]
            out[v.name] = level
        else:
            f = subs[0]
            base = layout.vocab_offset[f]
            n1 = sum(_tokens(theta, phi, base, base + 2, rng) for _ in range(words[v.name]))
            x = rng.beta(n1 + gamma[1], words[v.name] - n1 + gamma[0])
            out[v.name] = np.arccos(np.clip(1 - 2 * x, -1, 1)) / np.pi
    return pd.DataFrame(out)


def to_raw(values: pd.DataFrame, decl: Declaration, units: list[dict],
           missing: dict, rng: np.random.Generator, ids) -> pd.DataFrame:
    raw = pd.DataFrame(index=range(len(values)))
    raw["id"] = ids
    for i, v in enumerate(decl.of_kind("continuous")):
        u = units[i]
        scaled = u["lo"] + (u["hi"] - u["lo"]) * values[v.name].to_numpy()
        vals = np.exp(scaled) if u["log"] else scaled
        raw[v.name] = [f"{x:.4g}" for x in vals]
    for v in decl.variables:
        if v.kind != "continuous":
            raw[v.name] = [v.levels[int(c)] for c in values[v.name]]
    raw = raw[["id"] + decl.names]
    for v in decl.variables:
        holes = rng.random(len(raw)) < missing[v.name]
        raw.loc[holes, v.name] = ""
    return raw


def simulate(out_dir, K: int, n_records: int = 2500, n_categorical: int = 16,
             n_ordinal: int = 7, n_continuous: int = 10, kappa: float | None = None,
             beta: float = 0.5, residual_sd: float = 0.2, gamma=(0.5, 0.5),
             max_missing: float = 0.3, n_new: int = 200, new_missing: float = 0.2,
             seed: int = 0, name: str = "example") -> dict:
    """Write NAME.raw.csv, NAME.new.raw.csv, NAME.declaration.yaml and NAME.truth.*."""
    if K is None or K < 1:
        raise ValueError("simulate needs the number of true types K")
    kappa = float(K) if kappa is None else float(kappa)     # E[sum alpha] = K / kappa = 1
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    decl = declare(n_categorical, n_ordinal, n_continuous)
    words = {v.name: words_from_sd(residual_sd, sum(gamma)) for v in decl.of_kind("continuous")}
    truth = draw_truth(decl, K, n_records + n_new, kappa, beta, rng)
    units = scales(n_continuous, rng)
    values = generate(decl, truth, truth["theta"], words, gamma, rng)
    missing = {v.name: float(rng.uniform(0, max_missing)) for v in decl.variables}

    ids = [f"R{i + 1:05d}" for i in range(n_records + n_new)]
    raw = to_raw(values.iloc[:n_records].reset_index(drop=True), decl, units, missing, rng,
                 ids[:n_records])
    new_rate = {k: min(0.9, m + new_missing) for k, m in missing.items()}
    new = to_raw(values.iloc[n_records:].reset_index(drop=True), decl, units, new_rate, rng,
                 ids[n_records:])

    paths = {"raw": out / f"{name}.raw.csv", "new": out / f"{name}.new.raw.csv",
             "declaration": out / f"{name}.declaration.yaml",
             "truth": out / f"{name}.truth.yaml",
             "truth_variables": out / f"{name}.truth.tsv",
             "truth_arrays": out / f"{name}.truth.npz"}
    raw.to_csv(paths["raw"], index=False)
    new.to_csv(paths["new"], index=False)
    decl.source = paths["raw"].name
    decl.save(paths["declaration"])
    np.savez_compressed(paths["truth_arrays"], alpha=truth["alpha"],
                        theta=truth["theta"][:n_records],
                        theta_new=truth["theta"][n_records:], phi=truth["phi"],
                        values=values.iloc[:n_records].to_numpy(),
                        values_new=values.iloc[n_records:].to_numpy())
    dump_yaml({"K": K, "seed": seed, "kappa": kappa, "alpha": truth["alpha"],
               "alpha_sum": float(truth["alpha"].sum()),
               "alpha_normalised": truth["alpha"] / truth["alpha"].sum(),
               "theta_mean": truth["theta"][:n_records].mean(axis=0),
               "records": n_records, "new_records": n_new, "beta": beta,
               "gamma": list(gamma), "residual_sd": residual_sd,
               "sub_variables": list(truth["layout"].names),
               "files": {"truth.tsv": "per variable: kind, missingness, words, units",
                         "truth.npz": "alpha, theta, phi (sub-variable layout above), "
                                      "and the complete values before missingness"}},
              paths["truth"], header=f"The parameters {name}.raw.csv was drawn from.")
    observed = raw[decl.names].ne("")
    rows = []
    for v in decl.variables:
        row = {"variable": v.name, "kind": v.kind, "missing_rate": missing[v.name],
               "missing_realised": float(1 - observed[v.name].mean())}
        if v.kind == "continuous":
            u = units[decl.of_kind("continuous").index(v)]
            row.update(words=words[v.name], log_units=u["log"], lo=u["lo"], hi=u["hi"])
        else:
            counts = raw[v.name][observed[v.name]].value_counts()
            row["level_counts"] = ";".join(f"{l}={int(counts.get(l, 0))}" for l in v.levels)
        rows.append(row)
    write_table(pd.DataFrame(rows), paths["truth_variables"])
    return {k: str(p) for k, p in paths.items()}
