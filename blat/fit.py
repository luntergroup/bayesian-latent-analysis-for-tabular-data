"""`run`: fit one chain and write its summary and arrays.

The output directory holds

  summary.yaml      settings, token budget, timing, held-out scores (with a mask), label
                    stability and warnings
  types.tsv         per type: usage and alpha, with 95% intervals
  phi.tsv           per variable, level and type: phi with its 95% interval
  theta.tsv         per record: posterior mean theta and 95% interval
  diagnostics.tsv   per trace: ESS, autocorrelation time, MCSE, Geweke z, drift
  arrays.npz        the per-iteration traces (every iteration, burn-in included), the
                    per-draw phi and alpha, and the last draw's counts -- from which a
                    posterior draw of (alpha, theta, phi) is rebuilt to start a backward
                    integration
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import pandas as pd

from . import convert as C
from .config import ModelConfig, RunConfig, as_dict, sd_from_words
from .diagnostics import matched_cosine, summarise
from .encoding import encode
from .formats import write_table
from .io import Inputs, header, output_dir, write_summary
from .quantities import (DrawSummary, data_loglik, draw_phi, draw_theta, expected_y,
                         level_probabilities, log_jacobian, sweep_row, continuous_log_density)
from .sampler import PHASES, run_chain

#: Warning thresholds
MIN_OCCUPANCY = 0.01
MAX_GEWEKE = 3.0
MIN_ESS = 100.0
MIN_LABEL_COSINE = 0.95


class HeldOut:
    """Per-cell log predictive density of the hidden cells, accumulated over draws.

    A hidden cell is not in the fit, so theta_d from the chain has not seen it and the
    density is out of sample.  Discrete cells: the sum of log theta_d . phi over the
    cell's tokens, which for an ordinal is the log of its level probability.  Continuous
    cells: the exact marginal density of y (Jacobian included).
    """

    def __init__(self, inputs: Inputs, cfg: ModelConfig):
        decl = inputs.decl
        self.data = encode(inputs.codes, decl, cfg, hidden=~inputs.hidden)
        d = self.data
        names = decl.names
        var_of_sub = np.empty(d.n_sub, dtype=np.int64)
        for j, name in enumerate(names):
            var_of_sub[list(d.layout.subs[name])] = j
        keys = np.concatenate([d.disc_doc.astype(np.int64) * len(names)
                               + var_of_sub[d.disc_sub],
                               d.cell_doc.astype(np.int64) * len(names)
                               + var_of_sub[d.cell_sub]])
        self.keys, inverse = np.unique(keys, return_inverse=True)
        self.token_cell = inverse[:d.n_discrete_tokens]
        self.cont_cell = inverse[d.n_discrete_tokens:]
        self.kind = np.array([decl[names[k % len(names)]].kind for k in self.keys])
        self.jacobian = log_jacobian(d.cell_y)
        self.log_sum = np.full(self.keys.size, -np.inf)
        self.n = 0
        self.gamma = cfg.gamma

    def add(self, theta: np.ndarray, phi: np.ndarray) -> None:
        d = self.data
        tp = theta @ phi
        cell = np.zeros(self.keys.size)
        if d.n_discrete_tokens:
            np.add.at(cell, self.token_cell, np.log(tp[d.disc_doc, d.disc_word]))
        if d.n_cells:
            p1 = tp[d.cell_doc, d.vocab_offset[d.cell_sub] + 1]
            cell[self.cont_cell] += continuous_log_density(
                p1, d.cell_log_x, d.cell_log_1mx, d.cell_n, self.gamma) + self.jacobian
        self.log_sum = np.logaddexp(self.log_sum, cell)
        self.n += 1

    def lppd(self) -> np.ndarray:
        return self.log_sum - math.log(max(self.n, 1))

    def summary(self) -> dict:
        lppd = self.lppd()
        out = {"cells": int(self.keys.size), "lppd_mean": float(lppd.mean()) if lppd.size
               else math.nan}
        for kind in ("categorical", "ordinal", "continuous"):
            sel = self.kind == kind
            out[kind] = {"cells": int(sel.sum()), "lppd_sum": float(lppd[sel].sum()),
                         "lppd_mean": float(lppd[sel].mean()) if sel.any() else math.nan}
        return out


class Recorder:
    """The on_sweep callback: per-iteration rows always, per-draw work at the thinning stride."""

    def __init__(self, data, cfg: ModelConfig, run: RunConfig, rng, held: HeldOut | None,
                 ci_draws: int, progress: bool):
        self.data, self.cfg, self.run, self.rng, self.held = data, cfg, run, rng, held
        self.rows = []
        self.phases = []
        self.times = []
        n = run.n_draws
        self.draw_iter = np.zeros(n, dtype=np.int64)
        self.draw_alpha = np.zeros((n, cfg.K))
        self.draw_phi = np.zeros((n, cfg.K, data.n_vocab), dtype=np.float32)
        self.draw_log_lik = np.zeros(n)
        self.theta = DrawSummary(keep_every=max(1, n // max(ci_draws, 1)))
        self.kept = 0
        self.i = 0
        self.last = None
        self.progress = progress
        self.t0 = self.t_last = time.time()

    def __call__(self, state, phase, changed):
        now = time.time()
        self.rows.append(sweep_row(state, changed))
        self.phases.append(PHASES.index(phase))
        self.times.append(now - self.t_last)
        self.t_last = now
        if phase == "sampling" and (self.i - self.run.burnin) % self.run.thin == 0 \
                and self.kept < self.draw_iter.size:
            theta = draw_theta(state.counts_doc_topic, state.alpha, self.rng)
            phi = draw_phi(state.counts_topic_word, self.data, self.rng)
            j = self.kept
            self.draw_iter[j] = self.i
            self.draw_alpha[j] = state.alpha
            self.draw_phi[j] = phi
            self.draw_log_lik[j] = data_loglik(self.data, self.cfg.gamma, theta, phi)
            self.theta.add(theta, j)
            if self.held is not None:
                self.held.add(theta, phi)
            self.last = (state.counts_doc_topic.copy(), state.counts_topic_word.copy(),
                         state.alpha.copy())
            self.kept += 1
        self.i += 1
        if self.progress and (self.i % max(1, self.run.total // 20) == 0
                              or self.i == self.run.total):
            rate = self.i / (now - self.t0)
            eta = (self.run.total - self.i) / rate
            lik = f"  log lik {self.draw_log_lik[self.kept - 1]:.6g}" if self.kept else ""
            alpha = " ".join(f"{a:.3f}" for a in state.alpha)
            print(f"  iteration {self.i}/{self.run.total} ({phase})  log joint "
                  f"{self.rows[-1]['log_joint']:.6g}{lik}  alpha [{alpha}]  "
                  f"{rate:.1f} iterations/s  eta {eta:.0f}s",
                  file=sys.stderr, flush=True)

    def traces(self) -> dict:
        rows = self.rows
        out = {f"trace_{name}": np.array([r[name] for r in rows])
               for name in ("log_p_types", "log_p_words", "log_p_continuous", "log_joint",
                            "alpha_sum", "change_rate",
                            "change_discrete", "change_continuous", "k_eff",
                            "theta_entropy")}
        out["trace_alpha"] = np.array([r["alpha"] for r in rows])
        out["trace_occupancy"] = np.array([r["occupancy"] for r in rows])
        out["trace_phase"] = np.array(self.phases, dtype=np.int8)
        out["trace_seconds"] = np.array(self.times, dtype=np.float32)
        return out


def phi_table(draw_phi: np.ndarray, data, decl, cfg) -> pd.DataFrame:
    """phi in long form: per variable, level and type, the posterior mean and 95% interval.

    Discrete variables: the probability of each level (for an ordinal, reconstructed from
    its continuation sub-features).  Continuous variables: the value each type expects on
    the y scale and in original units, and p1 = phi_f1, the probability of a latent 1-word.
    Types are numbered 1..K in the sampler's order, as in theta.tsv and types.tsv.
    """
    lay = data.layout
    rows = []

    def add(name, kind, level, values):              # values: (draws, K)
        lo, hi = np.percentile(values, [2.5, 97.5], axis=0)
        for k in range(values.shape[1]):
            rows.append(dict(variable=name, kind=kind, level=level, type=k + 1,
                             mean=float(values[:, k].mean()), lo=float(lo[k]),
                             hi=float(hi[k])))

    for v in decl.variables:
        if v.kind == "continuous":
            base = lay.vocab_offset[lay.subs[v.name][0]]
            p1 = draw_phi[:, :, base + 1]
            ey = expected_y(p1, data.words[v.name], cfg.gamma)
            add(v.name, v.kind, "expected (original units)", C.inverse(ey, v.transform))
            add(v.name, v.kind, "expected (y scale)", ey)
            add(v.name, v.kind, "p1", p1)
        else:
            probs = level_probabilities(draw_phi, lay, v.name)     # (draws, K, levels)
            for j, level in enumerate(v.levels):
                add(v.name, v.kind, level, probs[:, :, j])
    return pd.DataFrame(rows)


def chain_diagnostics(traces: dict, draw_log_lik: np.ndarray, draw_phi: np.ndarray,
                      run: RunConfig) -> tuple[pd.DataFrame, dict]:
    """(table of trace summaries, dict of label stability, occupancy and warnings)."""
    sampling = traces["trace_phase"] == PHASES.index("sampling")
    per_sweep = {name: summarise(traces[f"trace_{name}"][sampling])
                 for name in ("log_joint", "log_p_types", "log_p_words", "log_p_continuous",
                              "alpha_sum", "k_eff",
                              "change_rate", "theta_entropy")}
    per_sweep["log_lik"] = summarise(draw_log_lik)
    warnings = []
    for name in ("log_joint", "log_lik", "alpha_sum", "k_eff"):
        s = per_sweep[name]
        if abs(s.get("geweke_z", 0.0) or 0.0) > MAX_GEWEKE:
            warnings.append(f"{name}: Geweke |z| = {abs(s['geweke_z']):.1f} > {MAX_GEWEKE}; "
                            "the chain may still be drifting")
        if s.get("ess", math.inf) < MIN_ESS:
            warnings.append(f"{name}: effective sample size {s['ess']:.0f} < {MIN_ESS:.0f}")
    occupancy = traces["trace_occupancy"][sampling][-max(1, run.n_draws):].mean(axis=0)
    if np.any(occupancy < MIN_OCCUPANCY):
        warnings.append(f"{int(np.sum(occupancy < MIN_OCCUPANCY))} type(s) hold less than "
                        f"{MIN_OCCUPANCY:.0%} of the tokens: nearly empty types")
    half = len(draw_phi) // 2
    stability = None
    if half >= 2:
        cos = matched_cosine(draw_phi[:half].mean(axis=0), draw_phi[half:].mean(axis=0))
        stability = {"matched_cosine": cos, "min": float(cos.min())}
        if cos.min() < MIN_LABEL_COSINE:
            warnings.append(f"type vectors of the two halves of the sampling phase match "
                            f"with cosine {cos.min():.3f} < {MIN_LABEL_COSINE}: possible "
                            "label switching, so per-type means and intervals are unreliable")
    table = pd.DataFrame([{"quantity": q, **s} for q, s in per_sweep.items()])
    return table, {"occupancy_sorted": occupancy, "label_stability": stability,
                   "warnings": warnings}


def fit(inputs: Inputs, cfg: ModelConfig, run: RunConfig, out, ci_draws: int = 500,
        progress: bool = True) -> str:
    """Run one chain; write OUT/summary.yaml, types.tsv, phi.tsv, theta.tsv,
    diagnostics.tsv and arrays.npz."""
    decl = inputs.decl
    data = encode(inputs.codes, decl, cfg, hidden=inputs.hidden)
    held = HeldOut(inputs, cfg) if inputs.hidden.any() else None
    rng = np.random.default_rng(run.seed)
    if progress:
        print(f"run: K={cfg.K}, {data.n_records} records, {data.n_tokens} tokens "
              f"({data.n_continuous_tokens} continuous), {run.total} iterations, "
              f"seed {run.seed}", file=sys.stderr, flush=True)
    recorder = Recorder(data, cfg, run, rng, held, ci_draws, progress)
    t0 = time.time()
    run_chain(data, cfg, run, rng, on_sweep=recorder)
    seconds = time.time() - t0

    n = recorder.kept
    draw_phi = recorder.draw_phi[:n]
    traces = recorder.traces()
    theta_mean = recorder.theta.mean()
    theta_lo, theta_hi = recorder.theta.interval()
    alpha = recorder.draw_alpha[:n]
    usage = theta_mean.mean(axis=0)
    diagnostics, checks = chain_diagnostics(traces, recorder.draw_log_lik[:n], draw_phi, run)

    out = output_dir(out)
    K = cfg.K
    types = pd.DataFrame({"type": np.arange(1, K + 1),
                          "usage": usage, "usage_rank": np.argsort(np.argsort(-usage)) + 1,
                          "alpha": alpha.mean(axis=0),
                          "alpha_lo": np.percentile(alpha, 2.5, axis=0),
                          "alpha_hi": np.percentile(alpha, 97.5, axis=0)})
    write_table(types, out / "types.tsv")
    write_table(phi_table(draw_phi, data, decl, cfg), out / "phi.tsv")
    theta = pd.DataFrame(index=inputs.frame.index)
    for label, values in (("", theta_mean), ("_lo", theta_lo), ("_hi", theta_hi)):
        for k in range(K):
            theta[f"type{k + 1}{label}"] = values[:, k]
    write_table(theta, out / "theta.tsv", index=True)
    write_table(diagnostics, out / "diagnostics.tsv")

    record = header("run")
    record.update(
        inputs=inputs.describe(out),
        model=as_dict(cfg), run=as_dict(run),
        words=data.words,
        residual_sd_of_words={k: sd_from_words(w, cfg.gamma_sum)
                              for k, w in data.words.items()},
        tokens={"discrete": data.n_discrete_tokens, "continuous": data.n_continuous_tokens,
                "continuous_share": data.n_continuous_tokens / max(data.n_tokens, 1),
                "per_record": data.n_tokens / max(data.n_records, 1)},
        timing={"seconds": seconds, "iterations_per_second": run.total / seconds},
        draws=n,
        alpha_sum=float(alpha.sum(axis=1).mean()),
        log_lik_mean=float(recorder.draw_log_lik[:n].mean()),
        **checks,
        tables={"types.tsv": "per type: usage (mean theta), alpha, 95% interval",
                "phi.tsv": "per variable, level and type: phi, 95% interval",
                "theta.tsv": "per record: posterior mean theta, 95% interval",
                "diagnostics.tsv": "per trace, sampling phase: ESS, MCSE, Geweke z, drift"})
    if held is not None:
        record["heldout"] = held.summary()

    counts_doc_topic, counts_topic_word, last_alpha = recorder.last
    arrays = dict(traces,
                  draw_iter=recorder.draw_iter[:n], draw_alpha=alpha,
                  draw_phi=draw_phi, draw_log_lik=recorder.draw_log_lik[:n],
                  theta_mean=theta_mean, theta_lo=theta_lo, theta_hi=theta_hi,
                  phi_mean=draw_phi.mean(axis=0).astype(float),
                  last_counts_doc_topic=counts_doc_topic,
                  last_counts_topic_word=counts_topic_word, last_alpha=last_alpha)
    if held is not None:
        arrays.update(heldout_key=held.keys, heldout_lppd=held.lppd())
    write_summary(out, record, arrays)
    if progress:
        print(f"run: wrote {out} ({seconds:.0f}s)", file=sys.stderr)
        for w in checks["warnings"]:
            print(f"  warning: {w}", file=sys.stderr)
    return str(out)
