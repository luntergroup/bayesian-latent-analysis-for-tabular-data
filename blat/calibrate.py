"""`calibrate` (optional): estimate s_f, hence N_f, per continuous variable.

s_f is the residual sd of a continuous variable on the y scale, and sets N_f, the number
of latent words a cell carries -- and so how much weight the variable has against the
discrete ones.  The paper's final fit estimated it rather than fixing s_f = 0.2:

    s_f^2 = 1/(n_f - 1) sum_d (y(mu_df) - y_df)^2,   mu_df = (N_f a_df + gamma_1)/(N_f + gamma)
    N_f   = max(1, floor(2 / (pi^2 s_f^2) - gamma - 1))

with a_df = theta_d . phi_f1 computed *predictively*: theta_d re-inferred from the
record's other cells, phi and alpha fixed (`explain.score_variable`).

N_f cannot change inside a chain without changing the target under the sampler, so the
estimate alternates: a complete short chain at fixed N_f, then one predictive pass, then
an update.  Optionally the first pass uses a finished run's phi instead of a short chain.
There is no stopping rule; the trace of every round is written so that a loop which has
not settled is visible.  The output directory holds declaration.yaml -- the input
declaration with `words` filled in, for `run` -- trace.tsv and summary.yaml.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from .config import ModelConfig, RunConfig, as_dict, words_from_sd
from .encoding import encode
from .explain import score_variable
from .formats import write_table
from .io import Inputs, header, output_dir, relative, write_summary
from .quantities import mean_phi
from .sampler import run_chain


def estimate(data, codes, decl, cfg, phi, alpha, burn, draws, seed, workers):
    from concurrent.futures import ThreadPoolExecutor

    columns = [v.name for v in decl.of_kind("continuous")]

    def one(c):
        return score_variable(data, codes, decl, cfg, c, phi, alpha, None, burn, draws,
                              seed)["s_hat"]

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(zip(columns, pool.map(one, columns)))
    return {c: one(c) for c in columns}


def calibrate(inputs: Inputs, cfg: ModelConfig, out, rounds: int = 4,
              iters: int = 1000, burn: int = 200, draws: int = 200, seed: int = 0,
              from_run=None, workers: int = 1, progress: bool = True) -> str:
    decl = inputs.decl.copy()
    codes = np.where(inputs.hidden, np.nan, inputs.codes)
    continuous = [v.name for v in decl.of_kind("continuous")]
    if not continuous:
        raise ValueError("the declaration has no continuous variables to calibrate")
    burnin = min(500, max(2, iters // 2))
    chain = RunConfig(seed=seed, alpha_frozen=min(200, burnin - 1), burnin=burnin,
                      iters=max(1, iters - burnin), thin=10)
    trace = []
    t0 = time.time()

    def record_iteration(i, sd, words_used):
        new = {c: words_from_sd(max(sd[c], cfg.residual_sd_floor), cfg.gamma_sum)
               for c in continuous}
        for c in continuous:
            trace.append(dict(round=i, variable=c, words_used=words_used[c],
                              s_hat=sd[c], floored=bool(sd[c] < cfg.residual_sd_floor),
                              words_next=new[c]))
        for c in continuous:
            decl[c].words = new[c]
        if progress:
            print(f"  calibrate {i}: N_f = {[new[c] for c in continuous]}  "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    if from_run is not None:
        data = from_run.data
        sd = estimate(data, codes, decl, cfg, from_run.phi_mean,
                      from_run.alpha_draws.mean(axis=0), burn, draws, seed, workers)
        record_iteration(0, sd, data.words)
    for i in range(1, rounds + 1):
        data = encode(codes, decl, cfg)
        state = run_chain(data, cfg, chain, np.random.default_rng(seed + i))
        phi = mean_phi(state.counts_topic_word, data)
        sd = estimate(data, codes, decl, cfg, phi, state.alpha, burn, draws, seed + i,
                      workers)
        record_iteration(i, sd, data.words)

    by_iteration = {}
    for row in trace:
        by_iteration.setdefault(row["round"], {})[row["variable"]] = row["words_next"]
    its = sorted(by_iteration)
    totals = [sum(by_iteration[i].values()) for i in its]
    last = by_iteration[its[-1]]
    prev = by_iteration[its[-2]] if len(its) > 1 else None
    jitter = max(abs(last[c] - prev[c]) for c in continuous) if prev else None
    words = {c: decl[c].words for c in continuous}

    out = output_dir(out)
    decl.preprocessed = relative(inputs.data_path, out)     # it must still find its data
    decl.save(out / "declaration.yaml")
    write_table(pd.DataFrame(trace), out / "trace.tsv")
    record = header("calibrate")
    record.update(
        inputs=inputs.describe(out), model=as_dict(cfg),
        settings=dict(rounds=rounds, iters_per_chain=iters, burn=burn,
                      draws=draws, seed=seed),
        from_run=relative(from_run.path, out) if from_run is not None else None,
        words=words, jitter=jitter, settled=jitter is not None and jitter <= 1,
        total_words_last_rounds=totals[-3:],
        floored=[r["variable"] for r in trace[-len(continuous):] if r["floored"]],
        tables={"declaration.yaml": "the input declaration with the calibrated words",
                "trace.tsv": "per round and variable: words used, s_hat, words next"})
    write_summary(out, record)
    if progress:
        print(f"calibrate: wrote {out}; N_f = {words}"
              f"{'' if record['settled'] else '  (not settled)'}", file=sys.stderr)
    return str(out)
