"""A finished run, re-opened: its settings, inputs, arrays and posterior draws."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

from .config import ModelConfig, RunConfig
from .encoding import encode
from .io import inputs_of, read_arrays, read_summary, result_dir
from .quantities import draw_phi, draw_theta, mean_phi


class Run:
    def __init__(self, path: str | Path):
        self.path = result_dir(path)
        self.record = read_summary(self.path)
        if self.record.get("format") != "blat-run/1":
            raise ValueError(f"{self.path} is not the output of `blat run`")
        self.cfg = ModelConfig(**{**self.record["model"],
                                  "gamma": tuple(self.record["model"]["gamma"])})
        self.run_cfg = RunConfig(**self.record["run"])

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def K(self) -> int:
        return self.cfg.K

    @cached_property
    def arrays(self) -> dict:
        return read_arrays(self.path)

    @cached_property
    def inputs(self):
        """The declaration, table and mask the run was fitted on."""
        return inputs_of(self.path, self.record)

    @property
    def decl(self):
        return self.inputs.decl

    @cached_property
    def data(self):
        """The encoded data exactly as the chain saw it (hidden cells left out)."""
        return encode(self.inputs.codes, self.decl, self.cfg, hidden=self.inputs.hidden)

    @property
    def phi_draws(self) -> np.ndarray:
        return self.arrays["draw_phi"].astype(float)

    @property
    def alpha_draws(self) -> np.ndarray:
        return self.arrays["draw_alpha"]

    @property
    def theta_mean(self) -> np.ndarray:
        return self.arrays["theta_mean"]

    @property
    def phi_mean(self) -> np.ndarray:
        return self.arrays["phi_mean"]

    def spread_draws(self, n: int) -> np.ndarray:
        """Indices of `n` stored draws spread evenly over the sampling phase."""
        total = self.phi_draws.shape[0]
        return np.unique(np.linspace(0, total - 1, min(n, total)).round().astype(int))

    def last_draw(self, rng: np.random.Generator, use_mean: bool = False):
        """(alpha, theta, phi) from the last stored draw's counts.

        theta_d ~ Dir(alpha + n_d.), phi_kf ~ Dir(beta_f + n_kf.): a posterior draw, since
        the counts come from a posterior sample of z.  With `use_mean` the conditional means
        are returned instead -- not a draw, so a chain started from them needs a burn-in.
        """
        a = self.arrays
        alpha = a["last_alpha"]
        ndk, nkw = a["last_counts_doc_topic"], a["last_counts_topic_word"]
        if use_mean:
            theta = (alpha + ndk) / (alpha.sum() + ndk.sum(axis=1, keepdims=True))
            return alpha, theta, mean_phi(nkw, self.data)
        return alpha, draw_theta(ndk, alpha, rng), draw_phi(nkw, self.data, rng)
