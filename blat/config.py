"""Model and run settings, and how they are resolved.

Parameters set in the declaration's "model" and "run" sections override the defaults
set in the file, while command-line parameters override both.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ModelConfig:
    """Everything that defines the target distribution."""

    K: int
    alpha_init: float = 0.1           # initial value; alpha is sampled after the frozen phase
    kappa: float = 1.0                # alpha_k ~ Exp(kappa)
    gscp_n0: float = 0.0              # n0 = 0 makes the GSCP prior exactly Exp(kappa)
    beta: float = 0.5                 # per-word Dirichlet concentration, every variable
    gamma: tuple[float, float] = (0.5, 0.5)   # (gamma_0, gamma_1); uniform on the y scale
    residual_sd: float = 0.2          # s_f, where a variable declares neither words nor s_f
    residual_sd_floor: float = 0.05   # lower bound on an estimated s_f

    def __post_init__(self):
        if self.K is None or int(self.K) < 1:
            raise ValueError("the number of types K must be given (there is no default) "
                             "and must be at least 1")
        object.__setattr__(self, "K", int(self.K))
        object.__setattr__(self, "gamma", tuple(float(g) for g in self.gamma))
        if len(self.gamma) != 2 or min(self.gamma) <= 0:
            raise ValueError("gamma must be two positive numbers")
        for name in ("alpha_init", "kappa", "beta", "residual_sd", "residual_sd_floor"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")

    @property
    def gamma_sum(self) -> float:
        return self.gamma[0] + self.gamma[1]


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines one chain but not the target."""

    seed: int = 0
    alpha_frozen: int = 2000          # iterations with alpha held at alpha_init
    burnin: int = 4000                # total burn-in iterations, alpha_frozen included
    iters: int = 30000                # sampling iterations after burn-in
    thin: int = 10                    # keep every thin-th sampling iteration
    gscp_iters_burnin: int = 20       # GSCP iterations per alpha update during burn-in
    gscp_iters_sampling: int = 2      # ... and after it

    def __post_init__(self):
        if not 0 <= self.alpha_frozen <= self.burnin:
            raise ValueError("need 0 <= alpha_frozen <= burnin")
        if self.iters < 1 or self.thin < 1:
            raise ValueError("iters and thin must be at least 1")
        if min(self.gscp_iters_burnin, self.gscp_iters_sampling) < 1:
            raise ValueError("GSCP iteration counts must be at least 1")

    @property
    def n_draws(self) -> int:
        return self.iters // self.thin

    @property
    def total(self) -> int:
        return self.burnin + self.iters


def resolve(cls, *layers: dict):
    """Build `cls` from dicts of settings, later layers overriding earlier ones.

    Keys that `cls` does not know are rejected, so a misspelt setting in a declaration
    fails loudly instead of being ignored.  `None` values do not override.
    """
    known = {f.name for f in fields(cls)}
    merged: dict = {}
    for layer in layers:
        for key, value in (layer or {}).items():
            if key not in known:
                raise ValueError(f"unknown {cls.__name__} setting {key!r}; "
                                 f"known: {sorted(known)}")
            if value is not None:
                merged[key] = value
    if cls is ModelConfig and "K" not in merged:
        raise ValueError("the number of types K must be given (--k, or \"K\" in the "
                         "declaration's \"model\" section); there is no default")
    return cls(**merged)


def as_dict(cfg) -> dict:
    out = asdict(cfg)
    if "gamma" in out:
        out["gamma"] = list(out["gamma"])
    return out


def words_from_sd(residual_sd: float, gamma_sum: float) -> int:
    """N_f = max(1, floor(2 / (pi^2 s_f^2) - gamma - 1)).  Paper Sec. 3.3.

    With latent words drawn from a Beta(n + gamma), var(y) ~ 2 / (pi^2 (N + gamma + 1))
    on the arccos scale, so this is the number of words whose spread matches s_f.
    """
    if not residual_sd > 0:
        raise ValueError("residual_sd must be positive")
    return max(1, math.floor(2.0 / (math.pi ** 2 * residual_sd ** 2) - gamma_sum - 1.0))


def sd_from_words(words: int, gamma_sum: float) -> float:
    """The residual sd that N_f words represent; the inverse of `words_from_sd`."""
    return math.sqrt(2.0 / (math.pi ** 2 * (words + gamma_sum + 1.0)))
