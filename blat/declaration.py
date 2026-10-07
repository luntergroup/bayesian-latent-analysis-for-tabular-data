"""The variable declaration: which columns are modelled, and how.

The declaration is the .yaml file produced by `convert` and subsequently edited by
the user.  Every later step reads this file.  It contains:

  * each modelled variable's kind (categorical, ordinal, continuous);
  * the level set and ordering of every discrete variable;
  * the direction of every ordinal (randomly assigned; must be set by the user);
  * the transform of every continuous variable, and its fitted parameters;
  * optionally the latent word count N_f or residual sd s_f per continuous variable,
    the Dirichlet concentration per variable, and the model and run settings.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

FORMAT = "blat-declaration/1"
KINDS = ("categorical", "ordinal", "continuous")

#: Defaults of a continuous transform.  `log` is decided from the data when "auto": log if
#: every value is positive and the skew exceeds `skew_threshold`.
TRANSFORM_DEFAULTS = {"log": "auto", "skew_threshold": 1.0,
                      "winsor": [0.005, 0.995], "range": [0.005, 0.995]}


@dataclass
class Variable:
    name: str
    kind: str
    levels: tuple[str, ...] = ()          # discrete only, code order
    confirmed: bool = True                # ordinal only: direction checked by a person
    transform: dict = field(default_factory=dict)   # continuous only
    words: int | None = None              # continuous: N_f, overrides residual_sd
    residual_sd: float | None = None      # continuous: s_f, sets N_f
    beta: float | None = None             # per-word Dirichlet concentration
    units: str | None = None              # for display only
    note: str | None = None

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def to_dict(self) -> dict:
        out = {"name": self.name, "type": self.kind}
        if self.kind != "continuous":
            out["levels"] = list(self.levels)
        if self.kind == "ordinal":
            out["confirmed"] = self.confirmed
        if self.kind == "continuous":
            # only what differs from the defaults (which the header comment lists)
            changed = {k: v for k, v in self.transform.items()
                       if TRANSFORM_DEFAULTS.get(k, object()) != v}
            if changed:
                out["transform"] = changed
            if self.words is not None:
                out["words"] = self.words
            if self.residual_sd is not None:
                out["residual_sd"] = self.residual_sd
        for key in ("beta", "units", "note"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Variable":
        kind = d.get("type")
        if kind not in KINDS:
            raise ValueError(f"variable {d.get('name')!r}: type must be one of {KINDS}, "
                             f"got {kind!r}")
        transform = {}
        if kind == "continuous":
            transform = {**copy.deepcopy(TRANSFORM_DEFAULTS), **d.get("transform", {})}
        levels = d.get("levels") or ()
        odd = [v for v in levels if not isinstance(v, str)]
        if odd:
            raise ValueError(f"variable {d.get('name')!r}: levels {odd} are not text; quote "
                             "labels that look like numbers or booleans, e.g. '1' or 'no'")
        return cls(name=str(d["name"]), kind=kind, levels=tuple(levels),
                   confirmed=bool(d.get("confirmed", True)),
                   transform=transform,
                   words=None if d.get("words") is None else int(d["words"]),
                   residual_sd=(None if d.get("residual_sd") is None
                                else float(d["residual_sd"])),
                   beta=None if d.get("beta") is None else float(d["beta"]),
                   units=d.get("units"), note=d.get("note"))


@dataclass
class Declaration:
    variables: list[Variable]
    id_column: str | None = None
    missing: tuple[str, ...] = ("", "NA", "NaN", "nan")
    separator: str | None = None
    mask_groups: list[list[str]] = field(default_factory=list)
    excluded: dict = field(default_factory=dict)       # column -> reason
    model: dict = field(default_factory=dict)
    run: dict = field(default_factory=dict)
    source: str | None = None
    preprocessed: str | None = None
    report: dict = field(default_factory=dict)

    # -- access ---------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    def __getitem__(self, name: str) -> Variable:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    def of_kind(self, kind: str) -> list[Variable]:
        return [v for v in self.variables if v.kind == kind]

    @property
    def fitted(self) -> bool:
        """True once every continuous transform has its parameters."""
        return all("fitted" in v.transform for v in self.of_kind("continuous"))

    # -- checks ---------------------------------------------------------------

    def validate(self, require_confirmed: bool = False) -> None:
        names = self.names
        if not names:
            raise ValueError("the declaration models no variables")
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate variable names: {sorted(duplicates)}")
        if self.id_column is not None and self.id_column in names:
            raise ValueError(f"the ID column {self.id_column!r} is also declared a variable")
        for v in self.variables:
            if v.kind != "continuous":
                if len(v.levels) < 2:
                    raise ValueError(f"{v.name!r} needs at least two levels")
                if len(set(v.levels)) != len(v.levels):
                    raise ValueError(f"{v.name!r} has duplicate levels")
            if v.words is not None and v.words < 1:
                raise ValueError(f"{v.name!r}: words must be >= 1")
            if v.residual_sd is not None and not v.residual_sd > 0:
                raise ValueError(f"{v.name!r}: residual_sd must be positive")
            if v.beta is not None and not v.beta > 0:
                raise ValueError(f"{v.name!r}: beta must be positive")
        if require_confirmed:
            pending = [v.name for v in self.of_kind("ordinal") if not v.confirmed]
            if pending:
                raise ValueError(
                    "the level order of these ordinal variables has not been confirmed: "
                    f"{pending}.  Check that each 'levels' list runs from lowest to highest "
                    "(or reorder it), then set confirmed: true.")
        for group in self.mask_groups:
            unknown = set(group) - set(names)
            if unknown:
                raise ValueError(f"mask group {group} names undeclared variables {unknown}")

    # -- io -------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "source": self.source,
            "preprocessed": self.preprocessed,
            "id_column": self.id_column,
            "missing": list(self.missing),
            "separator": self.separator,
            "variables": [v.to_dict() for v in self.variables],
            "mask_groups": self.mask_groups,
            "excluded": self.excluded,
            "model": self.model,
            "run": self.run,
            "report": self.report,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Declaration":
        if d.get("format") not in (None, FORMAT):
            raise ValueError(f"not a BLAT declaration (format {d.get('format')!r})")
        decl = cls(
            variables=[Variable.from_dict(v) for v in d.get("variables", [])],
            id_column=d.get("id_column"),
            missing=tuple(d.get("missing", cls.missing)),
            separator=d.get("separator"),
            mask_groups=[list(g) for g in d.get("mask_groups", [])],
            excluded=dict(d.get("excluded", {})),
            model=dict(d.get("model", {})),
            run=dict(d.get("run", {})),
            source=d.get("source"),
            preprocessed=d.get("preprocessed"),
            report=dict(d.get("report", {})),
        )
        decl.validate()
        return decl

    def save(self, path: str | Path) -> None:
        from .formats import dump_yaml

        dump_yaml(self.to_dict(), path, header=HEADER)

    @classmethod
    def load(cls, path: str | Path) -> "Declaration":
        from .formats import load_yaml

        return cls.from_dict(load_yaml(path) or {})

    def copy(self) -> "Declaration":
        return Declaration.from_dict(copy.deepcopy(self.to_dict()))


HEADER = """BLAT variable declaration.  Edit freely; see README.md.
  type:       categorical, ordinal or continuous
  levels:     discrete variables, in code order; for an ordinal, lowest first.
              Quote labels that look like numbers ('01', '1.0').
  confirmed:  ordinal variables: set to true once the level order is checked
  transform:  continuous variables; defaults {log: auto, skew_threshold: 1.0,
              winsor: [0.005, 0.995], range: [0.005, 0.995]} -- log if positive and
              skewed, winsorize at those quantiles, map linearly onto the range.
              'fitted' holds the parameters convert fitted -- delete it to refit
  words, residual_sd, beta: optional per-variable model settings
  model, run: settings for every step (the command line overrides them)"""
