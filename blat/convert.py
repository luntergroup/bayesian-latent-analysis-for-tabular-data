"""Raw table -> declaration and preprocessed table, and masks.

Two passes are needed because the kinds of the variables and the order of ordinal
levels are decisions a user must confirm:

  1. `draft`  infers a declaration from the raw table:
     - numeric columns with few integer values become ordinal candidates (direction
       unconfirmed);
     - other numeric columns become continuous;
     - text columns with few values become categorical;
     - anything else is excluded with a reason.
     Nothing is transformed yet.
  2. `fit` + `apply` take the (edited) declaration, fit the continuous transforms --
     log if positive and skewed, winsorize, map linearly onto [0.005, 0.995] -- and write
     the preprocessed table: discrete cells as their level label, continuous cells as
     their y value (constant variance, range 0-1).

The fitted declaration stores every transform parameter, so the same transform can be
applied to new records later without refitting (`apply` with a fitted declaration).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .declaration import Declaration, Variable

MAX_CATEGORICAL_LEVELS = 50
MAX_ORDINAL_LEVELS = 20
MISSINGNESS_WARNING = 0.5


# --- reading -----------------------------------------------------------------------

def guess_separator(path: str | Path) -> str:
    return "\t" if Path(path).suffix.lower() in (".tsv", ".tab", ".txt") else ","


def read_raw(path: str | Path, separator: str | None = None,
             missing: tuple[str, ...] = ("", "NA", "NaN", "nan")) -> pd.DataFrame:
    """Every column as stripped strings, missing cells as NaN.

    Whitespace-only cells count as missing, as do the declared missing tokens.
    """
    frame = pd.read_csv(path, sep=separator or guess_separator(path), dtype=str,
                        keep_default_na=False, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    tokens = {m.strip() for m in missing}
    stripped = frame.apply(lambda col: col.str.strip())
    return stripped.mask(stripped.isin(tokens) | (stripped == ""))


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# --- pass 1: the draft --------------------------------------------------------------

def draft(raw: pd.DataFrame, id_column: str | None = None,
          max_categorical: int = MAX_CATEGORICAL_LEVELS,
          max_ordinal: int = MAX_ORDINAL_LEVELS) -> Declaration:
    """A declaration inferred from the table, for a person to check and edit."""
    if id_column is not None and id_column not in raw.columns:
        raise ValueError(f"no ID column {id_column!r} in the table")
    variables, excluded = [], {}
    for column in raw.columns:
        if column == id_column:
            continue
        values = raw[column].dropna()
        distinct = values.unique()
        if values.empty:
            excluded[column] = "every cell is missing"
            continue
        if len(distinct) == 1:
            excluded[column] = f"constant ({distinct[0]!r})"
            continue
        numbers = _numeric(values)
        if numbers.notna().all():
            integral = bool(np.all(np.isclose(numbers, np.round(numbers))))
            if len(distinct) == 2:
                levels = sorted(distinct, key=lambda s: float(s))
                variables.append(Variable(column, "categorical", tuple(levels)))
            elif integral and len(distinct) <= max_ordinal:
                levels = sorted(distinct, key=lambda s: float(s))
                variables.append(Variable(
                    column, "ordinal", tuple(levels), confirmed=False,
                    note="few integer values: taken as ordinal, lowest first.  Confirm the "
                         "order, or change the type to categorical or continuous."))
            else:
                variables.append(Variable.from_dict({"name": column, "type": "continuous"}))
        elif numbers.notna().any():
            bad = sorted(values[numbers.isna()].unique())[:5]
            excluded[column] = (f"mostly numeric but with non-numeric values {bad}; "
                                "declare them as missing tokens, or declare the column")
        elif len(distinct) <= max_categorical:
            variables.append(Variable(
                column, "categorical", tuple(sorted(distinct)),
                note="text with few values: taken as categorical.  If the levels are "
                     "ordered, change the type to ordinal and list them lowest first."))
        else:
            excluded[column] = (f"{len(distinct)} distinct text values: an identifier, "
                                "a date or free text?")
    return Declaration(variables=variables, id_column=id_column, excluded=excluded,
                       model={"K": None})


# --- pass 2: fitting and applying the transform --------------------------------------

def fit(raw: pd.DataFrame, decl: Declaration) -> Declaration:
    """Fit the continuous transforms and return a new declaration that stores them.

    Transforms that are already fitted are kept, so fitting is idempotent.
    """
    decl = decl.copy()
    decl.validate(require_confirmed=True)
    _check_columns(raw, decl)
    for v in decl.of_kind("continuous"):
        if "fitted" in v.transform:
            continue
        values = _numeric(raw[v.name]).dropna()
        if values.empty:
            raise ValueError(f"{v.name!r} has no numeric values to fit a transform on")
        t = v.transform
        if t["log"] == "auto":
            use_log = bool(values.min() > 0 and values.skew() > t["skew_threshold"])
        else:
            use_log = bool(t["log"])
            if use_log and values.min() <= 0:
                raise ValueError(f"{v.name!r} is declared log-transformed but has values "
                                 "<= 0")
        scaled = np.log(values) if use_log else values
        lo, hi = (float(scaled.quantile(q)) for q in t["winsor"])
        if not hi > lo:
            raise ValueError(f"{v.name!r}: its winsorization range is empty "
                             f"({lo} to {hi}); is it nearly constant?")
        # stored to 4 significant digits, like every number in the declaration, and
        # rounded before use, so the table written now and new records converted later
        # with --apply go through exactly the same transform
        lo, hi = float(f"{lo:.4g}"), float(f"{hi:.4g}")
        if not hi > lo:
            raise ValueError(f"{v.name!r}: its winsorization range is empty at 4 "
                             "significant digits; set transform.log or rescale the column")
        t["fitted"] = {"log": use_log, "lo": lo, "hi": hi,
                       "skew": float(f"{values.skew():.4g}"),
                       "skew_after": float(f"{scaled.skew():.4g}")}
    return decl


def _check_columns(raw: pd.DataFrame, decl: Declaration) -> None:
    missing = [n for n in decl.names if n not in raw.columns]
    if missing:
        raise ValueError(f"declared variables not in the table: {missing}")
    if decl.id_column is not None and decl.id_column not in raw.columns:
        raise ValueError(f"no ID column {decl.id_column!r} in the table")


def forward(values: np.ndarray, transform: dict) -> np.ndarray:
    """Original units -> y in [range], with the fitted parameters only."""
    p = transform["fitted"]
    r0, r1 = transform["range"]
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.log(np.where(values > 0, values, np.nan)) if p["log"] else values
    clipped = np.clip(scaled, p["lo"], p["hi"])
    return r0 + (r1 - r0) * (clipped - p["lo"]) / (p["hi"] - p["lo"])


def inverse(y: np.ndarray, transform: dict) -> np.ndarray:
    """y -> original units.  Winsorized values come back as the bound they were clipped to."""
    p = transform["fitted"]
    r0, r1 = transform["range"]
    scaled = p["lo"] + (np.asarray(y, dtype=float) - r0) * (p["hi"] - p["lo"]) / (r1 - r0)
    return np.exp(scaled) if p["log"] else scaled


def apply(raw: pd.DataFrame, decl: Declaration) -> pd.DataFrame:
    """The preprocessed table: labels for discrete cells, y for continuous, NaN if missing.

    Uses only what the declaration stores, so it applies to new records unchanged.  A
    discrete value outside the declared levels is an error, not a silent missing value.
    """
    if not decl.fitted:
        raise ValueError("the declaration has unfitted continuous transforms; run fit first")
    _check_columns(raw, decl)
    out = pd.DataFrame(index=_index(raw, decl))
    for v in decl.variables:
        column = raw[v.name].to_numpy()
        if v.kind == "continuous":
            numbers = _numeric(raw[v.name])
            bad = raw[v.name].notna() & numbers.isna()
            if bad.any():
                raise ValueError(f"{v.name!r} has non-numeric values "
                                 f"{sorted(raw.loc[bad, v.name].unique())[:5]}")
            if v.transform["fitted"]["log"] and (numbers <= 0).any():
                raise ValueError(f"{v.name!r} is log-transformed but has values <= 0")
            out[v.name] = forward(numbers.to_numpy(), v.transform)
        else:
            observed = pd.Series(column).dropna()
            unknown = sorted(set(observed) - set(v.levels))
            if unknown:
                raise ValueError(f"{v.name!r} has values outside its declared levels: "
                                 f"{unknown[:10]}; declared {list(v.levels)}")
            out[v.name] = column
    return out


def _index(raw: pd.DataFrame, decl: Declaration) -> pd.Index:
    if decl.id_column is None:
        return pd.RangeIndex(len(raw), name="row")
    ids = raw[decl.id_column]
    if ids.isna().any():
        raise ValueError(f"the ID column {decl.id_column!r} has missing values")
    if ids.duplicated().any():
        raise ValueError(f"the ID column {decl.id_column!r} has duplicate values: "
                         f"{sorted(ids[ids.duplicated()].unique())[:5]}")
    return pd.Index(ids.to_numpy(), name=decl.id_column)


# --- the preprocessed table on disk ---------------------------------------------------

def write_preprocessed(frame: pd.DataFrame, path: str | Path) -> None:
    frame.to_csv(path, sep="\t", na_rep="", float_format="%.10g")


def read_preprocessed(path: str | Path, decl: Declaration) -> pd.DataFrame:
    """The preprocessed table as the model reads it: level codes and y, NaN if missing."""
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, index_col=0)
    return to_codes(frame.replace("", np.nan), decl)


def to_codes(frame: pd.DataFrame, decl: Declaration) -> pd.DataFrame:
    """Labels -> integer level codes (as float, NaN missing); continuous as float."""
    missing = [n for n in decl.names if n not in frame.columns]
    if missing:
        raise ValueError(f"declared variables not in the preprocessed table: {missing}")
    out = pd.DataFrame(index=frame.index)
    for v in decl.variables:
        column = frame[v.name]
        if v.kind == "continuous":
            out[v.name] = pd.to_numeric(column, errors="raise").astype(float)
            continue
        codes = {level: float(i) for i, level in enumerate(v.levels)}
        mapped = column.map(lambda s: codes.get(s, -1.0) if isinstance(s, str) else np.nan)
        if (mapped == -1.0).any():
            unknown = sorted(column[mapped == -1.0].unique())[:10]
            raise ValueError(f"{v.name!r} has values outside its declared levels: {unknown}")
        out[v.name] = mapped.astype(float)
    return out


def from_codes(codes: pd.DataFrame, decl: Declaration) -> pd.DataFrame:
    """The inverse of `to_codes`: level labels for discrete columns."""
    out = codes.copy().astype(object)
    for v in decl.variables:
        if v.kind != "continuous":
            out[v.name] = [v.levels[int(c)] if np.isfinite(c) else np.nan
                           for c in codes[v.name].to_numpy(dtype=float)]
    return out


# --- report -----------------------------------------------------------------------

def report(raw: pd.DataFrame, pre: pd.DataFrame, decl: Declaration
           ) -> tuple[dict, pd.DataFrame]:
    """(overall numbers and warnings, one row of diagnostics per variable)."""
    from .config import words_from_sd

    columns, warnings = {}, []
    n = len(pre)
    gamma_sum = sum(decl.model.get("gamma") or (0.5, 0.5))
    default_sd = decl.model.get("residual_sd") or 0.2
    tokens = {"categorical": 0, "ordinal": 0, "continuous": 0}
    for v in decl.variables:
        observed = pre[v.name].notna()
        entry = {"type": v.kind, "missing": int(n - observed.sum()),
                 "missing_fraction": float(1 - observed.mean()) if n else math.nan}
        if v.kind == "continuous":
            numbers = _numeric(raw[v.name]).dropna()
            f = v.transform["fitted"]
            scaled = np.log(numbers) if f["log"] else numbers
            entry.update(min=float(numbers.min()), max=float(numbers.max()),
                         skew=f["skew"], skew_after=f["skew_after"], log=f["log"],
                         winsor_lo=f["lo"], winsor_hi=f["hi"],
                         clipped_low=int((scaled < f["lo"]).sum()),
                         clipped_high=int((scaled > f["hi"]).sum()),
                         distinct=int(numbers.nunique()))
            words = v.words or words_from_sd(v.residual_sd or default_sd, gamma_sum)
            entry["words"] = words
            tokens["continuous"] += int(observed.sum()) * words
            if numbers.nunique() < 10:
                warnings.append(f"{v.name}: continuous with only {numbers.nunique()} "
                                "distinct values")
        else:
            counts = pre[v.name].value_counts()
            entry["levels"] = {level: int(counts.get(level, 0)) for level in v.levels}
            never = [level for level in v.levels if counts.get(level, 0) == 0]
            if never:
                entry["levels_never_observed"] = never
                warnings.append(f"{v.name}: declared levels never observed: {never}")
            top = counts.max() / observed.sum() if observed.sum() else 0.0
            if top > 0.98:
                warnings.append(f"{v.name}: nearly constant ({top:.1%} in one level)")
            per_cell = 1
            if v.kind == "ordinal":
                codes = pre[v.name].map({l: i for i, l in enumerate(v.levels)}).dropna()
                per_cell = float(np.minimum(codes + 1, v.n_levels - 1).mean()) \
                    if len(codes) else 0.0
            tokens[v.kind] += int(round(observed.sum() * per_cell))
        if entry["missing_fraction"] > MISSINGNESS_WARNING:
            warnings.append(f"{v.name}: {entry['missing_fraction']:.0%} missing")
        columns[v.name] = entry
    empty = int(pre.isna().all(axis=1).sum())
    if empty:
        warnings.append(f"{empty} records have every modelled cell missing")
    pending = [v.name for v in decl.of_kind("ordinal") if not v.confirmed]
    if pending:
        warnings.append(f"ordinal order not confirmed: {pending}")
    total = sum(tokens.values())
    table = pd.DataFrame([{"variable": name, **{k: (";".join(f"{l}={c}" for l, c in v.items())
                                                    if isinstance(v, dict) else
                                                    ";".join(v) if isinstance(v, list) else v)
                                                for k, v in entry.items()}}
                          for name, entry in columns.items()])
    return {"records": n, "variables": len(decl.variables),
            "tokens": tokens,
            "continuous_token_share": tokens["continuous"] / total if total else math.nan,
            "records_all_missing": empty, "warnings": warnings}, table


# --- masks --------------------------------------------------------------------------

def make_mask(codes: pd.DataFrame, decl: Declaration, spec: float,
              rng: np.random.Generator) -> pd.DataFrame:
    """A binary mask (spec a fraction in (0, 1)) or a fold mask (spec an integer >= 2).

    Binary: 1 marks a masked cell; that fraction of the observed cells is chosen at random.
    Fold: every observed cell gets a fold number 1..F in near-equal numbers; missing
    cells get 0.  Cells of one mask group in one record are one unit: they are masked
    together, and share a fold.
    """
    observed = codes[decl.names].notna().to_numpy()
    names = decl.names
    unit = np.arange(observed.size).reshape(observed.shape)      # one unit per cell ...
    for group in decl.mask_groups:                               # ... or per group row
        cols = [names.index(c) for c in group]
        unit[:, cols] = unit[:, [cols[0]]]
    unit_ids = np.unique(unit[observed])
    out = np.zeros(observed.shape, dtype=np.int64)
    flat_unit = unit.reshape(-1)

    if float(spec).is_integer() and spec >= 2:
        n_folds = int(spec)
        fold_of = dict(zip(unit_ids, rng.permutation(unit_ids.size) % n_folds + 1))
        out.reshape(-1)[:] = [fold_of.get(u, 0) for u in flat_unit]
    elif 0 < spec < 1:
        n_mask = int(round(spec * unit_ids.size))
        chosen = set(rng.choice(unit_ids, size=n_mask, replace=False).tolist())
        out.reshape(-1)[:] = [1 if u in chosen else 0 for u in flat_unit]
    else:
        raise ValueError("--make-mask takes a fraction in (0, 1) for a binary mask or an "
                         f"integer >= 2 for a fold mask; got {spec}")
    out[~observed] = 0
    return pd.DataFrame(out, index=codes.index, columns=names)


def write_mask(mask: pd.DataFrame, path: str | Path) -> None:
    mask.to_csv(path, sep="\t")


def read_mask(path: str | Path, decl: Declaration, index: pd.Index,
              fold: int | None = None) -> np.ndarray:
    """Boolean (records, variables) array of the cells to hide.

    Rows are matched on the ID column if the declaration has one, else by position.  A
    fold file (values 0..F) needs `fold`; a binary file (values 0/1) must not have one.
    """
    frame = pd.read_csv(path, sep="\t", index_col=0 if decl.id_column else None)
    if decl.id_column is None and frame.columns[0] in ("row", "Unnamed: 0"):
        frame = frame.drop(columns=frame.columns[0])
    missing = [n for n in decl.names if n not in frame.columns]
    if missing:
        raise ValueError(f"mask {path} lacks columns {missing}")
    frame = frame[decl.names]
    if decl.id_column is not None:
        frame.index = frame.index.astype(str)
        wanted = index.astype(str)
        absent = wanted.difference(frame.index)
        if len(absent):
            raise ValueError(f"mask {path} has no row for IDs {list(absent[:5])}")
        frame = frame.loc[wanted]
    elif len(frame) != len(index):
        raise ValueError(f"mask {path} has {len(frame)} rows, the data {len(index)}")
    values = frame.to_numpy()
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"mask {path} must hold integers")
    top = int(values.max()) if values.size else 0
    if values.min() < 0:
        raise ValueError(f"mask {path} has negative entries")
    if fold is None:
        if top > 1:
            raise ValueError(f"mask {path} is a fold file (values up to {top}); "
                             "give --fold n")
        return values == 1
    if top < 2:
        raise ValueError(f"mask {path} is a binary file; --fold does not apply")
    if not 1 <= fold <= top:
        raise ValueError(f"--fold {fold} is outside 1..{top}")
    return values == fold
