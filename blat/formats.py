"""File formats: YAML for settings and summaries, TSV for tables, NPZ for large arrays.

Floats are written to 4 significant digits, except where a later calculation depends on
more (the score sums, which folds are added over, and the preprocessed data).

YAML is read with YAML 1.2 booleans -- only true/false -- so that level labels such as
`no` and `yes` stay strings even when written unquoted.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SIGNIFICANT = 4


def plain(obj):
    """numpy values, tuples and paths -> plain Python, recursively."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def round_float(value: float, digits: int = SIGNIFICANT) -> str:
    """A float as YAML text with `digits` significant digits (always with a '.')."""
    if math.isnan(value):
        return ".nan"
    if math.isinf(value):
        return ".inf" if value > 0 else "-.inf"
    text = f"{value:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        if "." not in mantissa:
            mantissa += ".0"
        return f"{mantissa}e{exponent}"
    return text if "." in text else text + ".0"


class _Dumper(yaml.SafeDumper):
    pass


def _float(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:float", round_float(value))


_Dumper.add_representer(float, _float)


def _short(value) -> bool:
    return isinstance(value, (int, float, bool)) or value is None or \
        (isinstance(value, str) and len(value) <= 24)


def _list(dumper, value):
    flow = all(_short(v) for v in value)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=flow)


def _flat_list(value) -> bool:
    return isinstance(value, list) and all(_short(v) for v in value)


def _dict(dumper, value):
    """One line if every value is a short scalar or a flat list, and the line is short."""
    flow = (all(_short(v) or _flat_list(v) for v in value.values())
            and (len(value) <= 12 or all(_short(v) for v in value.values()))
            and sum(len(str(k)) + len(str(v)) + 4 for k, v in value.items()) <= 90)
    return dumper.represent_mapping("tag:yaml.org,2002:map", value.items(),
                                    flow_style=flow)


_Dumper.add_representer(list, _list)
_Dumper.add_representer(dict, _dict)


class _Loader(yaml.SafeLoader):
    pass


# YAML 1.2 booleans: drop the 1.1 resolver (yes/no/on/off) and add true/false only.
_Loader.yaml_implicit_resolvers = {
    key: [(tag, regex) for tag, regex in resolvers if tag != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()}
_Loader.add_implicit_resolver("tag:yaml.org,2002:bool",
                              re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
                              list("tTfF"))


def dump_yaml(obj, path: str | Path, header: str | None = None) -> None:
    """Write YAML: short collections of numbers or short labels on one line, everything
    else one item per line; floats to 4 significant digits."""
    text = yaml.dump(plain(obj), Dumper=_Dumper, sort_keys=False, width=100,
                     allow_unicode=True)
    if header:
        text = "".join(f"# {line}\n" if line else "#\n"
                       for line in header.splitlines()) + "\n" + text
    Path(path).write_text(text)


def load_yaml(path: str | Path):
    return yaml.load(Path(path).read_text(), Loader=_Loader)


def write_table(frame: pd.DataFrame, path: str | Path, exact=(), index: bool = False) -> None:
    """TSV; floats to 4 significant digits except the columns named in `exact`."""
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            fmt = "{:.10g}" if column in exact else "{:.%dg}" % SIGNIFICANT
            out[column] = [fmt.format(v) if np.isfinite(v) else "" for v in out[column]]
    out.to_csv(path, sep="\t", index=index)
