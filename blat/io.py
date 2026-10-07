"""Reading inputs and writing step outputs.

Every step that produces results writes a directory:

    summary.yaml    settings, headline numbers, diagnostics, warnings
    *.tsv           tables (one row per variable, type, record, ...)
    arrays.npz      large arrays (traces, posterior draws), where there are any

Paths inside summary.yaml are relative to the directory, so it can be moved as a whole.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .convert import read_mask, read_preprocessed
from .declaration import Declaration
from .formats import dump_yaml, load_yaml

SUMMARY = "summary.yaml"
ARRAYS = "arrays.npz"


def relative(path: str | Path, start: str | Path) -> str:
    """`path` relative to the directory `start`, or absolute if that is simpler
    (different drive, or more than two levels up)."""
    path = Path(path).resolve()
    try:
        rel = os.path.relpath(path, Path(start).resolve())
    except ValueError:
        return str(path)
    return str(path) if rel.startswith(os.path.join("..", "..", "..")) else rel


def locate(stored: str, directory: str | Path) -> Path:
    """Resolve a path stored in a summary written in `directory`."""
    p = Path(stored)
    return p if p.is_absolute() else Path(directory).resolve() / p


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def header(step: str) -> dict:
    return {"format": f"blat-{step}/1", "version": __version__,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "command": " ".join(sys.argv)}


def output_dir(out: str | Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_summary(out: Path, record: dict, arrays: dict | None = None) -> Path:
    if arrays:
        np.savez_compressed(out / ARRAYS, **arrays)
        record["arrays"] = ARRAYS
    dump_yaml(record, out / SUMMARY)
    return out


def result_dir(path: str | Path) -> Path:
    """A step's output directory, given it or its summary.yaml."""
    path = Path(path)
    return path.parent if path.name == SUMMARY else path


def read_summary(path: str | Path) -> dict:
    return load_yaml(result_dir(path) / SUMMARY)


def read_arrays(path: str | Path) -> dict:
    file = result_dir(path) / ARRAYS
    if not file.exists():
        return {}
    with np.load(file, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class Inputs:
    """A declaration, its preprocessed table, and an optional mask, loaded together."""

    def __init__(self, declaration: str | Path, data: str | Path | None = None,
                 mask: str | Path | None = None, fold: int | None = None):
        self.declaration_path = Path(declaration)
        self.decl = Declaration.load(declaration)
        self.decl.validate(require_confirmed=True)
        if not self.decl.fitted:
            raise ValueError(f"{declaration} has unfitted transforms: pass it through "
                             "`blat convert` first")
        if data is None:
            if not self.decl.preprocessed:
                raise ValueError("no data given and the declaration names no preprocessed "
                                 "table")
            data = locate(self.decl.preprocessed, Path(declaration).parent)
        self.data_path = Path(data)
        self.frame = read_preprocessed(data, self.decl)
        self.codes = self.frame[self.decl.names].to_numpy(dtype=float)
        self.mask_path = Path(mask) if mask else None
        self.fold = fold
        if mask is not None:
            hidden = read_mask(mask, self.decl, self.frame.index, fold)
            self.hidden = hidden & np.isfinite(self.codes)
        else:
            if fold is not None:
                raise ValueError("--fold needs --mask")
            self.hidden = np.zeros(self.codes.shape, dtype=bool)

    def describe(self, directory: Path) -> dict:
        out = {"declaration": relative(self.declaration_path, directory),
               "data": relative(self.data_path, directory),
               "data_digest": digest(self.data_path),
               "records": int(self.codes.shape[0])}
        if self.mask_path is not None:
            out.update(mask=relative(self.mask_path, directory), fold=self.fold,
                       hidden_cells=int(self.hidden.sum()))
        return out


def inputs_of(directory: str | Path, record: dict | None = None,
              with_mask: bool = True) -> Inputs:
    """The inputs a step was run on, re-read from the paths its summary stores."""
    directory = result_dir(directory)
    record = record or read_summary(directory)
    i = record["inputs"]
    mask = locate(i["mask"], directory) if (with_mask and i.get("mask")) else None
    return Inputs(locate(i["declaration"], directory), locate(i["data"], directory),
                  mask, i.get("fold") if mask is not None else None)
