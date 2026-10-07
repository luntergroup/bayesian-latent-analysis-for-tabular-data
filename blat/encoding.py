"""The preprocessed table as seen by the sampler: flat token arrays.

Sub-variables f' are numbered in one sequence -- categorical, then ordinal sub-features,
then continuous, each in declaration order:

    [ categorical ] [ ordinal sub-features ] [ continuous ]
      one per         C-1 per ordinal          one per
      variable        variable                 variable

Word indices are global: sub-variable f corresponds to metavocabulary columns
vocab_offset[f] : vocab_offset[f+1], so phi is a single dense (K, V) array (paper Sec. 3.4).

Tokens:
  discrete    one per observed categorical cell and per observed ordinal sub-feature; the
              word is observed.  An ordinal at level c emits "continue" (word 0) on
              sub-features 0..c-1 and "stop" (word 1) on sub-feature c if c < C-1; the
              sub-features above are unobserved (paper Sec. 3.2).
  continuous  N_f latent tokens per observed cell; the binary word is latent and resampled
              jointly with the type.  The model's Beta variate is x = (1 - cos(pi y)) / 2.

Missing and hidden cells emit no tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig, words_from_sd
from .declaration import Declaration

IDX = np.int32
COUNT = np.int32


@dataclass(frozen=True)
class Layout:
    """Where every variable sits in the metavocabulary."""

    names: tuple[str, ...]               # sub-variable names
    vocab_size: np.ndarray               # (F',)
    vocab_offset: np.ndarray             # (F'+1,)
    is_continuous: np.ndarray            # (F',) bool
    subs: dict                           # variable -> tuple of its sub-variables
    kinds: dict                          # variable -> kind
    columns: tuple[str, ...]             # variables, in declaration order

    @property
    def n_sub(self) -> int:
        return len(self.names)

    @property
    def n_vocab(self) -> int:
        return int(self.vocab_offset[-1])

    @property
    def first_continuous(self) -> int:
        return self.n_sub - int(self.is_continuous.sum())

    def base(self, sub: int) -> int:
        return int(self.vocab_offset[sub])

    def continuous_columns(self) -> list[str]:
        return [c for c in self.columns if self.kinds[c] == "continuous"]


def build_layout(decl: Declaration) -> Layout:
    names, sizes, subs, kinds = [], [], {}, {}
    for kind in ("categorical", "ordinal", "continuous"):
        for v in decl.of_kind(kind):
            kinds[v.name] = kind
            if kind == "categorical":
                subs[v.name] = (len(names),)
                names.append(v.name)
                sizes.append(v.n_levels)
            elif kind == "ordinal":
                idx = []
                for c in range(v.n_levels - 1):
                    idx.append(len(names))
                    names.append(f"{v.name}[{c}]")
                    sizes.append(2)
                subs[v.name] = tuple(idx)
            else:
                subs[v.name] = (len(names),)
                names.append(v.name)
                sizes.append(2)
    vocab_size = np.asarray(sizes, dtype=IDX)
    offset = np.concatenate(([0], np.cumsum(vocab_size))).astype(IDX)
    is_cont = np.zeros(len(names), dtype=bool)
    for v in decl.of_kind("continuous"):
        is_cont[subs[v.name][0]] = True
    return Layout(tuple(names), vocab_size, offset, is_cont, subs, kinds,
                  tuple(decl.names))


def words_per_variable(decl: Declaration, cfg: ModelConfig) -> dict[str, int]:
    """N_f for every continuous variable: declared words, else from s_f, else the default."""
    out = {}
    for v in decl.of_kind("continuous"):
        if v.words is not None:
            out[v.name] = int(v.words)
        else:
            out[v.name] = words_from_sd(v.residual_sd or cfg.residual_sd, cfg.gamma_sum)
    return out


def beta_per_sub(decl: Declaration, cfg: ModelConfig, layout: Layout) -> np.ndarray:
    out = np.full(layout.n_sub, float(cfg.beta))
    for v in decl.variables:
        if v.beta is not None:
            out[list(layout.subs[v.name])] = v.beta
    return out


@dataclass(frozen=True)
class EncodedData:
    """Flat, contiguous arrays; see the module docstring for the layout."""

    layout: Layout
    n_records: int
    beta_per_sub: np.ndarray        # (F',)
    words: dict                     # continuous variable -> N_f

    disc_doc: np.ndarray            # (W_d,) IDX
    disc_sub: np.ndarray
    disc_word: np.ndarray           # global metavocabulary index

    cont_doc: np.ndarray            # (W_c,) IDX, N_f consecutive tokens per cell
    cont_sub: np.ndarray
    cont_cell: np.ndarray

    cell_doc: np.ndarray            # (n_cells,) IDX
    cell_sub: np.ndarray
    cell_n: np.ndarray              # N_f of each cell
    cell_y: np.ndarray              # the preprocessed value
    cell_x: np.ndarray              # the Beta variate
    cell_log_x: np.ndarray
    cell_log_1mx: np.ndarray

    @property
    def vocab_offset(self) -> np.ndarray:
        return self.layout.vocab_offset

    @property
    def vocab_size(self) -> np.ndarray:
        return self.layout.vocab_size

    @property
    def n_sub(self) -> int:
        return self.layout.n_sub

    @property
    def n_vocab(self) -> int:
        return self.layout.n_vocab

    @property
    def first_continuous(self) -> int:
        return self.layout.first_continuous

    @property
    def n_continuous_sub(self) -> int:
        return int(self.layout.is_continuous.sum())

    @property
    def n_discrete_tokens(self) -> int:
        return int(self.disc_doc.size)

    @property
    def n_continuous_tokens(self) -> int:
        return int(self.cont_doc.size)

    @property
    def n_tokens(self) -> int:
        return self.n_discrete_tokens + self.n_continuous_tokens

    @property
    def n_cells(self) -> int:
        return int(self.cell_doc.size)

    @property
    def max_words(self) -> int:
        return int(self.cell_n.max()) if self.n_cells else 1

    def validate(self) -> None:
        lay = self.layout
        assert np.all(lay.vocab_size > 0)
        if self.n_discrete_tokens:
            assert not lay.is_continuous[self.disc_sub].any()
            lo = lay.vocab_offset[self.disc_sub]
            hi = lay.vocab_offset[self.disc_sub + 1]
            assert np.all((self.disc_word >= lo) & (self.disc_word < hi))
        if self.n_cells:
            assert lay.is_continuous[self.cell_sub].all()
            assert np.all(self.cell_n >= 1)
            assert self.n_continuous_tokens == int(self.cell_n.sum())
            assert np.array_equal(self.cont_cell,
                                  np.repeat(np.arange(self.n_cells, dtype=IDX), self.cell_n))
            if not np.all((self.cell_x > 0) & (self.cell_x < 1)):
                raise ValueError("a continuous value lies on the boundary of [0, 1]; the "
                                 "preprocessed values must lie strictly inside")


def encode(codes: np.ndarray, decl: Declaration, cfg: ModelConfig,
           hidden: np.ndarray | None = None) -> EncodedData:
    """(records, variables) values in declaration order -> EncodedData.

    `codes` holds level codes for discrete variables and y for continuous ones, NaN where
    missing.  `hidden` marks cells to treat as missing (a mask); they emit no tokens.
    """
    layout = build_layout(decl)
    codes = np.asarray(codes, dtype=float)
    n_records, n_vars = codes.shape
    if n_vars != len(decl.names):
        raise ValueError(f"{n_vars} columns for {len(decl.names)} declared variables")
    usable = np.isfinite(codes)
    if hidden is not None:
        if hidden.shape != codes.shape:
            raise ValueError(f"mask shape {hidden.shape} does not match data {codes.shape}")
        usable &= ~hidden
    col = {name: j for j, name in enumerate(decl.names)}
    words = words_per_variable(decl, cfg)

    doc, sub, word = [], [], []
    for v in decl.of_kind("categorical"):
        j, f = col[v.name], layout.subs[v.name][0]
        rows = np.flatnonzero(usable[:, j])
        c = codes[rows, j].astype(np.int64)
        if np.any((c < 0) | (c >= v.n_levels)):
            raise ValueError(f"{v.name!r} has codes outside 0..{v.n_levels - 1}")
        doc.append(rows)
        sub.append(np.full(rows.size, f))
        word.append(layout.base(f) + c)
    for v in decl.of_kind("ordinal"):
        j = col[v.name]
        rows = np.flatnonzero(usable[:, j])
        levels = codes[rows, j].astype(np.int64)
        if np.any((levels < 0) | (levels >= v.n_levels)):
            raise ValueError(f"{v.name!r} has codes outside 0..{v.n_levels - 1}")
        for c, f in enumerate(layout.subs[v.name]):
            emits = levels >= c
            doc.append(rows[emits])
            sub.append(np.full(int(emits.sum()), f))
            word.append(layout.base(f) + (levels[emits] == c))

    def cat(parts):
        return np.concatenate(parts).astype(IDX) if parts else np.empty(0, IDX)

    doc_a, sub_a, word_a = cat(doc), cat(sub), cat(word)
    order = np.lexsort((sub_a, doc_a))                 # record-major, for locality
    doc_a, sub_a, word_a = (np.ascontiguousarray(a[order]) for a in (doc_a, sub_a, word_a))

    cdoc, csub, cy, cn = [], [], [], []
    for v in decl.of_kind("continuous"):
        j, f = col[v.name], layout.subs[v.name][0]
        rows = np.flatnonzero(usable[:, j])
        cdoc.append(rows)
        csub.append(np.full(rows.size, f))
        cy.append(codes[rows, j])
        cn.append(np.full(rows.size, words[v.name]))
    cell_doc, cell_sub, cell_n = cat(cdoc), cat(csub), cat(cn)
    y = np.concatenate(cy) if cy else np.empty(0)
    order = np.lexsort((cell_sub, cell_doc))
    cell_doc, cell_sub, cell_n = (np.ascontiguousarray(a[order])
                                  for a in (cell_doc, cell_sub, cell_n))
    y = np.ascontiguousarray(y[order], dtype=float)
    x = 0.5 - 0.5 * np.cos(np.pi * y)
    cont_cell = np.repeat(np.arange(cell_doc.size, dtype=IDX), cell_n)

    data = EncodedData(
        layout=layout, n_records=n_records,
        beta_per_sub=beta_per_sub(decl, cfg, layout), words=words,
        disc_doc=doc_a, disc_sub=sub_a, disc_word=word_a,
        cont_doc=np.ascontiguousarray(cell_doc[cont_cell]),
        cont_sub=np.ascontiguousarray(cell_sub[cont_cell]),
        cont_cell=cont_cell,
        cell_doc=cell_doc, cell_sub=cell_sub, cell_n=cell_n, cell_y=y, cell_x=x,
        cell_log_x=np.log(x), cell_log_1mx=np.log1p(-x))
    data.validate()
    return data


def make_counts(data: EncodedData, K: int) -> dict[str, np.ndarray]:
    return {
        "counts_doc_topic": np.zeros((data.n_records, K), dtype=COUNT),
        "counts_topic_word": np.zeros((K, data.n_vocab), dtype=COUNT),
        "counts_topic_feature": np.zeros((K, data.n_sub), dtype=COUNT),
        "counts_doc": np.zeros(data.n_records, dtype=COUNT),
        "counts_doc_word": np.zeros((data.n_records, 2 * data.n_continuous_sub),
                                    dtype=COUNT),
    }


def normalise_blocks(weights: np.ndarray, data_or_layout) -> np.ndarray:
    """Normalise each sub-variable's block of the last axis to sum to one."""
    lay = getattr(data_or_layout, "layout", data_or_layout)
    starts = lay.vocab_offset[:-1].astype(np.intp)
    sums = np.add.reduceat(weights, starts, axis=-1)
    return weights / np.repeat(sums, lay.vocab_size, axis=-1)
