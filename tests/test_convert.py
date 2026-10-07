import numpy as np
import pandas as pd
import pytest

from blat import convert as C
from blat.declaration import Declaration


def raw_table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": [f"r{i}" for i in range(n)],
        "flag": rng.choice(["no", "yes"], n),
        "colour": rng.choice(["red", "green", "blue"], n),
        "score": rng.integers(0, 5, n).astype(str),
        "skewed": [f"{v:.4g}" for v in rng.lognormal(0, 1, n)],
        "normal": [f"{v:.4g}" for v in rng.normal(10, 2, n)],
        "note": [f"text {i}" for i in range(n)],
        "constant": ["k"] * n,
    }).astype(str)


def as_raw(frame):
    return frame.mask(frame == "")


def test_draft_infers_kinds_and_excludes():
    decl = C.draft(as_raw(raw_table()), id_column="id")
    kinds = {v.name: v.kind for v in decl.variables}
    assert kinds == {"flag": "categorical", "colour": "categorical", "score": "ordinal",
                     "skewed": "continuous", "normal": "continuous"}
    assert not decl["score"].confirmed
    assert decl["score"].levels == ("0", "1", "2", "3", "4")
    assert set(decl.excluded) == {"note", "constant"}


def test_fit_requires_confirmed_ordinals():
    raw = as_raw(raw_table())
    decl = C.draft(raw, id_column="id")
    with pytest.raises(ValueError, match="confirmed"):
        C.fit(raw, decl)


def fitted(raw=None):
    raw = as_raw(raw_table()) if raw is None else raw
    decl = C.draft(raw, id_column="id")
    decl["score"].confirmed = True
    return raw, C.fit(raw, decl)


def test_transform_decisions_and_roundtrip():
    raw, decl = fitted()
    assert decl["skewed"].transform["fitted"]["log"] is True
    assert decl["normal"].transform["fitted"]["log"] is False
    pre = C.apply(raw, decl)
    for name in ("skewed", "normal"):
        y = pre[name].to_numpy()
        assert np.nanmin(y) >= 0.005 - 1e-12 and np.nanmax(y) <= 0.995 + 1e-12
        original = pd.to_numeric(raw[name]).to_numpy()
        back = C.inverse(y, decl[name].transform)
        f = decl[name].transform["fitted"]
        scaled = np.log(original) if f["log"] else original
        inside = (scaled > f["lo"]) & (scaled < f["hi"])
        assert np.allclose(back[inside], original[inside], rtol=1e-9)


def test_apply_rejects_unknown_levels():
    raw, decl = fitted()
    raw2 = raw.copy()
    raw2.loc[0, "colour"] = "purple"
    with pytest.raises(ValueError, match="outside its declared levels"):
        C.apply(raw2, decl)


def test_declared_but_unobserved_levels_are_kept(tmp_path):
    raw, decl = fitted()
    decl["colour"].levels = ("red", "green", "blue", "violet")
    pre = C.apply(raw, decl)
    codes = C.to_codes(pre, decl)
    assert set(codes["colour"].dropna().unique()) <= {0.0, 1.0, 2.0}
    summary, table = C.report(raw, pre, decl)
    row = table.set_index("variable").loc["colour"]
    assert row["levels_never_observed"] == "violet"
    assert any("violet" in w for w in summary["warnings"])


def test_preprocessed_roundtrip(tmp_path):
    raw, decl = fitted()
    pre = C.apply(raw, decl)
    path = tmp_path / "t.preproc.tsv"
    C.write_preprocessed(pre, path)
    codes = C.read_preprocessed(path, decl)
    assert codes.shape == (len(raw), len(decl.names))
    assert np.allclose(codes["normal"], pre["normal"], atol=1e-9, equal_nan=True)
    labels = C.from_codes(codes, decl)
    assert (labels["flag"] == pre["flag"]).all()


def test_masks_binary_and_fold(tmp_path):
    raw, decl = fitted()
    raw.loc[:9, "normal"] = np.nan
    decl.mask_groups = [["skewed", "normal"]]
    codes = C.to_codes(C.apply(raw, decl), decl)
    observed = codes.notna().to_numpy()

    folds = C.make_mask(codes, decl, 5, np.random.default_rng(1))
    f = folds.to_numpy()
    assert np.all(f[~observed] == 0) and np.all((f[observed] >= 1) & (f[observed] <= 5))
    counts = np.bincount(f[observed])[1:]
    assert counts.max() - counts.min() <= 0.1 * counts.mean()
    both = observed[:, decl.names.index("skewed")] & observed[:, decl.names.index("normal")]
    assert np.all(f[both, decl.names.index("skewed")] == f[both, decl.names.index("normal")])

    binary = C.make_mask(codes, decl, 0.2, np.random.default_rng(2)).to_numpy()
    assert set(np.unique(binary)) <= {0, 1}
    assert np.all(binary[~observed] == 0)

    path = tmp_path / "m.tsv"
    C.write_mask(folds, path)
    hidden = C.read_mask(path, decl, codes.index, fold=3)
    assert np.array_equal(hidden, f == 3)
    with pytest.raises(ValueError, match="--fold"):
        C.read_mask(path, decl, codes.index)
    C.write_mask(pd.DataFrame(binary, index=codes.index, columns=decl.names), path)
    assert np.array_equal(C.read_mask(path, decl, codes.index), binary == 1)
    with pytest.raises(ValueError, match="binary"):
        C.read_mask(path, decl, codes.index, fold=1)


def test_mask_without_id_column_matches_by_position(tmp_path):
    raw, decl = fitted()
    decl.id_column = None
    codes = C.to_codes(C.apply(raw.drop(columns="id"), decl), decl)
    mask = C.make_mask(codes, decl, 0.3, np.random.default_rng(0))
    path = tmp_path / "m.tsv"
    C.write_mask(mask, path)
    assert np.array_equal(C.read_mask(path, decl, codes.index), mask.to_numpy() == 1)


def test_declaration_yaml_roundtrip(tmp_path):
    raw, decl = fitted()
    decl.save(tmp_path / "d.yaml")
    again = Declaration.load(tmp_path / "d.yaml")
    assert again.to_dict() == decl.to_dict()
    assert again.fitted


def test_yaml_levels_stay_text(tmp_path):
    """Unquoted no/yes stay strings; unquoted numbers are refused, not converted."""
    path = tmp_path / "d.yaml"
    path.write_text("variables:\n- {name: a, type: categorical, levels: [no, yes]}\n")
    assert Declaration.load(path)["a"].levels == ("no", "yes")
    path.write_text("variables:\n- {name: a, type: categorical, levels: [01, 2]}\n")
    with pytest.raises(ValueError, match="quote"):
        Declaration.load(path)


def test_floats_are_rounded(tmp_path):
    from blat.formats import dump_yaml, load_yaml
    dump_yaml({"x": 3.14159265, "y": 123456.0, "z": 1e-7, "n": float("nan")},
              tmp_path / "f.yaml")
    back = load_yaml(tmp_path / "f.yaml")
    assert back["x"] == 3.142 and back["y"] == 123500.0 and back["z"] == 1e-7
    assert back["n"] != back["n"]
