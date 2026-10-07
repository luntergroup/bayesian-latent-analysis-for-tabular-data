"""Every step through the command line, on a small simulated table."""

import numpy as np
import pandas as pd
import pytest

from blat.cli import main
from blat.diagnostics import matched_cosine
from blat.formats import load_yaml
from blat.runs import Run
from blat.scores import combine_scores

SHORT = ["--alpha-frozen", "50", "--burnin", "150", "--iters", "300", "--thin", "3",
         "--quiet"]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipeline")
    assert main(["simulate", "--k", "3", "--out", str(d), "--records", "400",
                 "--new-records", "40", "--seed", "13"]) == 0   # types 30-36% each
    assert main(["convert", str(d / "example.raw.csv"), "--decl",
                 str(d / "example.declaration.yaml"), "--out", str(d / "ex"),
                 "--make-mask", "3", "--seed", "1"]) == 0
    return d


def summary(path):
    return load_yaml(path / "summary.yaml")


def test_draft_then_convert(workdir, tmp_path):
    assert main(["convert", str(workdir / "example.raw.csv"), "--id", "id",
                 "--out", str(tmp_path / "d")]) == 0
    draft = load_yaml(tmp_path / "d.draft.yaml")
    kinds = {v["name"]: v["type"] for v in draft["variables"]}
    assert kinds["num01"] == "continuous" and kinds["cat01"] == "categorical"
    assert draft["model"]["K"] is None
    assert (workdir / "ex.report.tsv").exists()


def test_run_recovers_types(workdir):
    assert main(["run", "--decl", str(workdir / "ex.yaml"), "--k", "3",
                 "--out", str(workdir / "run"), "--seed", "1"] + SHORT) == 0
    run = Run(workdir / "run")
    truth = np.load(workdir / "example.truth.npz")
    lay = run.data.layout
    discrete = ~np.repeat(lay.is_continuous, lay.vocab_size)
    cos = matched_cosine(run.phi_mean[:, discrete], truth["phi"][:, discrete])
    assert cos.min() > 0.9, cos
    assert run.record["model"]["K"] == 3 and run.record["draws"] == 100
    assert run.arrays["trace_log_joint"].size == 450
    phi = pd.read_csv(workdir / "run" / "phi.tsv", sep="\t")
    assert set(phi["variable"]) == set(run.decl.names)
    theta = pd.read_csv(workdir / "run" / "theta.tsv", sep="\t", index_col=0)
    assert theta.shape == (400, 9)
    assert (workdir / "run" / "types.tsv").exists()
    assert (workdir / "run" / "diagnostics.tsv").exists()


def test_k_is_required(workdir, capsys):
    assert main(["run", "--decl", str(workdir / "ex.yaml"), "--out",
                 str(workdir / "nok")] + SHORT) == 2
    assert "K must be given" in capsys.readouterr().err


def test_cross_validation_folds_combine(workdir):
    outputs = []
    for fold in (1, 2, 3):
        run = workdir / f"cv{fold}"
        assert main(["run", "--decl", str(workdir / "ex.yaml"), "--k", "3", "--mask",
                     str(workdir / "ex.mask.tsv"), "--fold", str(fold), "--out",
                     str(run)] + SHORT) == 0
        assert main(["impute", "--run", str(run), "--mask", str(workdir / "ex.mask.tsv"),
                     "--fold", str(fold), "--out", str(workdir / f"imp{fold}"), "--cells",
                     "--phi-draws", "5", "--theta-draws", "5", "--burn", "30",
                     "--quiet"]) == 0
        assert not summary(workdir / f"imp{fold}")["warnings"]
        assert (workdir / f"imp{fold}" / "cells.tsv").exists()
        outputs.append(workdir / f"imp{fold}")
    pooled = combine_scores(outputs)
    observed = (np.loadtxt(workdir / "ex.mask.tsv", skiprows=1, usecols=range(1, 34),
                           delimiter="\t") > 0).sum()
    assert pooled["n"].sum() == observed
    assert (pooled["r2"] > -0.5).all()
    assert 0.8 < pooled.loc[pooled["kind"] == "continuous", "coverage_95"].mean() <= 1.0


def test_impute_new_records_and_missing(workdir):
    assert main(["convert", str(workdir / "example.new.raw.csv"), "--decl",
                 str(workdir / "ex.yaml"), "--apply", "--out", str(workdir / "new")]) == 0
    assert main(["impute", "--run", str(workdir / "run"), "--data",
                 str(workdir / "new.preproc.tsv"), "--out", str(workdir / "impnew"),
                 "--intervals", "--phi-draws", "4", "--theta-draws", "4", "--quiet"]) == 0
    table = pd.read_csv(workdir / "impnew" / "imputed.tsv", sep="\t", index_col=0)
    assert len(table) == 40 and table.notna().all().all()
    assert (workdir / "impnew" / "intervals.tsv").exists()
    assert main(["impute", "--run", str(workdir / "run"), "--out",
                 str(workdir / "impmiss"), "--phi-draws", "4", "--theta-draws", "4",
                 "--quiet"]) == 0
    assert summary(workdir / "impmiss")["mode"] == "missing"


def test_ti_both_directions(workdir):
    common = ["--temperatures", "6", "--burn-per-temperature", "5",
              "--iters-per-temperature", "10", "--quiet"]
    assert main(["ti", "--decl", str(workdir / "ex.yaml"), "--k", "3", "--out",
                 str(workdir / "tif"), "--per-temperature"] + common) == 0
    assert main(["ti", "--from-run", str(workdir / "run"), "--direction", "backward",
                 "--initial-burn", "20", "--out", str(workdir / "tib")] + common) == 0
    f, b = summary(workdir / "tif"), summary(workdir / "tib")
    assert f["K"] == b["K"] == 3
    assert (workdir / "tif" / "per_temperature.tsv").exists()
    assert b["start_point"] == "draw" and "start_burn_in" in b


def test_explain_and_calibrate(workdir):
    assert main(["explain", "--run", str(workdir / "run"), "--out",
                 str(workdir / "explain"), "--burn", "20", "--draws", "10",
                 "--variables", "cat01", "ord01", "num01", "--apparent", "--quiet"]) == 0
    scores = pd.read_csv(workdir / "explain" / "scores.tsv", sep="\t", index_col=0)
    assert set(scores.index) == {"cat01", "ord01", "num01"}
    assert 0.02 < scores.loc["num01", "s_hat"] < 0.5
    assert "r2_apparent" in scores.columns
    assert main(["calibrate", "--from-run", str(workdir / "run"), "--out",
                 str(workdir / "cal"), "--rounds", "1", "--iters", "60",
                 "--burn", "10", "--draws", "10", "--quiet"]) == 0
    words = summary(workdir / "cal")["words"]
    assert all(n >= 1 for n in words.values())
    assert (workdir / "cal" / "trace.tsv").exists()
    assert main(["run", "--decl", str(workdir / "cal" / "declaration.yaml"), "--k", "3",
                 "--out", str(workdir / "runcal")] + SHORT) == 0
    assert summary(workdir / "runcal")["words"] == words
