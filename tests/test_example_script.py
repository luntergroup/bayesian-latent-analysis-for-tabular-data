"""run_all.sh end to end, with every setting shrunk (notebooks included if available)."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent

TINY = dict(BLAT_SETTINGS="test", RECORDS="300", NEW_RECORDS="40", FOLDS="2", K_TRUE="3",
            K_GRID="2 3", K_FINAL="3", CHAINS="2", FINAL_CHAINS="2", ALPHA_FROZEN="20",
            BURNIN="50", ITERS="100", THIN="5", TI_TEMPERATURES="6", TI_BURN="5", TI_ITERS="10",
            TI_INITIAL_BURN="10", TI_REPLICATES="1", CAL_ROUNDS="1", CAL_ITERS="60",
            CAL_BURN="10", CAL_DRAWS="10", EXPLAIN_BURN="10", EXPLAIN_DRAWS="10",
            IMPUTE_PHI_DRAWS="3", IMPUTE_THETA_DRAWS="3", IMPUTE_BURN="10")


@pytest.mark.slow
def test_run_all(tmp_path):
    notebooks = all(importlib.util.find_spec(m) for m in
                    ("nbconvert", "ipykernel", "matplotlib", "sklearn"))
    env = dict(os.environ, **TINY, PYTHON=sys.executable, DATA=str(tmp_path / "data"),
               RESULTS=str(tmp_path / "results"), NOTEBOOKS="1" if notebooks else "0")
    result = subprocess.run([str(HERE / "run_all.sh")], capture_output=True, text=True,
                            env=env)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "test settings" in result.stdout
    results = tmp_path / "results"
    assert len(list((results / "runs").glob("*/summary.yaml"))) == 4
    assert len(list((results / "ti").glob("*/summary.yaml"))) == 4
    assert len(list((results / "final").glob("K3-c*/summary.yaml"))) == 2
    assert (results / "impute" / "cross_validation.tsv").exists()
    assert (results / "impute" / "new" / "imputed.tsv").exists()
    if notebooks:
        figures = {p.name for p in (results / "figures").glob("*.pdf")}
        assert {"figure1-convergence.pdf", "figure2-evidence.pdf", "figure3-tsne.pdf",
                "figure4-loadings-K3.pdf", "figure5-types-K3-discrete.pdf",
                "figure5-types-K3-continuous.pdf", "figure6-chains-K3.pdf"} <= figures


def test_run_all_needs_settings():
    env = {k: v for k, v in os.environ.items() if k != "BLAT_SETTINGS"}
    result = subprocess.run([str(HERE / "run_all.sh")], capture_output=True, text=True,
                            env=env)
    assert result.returncode == 2 and "quick.sh" in result.stderr
