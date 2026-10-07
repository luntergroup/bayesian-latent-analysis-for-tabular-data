import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blat.config import ModelConfig                      # noqa: E402
from blat.declaration import Declaration, Variable       # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true",
                     help="also run the slow tests (run_all.sh end to end)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        return
    skip = pytest.mark.skip(reason="slow; run with --all")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def tiny_declaration(continuous_words: int | None = 1) -> Declaration:
    """Two binary variables, one 3-level ordinal, and optionally one continuous one."""
    variables = [Variable("a", "categorical", ("no", "yes")),
                 Variable("b", "categorical", ("x", "y", "z")),
                 Variable("o", "ordinal", ("L0", "L1", "L2"))]
    if continuous_words is not None:
        v = Variable.from_dict({"name": "c", "type": "continuous"})
        v.words = continuous_words
        v.transform["fitted"] = {"log": False, "lo": 0.0, "hi": 1.0, "skew": 0.0,
                                 "skew_after": 0.0}
        variables.append(v)
    return Declaration(variables=variables)


@pytest.fixture
def tiny():
    decl = tiny_declaration()
    codes = np.array([[0, 2, 0, 0.3],
                      [1, 0, 2, 0.8],
                      [np.nan, 1, 1, np.nan]])
    cfg = ModelConfig(K=2, alpha_init=0.7, beta=0.6, gamma=(0.5, 0.5))
    return decl, codes, cfg
