"""Tests for component VaR."""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_dashboard.portfolio import build_weights, portfolio_returns
from risk_dashboard.component_var import component_var, portfolio_parametric_var


def _rets(seed=0, n=2000, k=4):
    rng = np.random.default_rng(seed)
    cols = list("ABCD")[:k]
    return pd.DataFrame(rng.normal(0.0003, 0.012, size=(n, k)),
                        index=pd.bdate_range("2018-01-01", periods=n), columns=cols)


def test_components_sum_to_parametric_var():
    rets = _rets()
    w = build_weights({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}, list("ABCD"))
    cv = component_var(w, rets, 0.95)
    assert cv["component_var"].sum() == pytest.approx(
        portfolio_parametric_var(w, rets, 0.95)
    )


def test_var_shares_sum_to_one():
    rets = _rets(seed=1)
    w = build_weights({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}, list("ABCD"))
    cv = component_var(w, rets, 0.95)
    assert cv["var_share"].sum() == pytest.approx(1.0)


def test_matches_closed_form_parametric_var():
    # For a single asset, component VaR equals z * weight * sigma.
    rng = np.random.default_rng(2)
    rets = pd.DataFrame({"A": rng.normal(0, 0.02, 5000)},
                        index=pd.bdate_range("2018-01-01", periods=5000))
    w = build_weights({"A": 1.0}, ["A"])
    z = abs(NormalDist().inv_cdf(0.05))
    expected = z * rets["A"].std(ddof=1)   # pandas .cov() uses ddof=1
    assert portfolio_parametric_var(w, rets, 0.95) == pytest.approx(expected, rel=1e-3)
