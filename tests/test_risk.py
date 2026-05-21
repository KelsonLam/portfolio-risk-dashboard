"""Tests for the risk measures and portfolio helpers.

Synthetic returns keep these fast and offline.
"""

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
from risk_dashboard import risk


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


def test_weights_normalize():
    w = build_weights({"A": 3.0, "B": 1.0}, ["A", "B"])
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] == pytest.approx(0.75)


def test_portfolio_return_weighted_average():
    rets = pd.DataFrame({"A": [0.02, -0.01], "B": [0.0, 0.03]}, index=_days(2))
    w = build_weights({"A": 0.5, "B": 0.5}, ["A", "B"])
    port = portfolio_returns(rets, w)
    pd.testing.assert_series_equal(port, 0.5 * rets["A"] + 0.5 * rets["B"], check_names=False)


def test_historical_var_matches_percentile():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 10_000), index=_days(10_000))
    var = risk.historical_var(r, confidence=0.95)
    expected = -np.percentile(r, 5)
    assert var == pytest.approx(expected)


def test_cvar_at_least_var():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.012, 10_000), index=_days(10_000))
    var = risk.historical_var(r, 0.95)
    cvar = risk.historical_cvar(r, 0.95)
    # The average tail loss cannot be smaller than the threshold loss.
    assert cvar >= var


def test_parametric_var_formula():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0005, 0.01, 5_000), index=_days(5_000))
    var = risk.parametric_var(r, 0.95)
    z = NormalDist().inv_cdf(0.05)
    expected = -(r.mean() + z * r.std(ddof=1))
    assert var == pytest.approx(expected)


def test_max_drawdown_negative():
    r = pd.Series([0.03, -0.08, 0.01, -0.02], index=_days(4))
    assert risk.max_drawdown(r) <= 0.0


def test_risk_contributions_sum_to_vol():
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(
        rng.normal(0.0004, 0.012, size=(1500, 4)),
        index=_days(1500), columns=list("ABCD"),
    )
    w = build_weights({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}, list("ABCD"))
    contrib = risk.risk_contributions(w, rets)
    port_vol = risk.annualized_volatility(portfolio_returns(rets, w))
    assert contrib["Risk contribution"].sum() == pytest.approx(port_vol, rel=1e-6)
    assert contrib["Risk share"].sum() == pytest.approx(1.0, rel=1e-6)
