"""Edge-case tests for the risk measures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_dashboard import risk
from risk_dashboard.portfolio import build_weights


def _days(n):
    return pd.bdate_range("2018-01-01", periods=n)


def test_weights_empty_raises():
    with pytest.raises(ValueError):
        build_weights({}, [])


def test_var_higher_confidence_is_larger_loss():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.02, 5000), index=_days(5000))
    assert risk.historical_var(r, 0.99) > risk.historical_var(r, 0.90)


def test_constant_returns_have_zero_volatility():
    r = pd.Series(np.zeros(100), index=_days(100))
    assert risk.annualized_volatility(r) == pytest.approx(0.0)


def test_max_drawdown_zero_for_monotonic_growth():
    r = pd.Series([0.01] * 50, index=_days(50))
    assert risk.max_drawdown(r) == pytest.approx(0.0)


def test_parametric_and_historical_var_close_for_normal_data():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.015, 20000), index=_days(20000))
    h = risk.historical_var(r, 0.95)
    p = risk.parametric_var(r, 0.95)
    # On genuinely normal data the two VaRs should be close.
    assert abs(h - p) < 0.002
