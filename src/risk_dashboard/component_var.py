"""Component VaR: which holding is actually driving the portfolio's risk.

Portfolio VaR is a single number. Component VaR splits it into additive
per-holding pieces that sum back to the total, so you can see who owns the risk,
not just who owns the capital. It is the VaR analogue of the volatility risk
decomposition already in risk.py.

Under a normal (parametric) assumption with daily covariance S and weights w:

    portfolio daily volatility = sqrt(w' S w)
    marginal VaR_i = z * (S w)_i / volatility
    component VaR_i = w_i * marginal VaR_i

where z is the standard-normal quantile for the confidence level. The component
VaRs sum to the parametric portfolio VaR (in the zero-mean convention).
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def component_var(
    weights: pd.Series, asset_returns: pd.DataFrame, confidence: float = 0.95
) -> pd.DataFrame:
    """Per-holding component VaR (one-day), with shares that sum to one."""
    cols = list(asset_returns.columns)
    w = weights.reindex(cols).fillna(0.0).to_numpy(dtype=float)

    cov = asset_returns.cov().to_numpy()      # daily covariance, so VaR is 1-day
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

    z = abs(NormalDist().inv_cdf(1.0 - confidence))

    if port_vol == 0:
        comp = np.zeros_like(w)
    else:
        marginal = z * (cov @ w) / port_vol
        comp = w * marginal

    total = comp.sum()
    share = comp / total if total != 0 else np.zeros_like(w)

    return pd.DataFrame(
        {"weight": w, "component_var": comp, "var_share": share}, index=cols
    )


def portfolio_parametric_var(
    weights: pd.Series, asset_returns: pd.DataFrame, confidence: float = 0.95
) -> float:
    """The one-day parametric portfolio VaR these components sum to."""
    return float(component_var(weights, asset_returns, confidence)["component_var"].sum())
