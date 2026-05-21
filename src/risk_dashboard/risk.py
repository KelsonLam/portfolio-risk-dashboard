"""The risk measures behind the dashboard.

Everything here treats a daily return series as the raw material. Value at risk
and conditional value at risk are reported as positive numbers, read as "a loss
of this size or worse." Volatility figures are annualized with 252 trading days.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """One-period VaR straight from the empirical distribution.

    At 95% confidence this is the loss that only the worst 5% of days exceed.
    """
    alpha = 1.0 - confidence
    quantile = np.percentile(returns.dropna(), alpha * 100.0)
    return float(-quantile)


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (expected shortfall): the average loss in the tail."""
    alpha = 1.0 - confidence
    r = returns.dropna()
    quantile = np.percentile(r, alpha * 100.0)
    tail = r[r <= quantile]
    if len(tail) == 0:
        return float(-quantile)
    return float(-tail.mean())


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """VaR under a normal assumption, for comparison with the historical figure.

    Where the two diverge, the return distribution is not normal, which is the
    interesting part: the normal model usually understates the real tail.
    """
    r = returns.dropna()
    mu = r.mean()
    sigma = r.std(ddof=1)
    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(alpha)   # negative number for alpha < 0.5
    return float(-(mu + z * sigma))


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def annualized_return(returns: pd.Series) -> float:
    return float(returns.mean() * TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    excess = returns - rf_daily
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float((excess.mean() / vol) * np.sqrt(TRADING_DAYS))


def rolling_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    """Annualized rolling standard deviation."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def risk_contributions(
    weights: pd.Series, asset_returns: pd.DataFrame
) -> pd.DataFrame:
    """Decompose portfolio volatility into each holding's contribution.

    For weights w and annualized covariance S, portfolio volatility is
    sqrt(w' S w). Holding i contributes w_i times (S w)_i / volatility, and the
    contributions sum exactly to the portfolio volatility.
    """
    cols = list(asset_returns.columns)
    w = weights.reindex(cols).fillna(0.0).to_numpy()
    cov = asset_returns.cov().to_numpy() * TRADING_DAYS
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

    if port_vol == 0:
        contributions = np.zeros_like(w)
    else:
        marginal = cov @ w / port_vol
        contributions = w * marginal
    pct = contributions / port_vol if port_vol > 0 else np.zeros_like(w)

    return pd.DataFrame(
        {"Weight": w, "Risk contribution": contributions, "Risk share": pct},
        index=cols,
    )


def summarize(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Headline risk numbers for the summary panel."""
    return {
        "Annual return": annualized_return(portfolio_returns),
        "Annual volatility": annualized_volatility(portfolio_returns),
        "Sharpe ratio": sharpe_ratio(portfolio_returns, risk_free_rate),
        "Max drawdown": max_drawdown(portfolio_returns),
        f"Historical VaR ({confidence:.0%})": historical_var(portfolio_returns, confidence),
        f"Historical CVaR ({confidence:.0%})": historical_cvar(portfolio_returns, confidence),
        f"Parametric VaR ({confidence:.0%})": parametric_var(portfolio_returns, confidence),
    }
