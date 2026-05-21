"""Weights and the portfolio return series.

Constant-weight portfolio: the weights are reset to their targets each day. It
is the cleanest definition of a portfolio with fixed weights and keeps the risk
analytics straightforward.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def build_weights(
    holdings: Mapping[str, float | None], tickers: Sequence[str]
) -> pd.Series:
    """Normalize a holdings mapping into weights over ``tickers``.

    Explicit weights are kept, missing ones split the remainder equally, and the
    result is normalized to sum to 1.
    """
    weights = pd.Series(0.0, index=list(tickers), dtype=float)
    known = {t: float(holdings[t]) for t in tickers
             if t in holdings and holdings[t] is not None}
    unknown = [t for t in tickers if t not in known]

    for ticker, value in known.items():
        weights[ticker] = value
    if unknown:
        leftover = max(0.0, 1.0 - sum(known.values()))
        share = leftover / len(unknown) if leftover > 0 else 0.0
        for ticker in unknown:
            weights[ticker] = share

    total = weights.sum()
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return weights / total


def portfolio_returns(
    asset_returns: pd.DataFrame, weights: pd.Series
) -> pd.Series:
    """Daily return of the constant-weight portfolio."""
    aligned = weights.reindex(asset_returns.columns).fillna(0.0)
    return asset_returns.mul(aligned, axis=1).sum(axis=1)
