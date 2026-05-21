"""Compose the six-panel risk dashboard into a single figure.

The panels are chosen so a reader can answer, at a glance: how has the value
grown, how bad were the falls, how volatile is it now, what does the loss tail
look like, where does the risk sit, and how correlated are the holdings.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import risk


def build_dashboard(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
    rolling_window: int = 63,
    risk_free_rate: float = 0.0,
    title: str = "Portfolio risk dashboard",
):
    """Return a Matplotlib Figure with the full six-panel dashboard."""
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.22)

    # 1. Equity curve
    ax = fig.add_subplot(gs[0, 0])
    equity = (1.0 + portfolio_returns).cumprod()
    ax.plot(equity.index, equity.values, color="tab:blue")
    ax.set_title("Cumulative return")
    ax.set_ylabel("Growth of 1 unit")
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = fig.add_subplot(gs[0, 1])
    dd = risk.drawdown_series(portfolio_returns)
    ax.fill_between(dd.index, dd.values, 0.0, color="tab:red", alpha=0.4)
    ax.set_title("Drawdown")
    ax.set_ylabel("Decline from peak")
    ax.grid(True, alpha=0.3)

    # 3. Rolling volatility
    ax = fig.add_subplot(gs[1, 0])
    rvol = risk.rolling_volatility(portfolio_returns, rolling_window)
    ax.plot(rvol.index, rvol.values, color="tab:purple")
    ax.set_title(f"Rolling volatility ({rolling_window}-day, annualized)")
    ax.set_ylabel("Volatility")
    ax.grid(True, alpha=0.3)

    # 4. Return distribution with VaR and CVaR
    ax = fig.add_subplot(gs[1, 1])
    var = risk.historical_var(portfolio_returns, confidence)
    cvar = risk.historical_cvar(portfolio_returns, confidence)
    ax.hist(portfolio_returns.dropna(), bins=60, color="tab:blue", alpha=0.7)
    ax.axvline(-var, color="tab:orange", linestyle="--", linewidth=2,
               label=f"VaR {confidence:.0%}: {var:.2%}")
    ax.axvline(-cvar, color="tab:red", linestyle=":", linewidth=2,
               label=f"CVaR {confidence:.0%}: {cvar:.2%}")
    ax.set_title("Daily return distribution and loss tail")
    ax.set_xlabel("Daily return")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # 5. Weight vs risk share
    ax = fig.add_subplot(gs[2, 0])
    contrib = risk.risk_contributions(weights, asset_returns)
    labels = list(contrib.index)
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2, contrib["Weight"], width, label="Weight")
    ax.bar(x + width / 2, contrib["Risk share"], width, label="Risk share")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Capital weight vs risk share")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # 6. Correlation heatmap
    ax = fig.add_subplot(gs[2, 1])
    corr = asset_returns.corr()
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Return correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    return fig


def save_figure(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path
