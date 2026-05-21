"""Portfolio risk dashboard: a portfolio seen as a bundle of risk exposures.

The dashboard pulls the usual risk views into one picture: how the value grew,
how deep it fell, how volatile it has been lately, what the loss tail looks like
(value at risk and expected shortfall), where the risk actually sits across
holdings, and how correlated those holdings are.

Modules:

    data       multi-ticker price loading behind a swappable loader
    portfolio  weights and the portfolio return series
    risk       VaR, CVaR, rolling volatility, drawdown, risk contributions
    dashboard  compose the multi-panel figure
    report     render a self-contained HTML report around the figure
"""

__version__ = "0.1.0"
