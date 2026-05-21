"""Command line entry point: build the risk dashboard.

Examples
--------
Build from config.yaml (prints metrics, writes the dashboard image and HTML)::

    python scripts/build_dashboard.py

Only print the metrics, skip the files::

    python scripts/build_dashboard.py --no-files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_dashboard.data import YFinanceLoader, daily_returns
from risk_dashboard.portfolio import build_weights, portfolio_returns
from risk_dashboard import risk, dashboard, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a portfolio risk dashboard.")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
    )
    p.add_argument("--start", help="Override the start date (YYYY-MM-DD).")
    p.add_argument("--end", help="Override the end date (YYYY-MM-DD).")
    p.add_argument("--no-cache", action="store_true", help="Force a fresh download.")
    p.add_argument("--no-files", action="store_true", help="Print metrics only.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    holdings = cfg["holdings"]
    tickers = list(holdings.keys())
    start = args.start or cfg["period"]["start"]
    end = args.end or cfg["period"]["end"]
    confidence = cfg["risk"]["confidence"]
    window = cfg["risk"]["rolling_window"]
    rf = cfg.get("risk_free_rate", 0.0)

    print(f"Loading {len(tickers)} holdings from {start} to {end} ...")
    loader = YFinanceLoader(use_cache=not args.no_cache)
    prices = loader.load(tickers, start, end)
    rets = daily_returns(prices)

    weights = build_weights(holdings, list(rets.columns))
    port = portfolio_returns(rets, weights)
    metrics = risk.summarize(port, confidence=confidence, risk_free_rate=rf)

    print("\nPortfolio risk summary")
    print("-" * 40)
    width = max(len(k) for k in metrics)
    for key, value in metrics.items():
        pct = ("return" in key.lower() or "volatility" in key.lower()
               or "drawdown" in key.lower() or "var" in key.lower())
        shown = f"{value * 100:,.2f}%" if (pct and "sharpe" not in key.lower()) else f"{value:,.2f}"
        print(f"{key:<{width}}  {shown}")

    if not args.no_files:
        fig = dashboard.build_dashboard(
            port, rets, weights,
            confidence=confidence, rolling_window=window, risk_free_rate=rf,
        )
        img = dashboard.save_figure(fig, "results/dashboard.png")
        html = report.render_report(img, metrics, out_path="results/dashboard.html")
        print(f"\nSaved dashboard image to {img}")
        print(f"Saved HTML report to {html}")


if __name__ == "__main__":
    main()
