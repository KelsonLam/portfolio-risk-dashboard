"""Price loading behind a small, vendor-agnostic interface.

The dashboard only asks for a tidy frame of adjusted close prices. Swapping
yfinance for a paid feed is a one-class change. Downloads are cached to parquet.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")


def _cache_key(tickers: Sequence[str], start: str, end: str) -> str:
    raw = "|".join([",".join(sorted(tickers)), start, end])
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"prices_{digest}.parquet"


class YFinanceLoader:
    """Adjusted close prices from Yahoo Finance, with parquet caching."""

    def __init__(
        self, cache_dir: Path | str = DEFAULT_CACHE_DIR, use_cache: bool = True
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

    def load(
        self, tickers: Sequence[str], start: str, end: str
    ) -> pd.DataFrame:
        tickers = list(tickers)
        cache_path = self.cache_dir / _cache_key(tickers, start, end)
        if self.use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        prices = self._download(tickers, start, end)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)
        return prices

    @staticmethod
    def _download(
        tickers: Sequence[str], start: str, end: str
    ) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            list(tickers), start=start, end=end,
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise ValueError("No data returned. Check the tickers and dates.")
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].copy()
            prices.columns = [tickers[0]]
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().dropna(axis=1, how="all")
        ordered = [t for t in tickers if t in prices.columns]
        return prices[ordered]


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")
