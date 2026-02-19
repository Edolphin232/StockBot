# data/store.py
"""
DataStore: single in-memory cache for all market data.
Fetch once, slice forever. Never calls the API directly —
that's the fetcher's job.
"""
from dataclasses import dataclass, field
import pandas as pd
from typing import Optional


@dataclass
class DataStore:
    """
    Holds all fetched market data in memory.
    Both backtest and live modes use the same interface.
    """
    _spy_bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    _vix_bars: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------ #
    #  Loading (called once by the runner, never by calculators/filters)  #
    # ------------------------------------------------------------------ #

    def load_spy(self, bars: pd.DataFrame) -> None:
        """Load bulk SPY bars fetched from Alpaca."""
        self._spy_bars = bars

    def load_vix(self, bars: pd.DataFrame) -> None:
        """Load VIX bars from yfinance (flat date index)."""
        vix = bars.copy()
        vix.index = bars.index.astype(str).str[:10]  # normalize to YYYY-MM-DD
        self._vix_bars = vix

    # ------------------------------------------------------------------ #
    #  Slicing (what calculators and filters actually call)               #
    # ------------------------------------------------------------------ #

    def get_day_bars(self, date: str) -> pd.DataFrame:
        if self._spy_bars.empty:
            return pd.DataFrame()
        mask = self._spy_bars.index.get_level_values("timestamp").date.astype(str).astype(str) == str(date)
        return self._spy_bars[mask]

    def get_prev_close(self, date: str) -> Optional[float]:
        """
        Return SPY closing price of the day before date.
        Pulls from stored bars — no extra API call needed.
        """
        if self._spy_bars.empty:
            return None
        all_dates = sorted(set(
            self._spy_bars.index.get_level_values("timestamp").date.astype(str)
        ))
        if date not in all_dates:
            return None
        idx = all_dates.index(date)
        if idx == 0:
            return None  # No previous day available
        prev_date = all_dates[idx - 1]
        prev_bars = self.get_day_bars(prev_date)
        if prev_bars.empty:
            return None
        return float(prev_bars.iloc[-1]["close"])

    def get_vix(self, date: str) -> Optional[float]:
        if self._vix_bars.empty:
            return None
        mask = self._vix_bars.index == str(date)
        day = self._vix_bars[mask]
        if day.empty:
            return None
        return float(day.iloc[0]["open"])

    def get_trading_dates(self) -> list[str]:
        if self._spy_bars.empty:
            return []
        return sorted(set(
            self._spy_bars.index.get_level_values("timestamp").date.astype(str).tolist()
        ))

    def is_ready(self) -> bool:
        """Sanity check before running any pipeline."""
        return not self._spy_bars.empty and not self._vix_bars.empty