import os
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

from pathlib import Path

import finnhub
import pandas as pd
import pytz


def get_finnhub_client() -> finnhub.Client:
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY is not set")
    return finnhub.Client(api_key=api_key)


# ---------- Static universe (CSV) ----------
def _csv_path_for_universe(universe: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    name = universe.lower()
    mapping = {"sp500": "sp500.csv", "nasdaq100": "nasdaq100.csv", "dji": "dji.csv"}
    filename = mapping.get(name, f"{name}.csv")
    return root / "data" / "indices" / filename


def _read_tickers_csv(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        return []
    try:
        df = pd.read_csv(csv_path)
        # Accept either a single 'symbol'/'Symbol' column or first column
        for col in ("symbol", "Symbol", "ticker", "Ticker"):
            if col in df.columns:
                col_vals = df[col].dropna().astype(str).tolist()
                return [s.strip().upper() for s in col_vals if s and isinstance(s, str)]
        first_col = df.columns[0]
        col_vals = df[first_col].dropna().astype(str).tolist()
        return [s.strip().upper() for s in col_vals if s and isinstance(s, str)]
    except Exception:
        return []


def get_universe_tickers(universe: str, client: finnhub.Client | None = None) -> List[str]:
    """
    Load tickers for a given universe strictly from data/indices/<universe>.csv.
    """
    csv_path = _csv_path_for_universe(universe)
    tickers = _read_tickers_csv(csv_path)
    if not tickers:
        raise RuntimeError(f"CSV for universe '{universe}' not found or empty at {csv_path}")
    return tickers


def trading_day_et(now_et: datetime) -> datetime:
    # If weekend, roll back to Friday
    dt = now_et
    while dt.weekday() >= 5:  # 5=Sat, 6=Sun
        dt = dt - timedelta(days=1)
    return dt


def premarket_window_et(now_utc: datetime | None = None) -> Tuple[int, int]:
    eastern = pytz.timezone("US/Eastern")
    if now_utc is None:
        now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    now_et = now_utc.astimezone(eastern)
    # If before 4:00 ET, use previous trading day
    base_day = trading_day_et(now_et)
    if now_et.hour < 4 or (now_et.hour == 4 and now_et.minute == 0):
        base_day = trading_day_et(base_day - timedelta(days=1))
    start_et = eastern.localize(datetime(base_day.year, base_day.month, base_day.day, 4, 0))
    end_et = eastern.localize(datetime(base_day.year, base_day.month, base_day.day, 9, 30))
    start_utc = int(start_et.astimezone(pytz.utc).timestamp())
    end_utc = int(end_et.astimezone(pytz.utc).timestamp())
    return start_utc, end_utc


def fetch_premarket_metrics(
    client: finnhub.Client,
    tickers: List[str],
    from_ts: int,
    to_ts: int,
    sleep_sec: float = 0.4,
    max_tickers: int | None = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    universe = tickers[: max_tickers if isinstance(max_tickers, int) and max_tickers > 0 else None]
    for symbol in universe:
        try:
            bars = client.stock_candles(symbol, "1", from_ts, to_ts)
            if not bars or bars.get("s") != "ok" or not bars.get("c"):
                time.sleep(sleep_sec)
                continue
            last_premarket_price = bars["c"][-1]
            total_volume = sum(bars.get("v", [])) if bars.get("v") else 0

            quote = client.quote(symbol)
            previous_close = quote.get("pc")
            if not previous_close:
                time.sleep(sleep_sec)
                continue
            premarket_pct_change = (last_premarket_price - previous_close) / previous_close * 100.0

            results.append(
                {
                    "symbol": symbol,
                    "last_premarket_price": last_premarket_price,
                    "previous_close": previous_close,
                    "premarket_pct_change": premarket_pct_change,
                    "premarket_volume": total_volume,
                }
            )
        except Exception:
            # Skip problematic ticker and continue
            pass
        finally:
            time.sleep(sleep_sec)
    return results


def rank_top_movers(data: List[Dict[str, Any]], top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(data)
    if df.empty:
        return df, df
    df = df.sort_values(by="premarket_pct_change", ascending=False)
    return df.head(top_n), df.tail(top_n)


