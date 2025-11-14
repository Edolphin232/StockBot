import os
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

from pathlib import Path

import finnhub
import pandas as pd
import pytz
from tqdm import tqdm
# yfinance and pandas-ta are imported lazily inside the functions that need them


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
    universe = tickers[: max_tickers if isinstance(max_tickers, int) and max_tickers > 0 else None]
    results: List[Dict[str, Any]] = []
    for symbol in tqdm(universe, desc="Premarket symbols", unit="sym"):
        quote = client.quote(symbol)
        last_price = quote.get("c")
        previous_close = quote.get("pc")
        if last_price is None or previous_close is None:
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            continue
        total_volume = quote.get("v") or 0
        premarket_pct_change = (last_price - previous_close) / previous_close * 100.0
        results.append(
            {
                "symbol": symbol,
                "last_premarket_price": last_price,
                "previous_close": previous_close,
                "premarket_pct_change": premarket_pct_change,
                "premarket_volume": total_volume,
            }
        )
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return results


def fetch_premarket_metrics_yf(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    raise NotImplementedError("yfinance backend has been removed. Use Finnhub-only fetch.")


def fetch_technical_indicators_yf(
    tickers: List[str],
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Compute technical indicators via Yahoo Finance using only pandas:
    - last_close, last_volume
    - Bollinger Bands (20, 2)
    - MACD (12, 26, 9)
    - RSI(14)
    - SMA(20), SMA(50)
    Returns one row per symbol.
    """
    if not tickers:
        return pd.DataFrame(columns=["symbol"])

    # Lazy import to avoid hard dependency when not using tech features
    import yfinance as yf
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist

    def compute_bollinger(close: pd.Series, window: int = 20, num_std: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid = close.rolling(window=window, min_periods=window).mean()
        sd = close.rolling(window=window, min_periods=window).std()
        upper = mid + num_std * sd
        lower = mid - num_std * sd
        return upper, mid, lower

    results: list[dict[str, Any]] = []
    multi = isinstance(data.columns, pd.MultiIndex)

    for symbol in tickers:
        sd: pd.DataFrame
        if multi:
            if symbol not in data.columns.get_level_values(0):
                continue
            sd = data[symbol]
        else:
            # Single-ticker shape
            sd = data
        if sd.empty or "Close" not in sd or "Volume" not in sd:
            continue

        close = sd["Close"].dropna()
        volume = sd["Volume"].dropna()
        if close.empty:
            continue

        # Indicators
        rsi14 = compute_rsi(close, 14)
        macd_line, macd_signal, macd_hist = compute_macd(close, 12, 26, 9)
        bb_upper, bb_middle, bb_lower = compute_bollinger(close, 20, 2)
        sma20 = close.rolling(window=20, min_periods=20).mean()
        sma50 = close.rolling(window=50, min_periods=50).mean()

        # Volume context from the same dataset
        avg_vol_20d = float(volume.tail(20).mean()) if not volume.empty else None
        prev_vol_1d = float(volume.iloc[-1]) if not volume.empty else None

        results.append(
            {
                "symbol": symbol,
                "yf_last_close": float(close.iloc[-1]),
                "yf_last_volume": int(volume.iloc[-1]) if not volume.empty else None,
                "bb_upper": float(bb_upper.dropna().iloc[-1]) if not bb_upper.dropna().empty else None,
                "bb_middle": float(bb_middle.dropna().iloc[-1]) if not bb_middle.dropna().empty else None,
                "bb_lower": float(bb_lower.dropna().iloc[-1]) if not bb_lower.dropna().empty else None,
                "macd": float(macd_line.dropna().iloc[-1]) if not macd_line.dropna().empty else None,
                "macd_signal": float(macd_signal.dropna().iloc[-1]) if not macd_signal.dropna().empty else None,
                "macd_hist": float(macd_hist.dropna().iloc[-1]) if not macd_hist.dropna().empty else None,
                "rsi14": float(rsi14.dropna().iloc[-1]) if not rsi14.dropna().empty else None,
                "sma20": float(sma20.dropna().iloc[-1]) if not sma20.dropna().empty else None,
                "sma50": float(sma50.dropna().iloc[-1]) if not sma50.dropna().empty else None,
                "avg_volume_20d": avg_vol_20d,
                "prev_volume_1d": prev_vol_1d,
            }
        )

    return pd.DataFrame(results)


# ---------------- Convenience helpers (match requested guideline) ----------------
def get_premarket_data(symbols: List[str], finnhub_client: finnhub.Client) -> List[Dict[str, Any]]:
    """
    For each symbol, pull Finnhub quote and compute simple premarket fields.
    """
    premarket_moves: List[Dict[str, Any]] = []
    for symbol in symbols:
        quote = finnhub_client.quote(symbol)
        premarket_moves.append(
            {
                "symbol": symbol,
                "premarket_price": quote.get("c"),
                "change_pct": quote.get("dp"),
                "gap_from_close": (quote.get("c") or 0) - (quote.get("pc") or 0),
            }
        )
    return premarket_moves


def get_technical_indicators(symbol: str) -> pd.DataFrame:
    """
    Download recent daily data and compute RSI(14), MACD(12,26,9), SMA(20), SMA(50), Bollinger(20,2).
    Returns a DataFrame with added columns.
    """
    # Lazy import
    import yfinance as yf
    df = yf.download(symbol, period="60d", interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        return df
    close = df["Close"].dropna()
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd_line
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_line - macd_signal
    # SMAs
    df["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
    df["SMA_50"] = close.rolling(window=50, min_periods=50).mean()
    # Bollinger
    mid = close.rolling(window=20, min_periods=20).mean()
    sd = close.rolling(window=20, min_periods=20).std()
    df["BB_upper"] = mid + 2 * sd
    df["BB_middle"] = mid
    df["BB_lower"] = mid - 2 * sd
    return df


def get_volume_context(symbol: str) -> Tuple[float, float] | Tuple[None, None]:
    """
    Compute average daily volume over 20 days and previous day's volume.
    Returns (avg_volume, prev_volume). Returns (None, None) if unavailable.
    """
    # Lazy import
    import yfinance as yf
    df = yf.download(symbol, period="20d", interval="1d", auto_adjust=False, progress=False)
    if df.empty or "Volume" not in df:
        return None, None
    avg_volume = float(df["Volume"].mean())
    prev_volume = float(df["Volume"].iloc[-1])
    return avg_volume, prev_volume
def rank_top_movers(data: List[Dict[str, Any]], top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(data)
    if df.empty:
        return df, df
    df = df.sort_values(by="premarket_pct_change", ascending=False)
    return df.head(top_n), df.tail(top_n)


