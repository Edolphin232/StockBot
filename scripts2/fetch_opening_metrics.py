#!/usr/bin/env python3
"""
Fetch 10:00 AM opening validation metrics from Alpaca API.

Reads tickers from data/indices/sp500.csv and writes validation metrics
to data/premarket/yyyy-mm-dd_opening_1000.tsv

Usage:
    python fetch_opening_1000_metrics.py [--symbols AAPL,MSFT,GOOGL] [--date YYYY-MM-DD]
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if CURRENT_DIR.endswith("bin") else CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Lazy imports to avoid hard dependencies
def get_alpaca_client():
    """Initialize and return Alpaca historical data client"""
    from alpaca.data.historical import StockHistoricalDataClient
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_API_SECRET", "")
    if not api_key or not secret_key:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET environment variables not set")
    return StockHistoricalDataClient(api_key, secret_key)


def _read_tickers_csv(csv_path: Path) -> list[str]:
    """Read ticker symbols from CSV file"""
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
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return []


def _read_watchlist(path: Path) -> tuple[list[str], dict[str, str]]:
    """Read a TSV/CSV watchlist and return (symbols, symbol->gainer_loser label)."""
    if not path.exists():
        print(f"Watchlist not found: {path}")
        return [], {}
    try:
        sep = "\t" if path.suffix.lower() in [".tsv", ".tab"] else ","
        df = pd.read_csv(path, sep=sep)
        if "symbol" not in df.columns or df.empty:
            return [], {}
        symbols = df["symbol"].dropna().astype(str).str.upper().tolist()
        label_map: dict[str, str] = {}
        if "gainer_loser" in df.columns:
            for sym, lab in zip(df["symbol"], df["gainer_loser"]):
                if isinstance(sym, str):
                    label_map[sym.strip().upper()] = str(lab)
        return [s for s in symbols if s], label_map
    except Exception as e:
        print(f"Error reading watchlist {path}: {e}")
        return [], {}


def _safe_float(x):
    """Safely convert to float, return NaN on failure"""
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def compute_rsi(closes: list[float], length: int = 14) -> float:
    """Compute RSI for a series of closes, return latest value"""
    if len(closes) < length:
        return np.nan
    
    closes_array = np.array(closes, dtype=float)
    delta = np.diff(closes_array)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-length:])
    avg_loss = np.mean(loss[-length:])
    
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 0.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
    """
    Compute MACD for a series of closes.
    Returns (macd_line, signal_line, histogram) - all latest values
    """
    if len(closes) < slow:
        return np.nan, np.nan, np.nan
    
    closes_array = np.array(closes, dtype=float)
    
    # Compute EMAs
    ema_fast = pd.Series(closes_array).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(closes_array).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    macd_signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    macd_histogram = macd_line - macd_signal_line
    
    return float(macd_line[-1]), float(macd_signal_line[-1]), float(macd_histogram[-1])


def check_higher_lows(bars_data: list[dict]) -> bool:
    """Check if price action shows higher lows (bullish)"""
    if len(bars_data) < 3:
        return False
    
    lows = [bar.get("low") or bar.get("l") for bar in bars_data]
    lows = [l for l in lows if l is not None]
    
    if len(lows) < 3:
        return False
    
    # Check last 3 lows - each should be higher than previous
    return lows[-2] > lows[-3] and lows[-1] > lows[-2]


def check_higher_highs(bars_data: list[dict]) -> bool:
    """Check if price action shows higher highs (bullish)"""
    if len(bars_data) < 3:
        return False
    
    highs = [bar.get("high") or bar.get("h") for bar in bars_data]
    highs = [h for h in highs if h is not None]
    
    if len(highs) < 3:
        return False
    
    # Check last 3 highs - each should be higher than previous
    return highs[-2] > highs[-3] and highs[-1] > highs[-2]


def fetch_opening_1000_metrics(symbol: str, client, target_date: datetime) -> dict | None:
    """
    Fetch validation metrics at 10:00 AM ET for a given symbol.
    
    Returns dict with:
    - symbol, date, time
    - price_930_open, price_1000, move_930_to_1000_pct
    - rsi_1min, volume_trend, higher_lows, higher_highs
    - macd_hist_5min, volume_at_open
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    
    eastern = pytz.timezone("US/Eastern")
    
    # Define 9:30 and 10:00 in Eastern time
    start_930 = eastern.localize(datetime(target_date.year, target_date.month, target_date.day, 9, 30))
    end_1000 = eastern.localize(datetime(target_date.year, target_date.month, target_date.day, 10, 0))
    
    try:
        # Fetch 1-min bars from 9:30-10:00
        request_1min = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_930,
            end=end_1000,
        )
        bars_1min_resp = client.get_stock_bars(request_1min)
        df_1m = getattr(bars_1min_resp, "df", None)
        if df_1m is None or not isinstance(df_1m, pd.DataFrame) or df_1m.empty:
            return None
        # Filter to requested symbol
        if isinstance(df_1m.index, pd.MultiIndex) and "symbol" in df_1m.index.names:
            try:
                df_1m = df_1m.xs(symbol, level="symbol")
            except Exception:
                df_1m = df_1m.reset_index()
                df_1m = df_1m[df_1m["symbol"].astype(str).str.upper() == symbol]
        elif "symbol" in df_1m.columns:
            df_1m = df_1m[df_1m["symbol"].astype(str).str.upper() == symbol]
        if df_1m.empty:
            return None

        # Fetch 5-min bars from 9:30-10:00
        request_5min = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_930,
            end=end_1000,
        )
        bars_5min_resp = client.get_stock_bars(request_5min)
        df_5m = getattr(bars_5min_resp, "df", None)
        if isinstance(df_5m, pd.DataFrame) and not df_5m.empty:
            if isinstance(df_5m.index, pd.MultiIndex) and "symbol" in df_5m.index.names:
                try:
                    df_5m = df_5m.xs(symbol, level="symbol")
                except Exception:
                    df_5m = df_5m.reset_index()
                    df_5m = df_5m[df_5m["symbol"].astype(str).str.upper() == symbol]
            elif "symbol" in df_5m.columns:
                df_5m = df_5m[df_5m["symbol"].astype(str).str.upper() == symbol]
        else:
            df_5m = pd.DataFrame()

        # Extract closes and prices
        closes_1min = df_1m["close"].astype(float).tolist()
        closes_5min = df_5m["close"].astype(float).tolist() if not df_5m.empty else []
        if not closes_1min:
            return None
        price_930_open = closes_1min[0]
        price_1000 = closes_1min[-1]
        
        move_930_to_1000_pct = ((price_1000 - price_930_open) / price_930_open * 100) if price_930_open else np.nan
        
        # RSI on 1-min bars
        rsi_1min = compute_rsi(closes_1min, 14)
        
        # MACD on 5-min bars
        macd_line, macd_signal, macd_hist = compute_macd(closes_5min, 12, 26, 9)
        
        # Volume analysis
        volumes_1min = df_1m["volume"].astype(float).tolist()
        latest_volume = volumes_1min[-1] if volumes_1min else 0
        earlier_volume = np.mean(volumes_1min[:-5]) if len(volumes_1min) > 5 else np.mean(volumes_1min)
        
        volume_trend = "increasing" if latest_volume > earlier_volume else "declining"
        volume_at_open = volumes_1min[0] if volumes_1min else 0
        
        # Pattern detection
        lows_dicts = [{"low": lv} for lv in df_1m["low"].astype(float).tolist()]
        highs_dicts = [{"high": hv} for hv in df_1m["high"].astype(float).tolist()]
        higher_lows = check_higher_lows(lows_dicts)
        higher_highs = check_higher_highs(highs_dicts)
        
        # Volume confirmation (% of bars with volume > earlier average)
        volume_bars_above_avg = sum(1 for v in volumes_1min if v > earlier_volume)
        volume_confirmation_pct = (volume_bars_above_avg / len(volumes_1min) * 100) if volumes_1min else 0
        
        return {
            "symbol": symbol,
            "date": target_date.strftime("%Y-%m-%d"),
            "time": "10:00",
            "price_930_open": price_930_open,
            "price_1000": price_1000,
            "move_930_to_1000_pct": move_930_to_1000_pct,
            "rsi_1min": rsi_1min,
            "macd_hist_5min": macd_hist,
            "volume_trend": volume_trend,
            "volume_at_open": volume_at_open,
            "volume_confirmation_pct": volume_confirmation_pct,
            "higher_lows": higher_lows,
            "higher_highs": higher_highs,
            "num_1min_bars": int(len(df_1m)),
        }
    
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def _group_by_symbol(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return dict of symbol -> sub-DataFrame for convenience."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
        try:
            # groupby level symbol
            return {str(sym): g.droplevel("symbol") for sym, g in df.groupby(level="symbol", sort=False)}
        except Exception:
            pass
    if "symbol" in df.columns:
        sym_groups = {}
        for sym, g in df.groupby(df["symbol"].astype(str).str.upper(), sort=False):
            sym_groups[str(sym)] = g.drop(columns=[c for c in ["symbol"] if c in g.columns]).set_index(g.index)
        return sym_groups
    return {}


def fetch_opening_1000_metrics_batch(symbols: list[str], client, target_date: datetime, batch_size: int = 100) -> list[dict]:
    """
    Fetch validation metrics for many symbols at once by batching the Alpaca requests.
    Makes 2 API calls per batch (1m and 5m), dramatically reducing total calls.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    eastern = pytz.timezone("US/Eastern")
    start_930 = eastern.localize(datetime(target_date.year, target_date.month, target_date.day, 9, 30))
    end_1000 = eastern.localize(datetime(target_date.year, target_date.month, target_date.day, 10, 0))

    all_results: list[dict] = []

    # Process in batches to stay well under 200 calls/min
    for i in range(0, len(symbols), batch_size):
        batch_syms = [s.strip().upper() for s in symbols[i : i + batch_size] if s]
        if not batch_syms:
            continue

        # 1m bars for entire batch
        req_1m = StockBarsRequest(
            symbol_or_symbols=batch_syms,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_930,
            end=end_1000,
        )
        resp_1m = client.get_stock_bars(req_1m)
        df_1m = getattr(resp_1m, "df", None)

        # 5m bars for entire batch
        req_5m = StockBarsRequest(
            symbol_or_symbols=batch_syms,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_930,
            end=end_1000,
        )
        resp_5m = client.get_stock_bars(req_5m)
        df_5m = getattr(resp_5m, "df", None)

        sym_to_1m = _group_by_symbol(df_1m)
        sym_to_5m = _group_by_symbol(df_5m)

        for sym in batch_syms:
            try:
                df1 = sym_to_1m.get(sym, pd.DataFrame())
                if df1 is None or df1.empty:
                    continue
                # Ensure sorted by time
                df1 = df1.sort_index() if isinstance(df1.index, pd.DatetimeIndex) else df1

                df5 = sym_to_5m.get(sym, pd.DataFrame())
                df5 = df5.sort_index() if isinstance(df5.index, pd.DatetimeIndex) else df5

                closes_1min = df1["close"].astype(float).tolist() if "close" in df1.columns else []
                closes_5min = df5["close"].astype(float).tolist() if not df5.empty and "close" in df5.columns else []
                if not closes_1min:
                    continue

                price_930_open = closes_1min[0]
                price_1000 = closes_1min[-1]
                move_930_to_1000_pct = ((price_1000 - price_930_open) / price_930_open * 100) if price_930_open else np.nan

                rsi_1min = compute_rsi(closes_1min, 14)
                _, _, macd_hist = compute_macd(closes_5min, 12, 26, 9)

                volumes_1min = df1["volume"].astype(float).tolist() if "volume" in df1.columns else []
                latest_volume = volumes_1min[-1] if volumes_1min else 0
                earlier_volume = np.mean(volumes_1min[:-5]) if len(volumes_1min) > 5 else np.mean(volumes_1min) if volumes_1min else 0
                volume_trend = "increasing" if latest_volume > earlier_volume else "declining"
                volume_at_open = volumes_1min[0] if volumes_1min else 0

                lows_dicts = [{"low": lv} for lv in (df1["low"].astype(float).tolist() if "low" in df1.columns else [])]
                highs_dicts = [{"high": hv} for hv in (df1["high"].astype(float).tolist() if "high" in df1.columns else [])]
                higher_lows = check_higher_lows(lows_dicts)
                higher_highs = check_higher_highs(highs_dicts)

                volume_bars_above_avg = sum(1 for v in volumes_1min if v > earlier_volume) if volumes_1min else 0
                volume_confirmation_pct = (volume_bars_above_avg / len(volumes_1min) * 100) if volumes_1min else 0

                all_results.append(
                    {
                        "symbol": sym,
                        "date": target_date.strftime("%Y-%m-%d"),
                        "time": "10:00",
                        "price_930_open": price_930_open,
                        "price_1000": price_1000,
                        "move_930_to_1000_pct": move_930_to_1000_pct,
                        "rsi_1min": rsi_1min,
                        "macd_hist_5min": macd_hist,
                        "volume_trend": volume_trend,
                        "volume_at_open": volume_at_open,
                        "volume_confirmation_pct": volume_confirmation_pct,
                        "higher_lows": higher_lows,
                        "higher_highs": higher_highs,
                        "num_1min_bars": int(len(df1)),
                    }
                )
            except Exception as e:
                print(f"Error fetching {sym}: {e}")
                continue

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch 10:00 AM opening validation metrics from Alpaca API"
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL). If not provided, reads from data/indices/sp500.csv"
    )
    parser.add_argument(
        "--watchlist",
        default=None,
        help="Path to premarket watchlist TSV/CSV (uses 'symbol' column; merges 'gainer_loser' if present)"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Target date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Max symbols to process (default: all)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: data/premarket/yyyy-mm-dd_opening_1000.tsv)"
    )
    
    args = parser.parse_args()
    
    # Determine target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = datetime.now()
    
    # Get symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        # If watchlist provided, read symbols from it; else fall back to indices CSV
        if args.watchlist:
            wl_path = Path(args.watchlist)
            symbols, label_map = _read_watchlist(wl_path)
            if not symbols:
                print(f"No symbols in watchlist {wl_path}; falling back to indices list.")
                label_map = {}
        else:
            label_map = {}
            project_root = Path(__file__).parent.parent if Path(__file__).parent.name != "data" else Path(__file__).parent.parent.parent
            csv_path = project_root / "data" / "indices" / "sp500.csv"
            symbols = _read_tickers_csv(csv_path)
        
        if not symbols:
            print(f"No symbols found in {csv_path}")
            sys.exit(1)
    
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    
    print(f"Fetching 10:00 AM metrics for {len(symbols)} symbols on {target_date.strftime('%Y-%m-%d')}")
    
    # Initialize Alpaca client
    try:
        client = get_alpaca_client()
    except RuntimeError as e:
        print(f"Error initializing Alpaca client: {e}")
        sys.exit(1)
    
    # Fetch metrics in batches (2 requests per batch: 1m and 5m)
    results: list[dict] = []
    batch_size = 100
    for i in tqdm(range(0, len(symbols), batch_size), desc="Fetching metrics (batched)", unit="batch"):
        batch_syms = symbols[i : i + batch_size]
        batch_results = fetch_opening_1000_metrics_batch(batch_syms, client, target_date, batch_size=batch_size)
        results.extend(batch_results)
    
    if not results:
        print("No data fetched")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    # If a watchlist with labels was provided, merge label on symbol
    if 'label_map' not in locals():
        label_map = {}
    if label_map and "symbol" in df.columns:
        df["gainer_loser"] = df["symbol"].astype(str).str.upper().map(label_map)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if args.watchlist:
            # Save next to the watchlist by default
            wl_path = Path(args.watchlist)
            output_dir = wl_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{target_date.strftime('%Y-%m-%d')}_watchlist_opening_1000.tsv"
        else:
            project_root = Path(__file__).parent.parent if Path(__file__).parent.name != "data" else Path(__file__).parent.parent.parent
            output_dir = project_root / "data" / "premarket"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{target_date.strftime('%Y-%m-%d')}_opening_1000.tsv"
    
    # Write to TSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, sep="\t", float_format="%.4f")
    
    print(f"\nWrote {len(df)} rows to {output_path}")
    print(f"\nSample rows:")
    cols_to_show = [c for c in ["symbol", "time", "price_930_open", "price_1000", "move_930_to_1000_pct", "rsi_1min", "macd_hist_5min", "gainer_loser"] if c in df.columns]
    print(df[cols_to_show].head().to_string(index=False) if cols_to_show else df.head().to_string())


if __name__ == "__main__":
    main()