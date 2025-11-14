#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime, timedelta, time
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest, NewsRequest

# Ensure project root on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "premarket")


def eastern_now() -> datetime:
    return datetime.now(tz=pytz.timezone("US/Eastern"))


def normalize_symbols_for_alpaca(symbols: List[str]) -> tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Convert symbols to Alpaca/Polygon compatible forms.
    Example: 'BRK-B' -> 'BRK.B', 'BF-B' -> 'BF.B'
    Returns (normalized_list, orig_to_norm, norm_to_orig)
    """
    orig_to_norm: Dict[str, str] = {}
    norm_to_orig: Dict[str, str] = {}
    norm_list: List[str] = []
    seen: set[str] = set()
    for s in symbols:
        s_up = s.strip().upper()
        if "-" in s_up:
            # Map hyphen class-share syntax to dot notation commonly used by Alpaca/Polygon
            s_norm = s_up.replace("-", ".")
        else:
            s_norm = s_up
        # Deduplicate normalized symbols while preserving first mapping
        if s_norm not in seen:
            seen.add(s_norm)
            norm_list.append(s_norm)
            norm_to_orig[s_norm] = s_up
        # Always keep last mapping from original to normalized
        orig_to_norm[s_up] = s_norm
    return norm_list, orig_to_norm, norm_to_orig


def parse_date(date_str: Optional[str]) -> datetime:
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=pytz.timezone("US/Eastern"))
    return eastern_now()


def default_symbols(date_dt: datetime) -> List[str]:
    # Prefer same-day premarket csv if available; else S&P500 list
    pm_csv = os.path.join(DATA_DIR, f"{date_dt.strftime('%Y-%m-%d')}.csv")
    if os.path.exists(pm_csv):
        try:
            df = pd.read_csv(pm_csv)
            if "symbol" in df.columns and not df.empty:
                return df["symbol"].astype(str).dropna().unique().tolist()
        except Exception:
            pass
    sp500 = os.path.join(PROJECT_ROOT, "data", "indices", "sp500.csv")
    if os.path.exists(sp500):
        df = pd.read_csv(sp500)
        if "symbol" in df.columns:
            return df["symbol"].astype(str).dropna().unique().tolist()
        # legacy simple list
        if "symbol" not in df.columns and "MMM" in df.columns or df.shape[1] == 1:
            return df.iloc[:, 0].astype(str).tolist()
    return []


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr


def get_alpaca_clients() -> Tuple[StockHistoricalDataClient, NewsClient]:
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_API_SECRET (or APCA_API_KEY_ID/APCA_API_SECRET_KEY).")
    return StockHistoricalDataClient(api_key, api_secret), NewsClient(api_key, api_secret)


def fetch_minute_bars(client: StockHistoricalDataClient, symbols: List[str], start_dt: datetime, end_dt: datetime, tf=TimeFrame.Minute, extended=True) -> pd.DataFrame:
    # Batch symbols to avoid per-request item limits; also paginate through all pages
    frames: List[pd.DataFrame] = []
    chunk = 100
    for i in range(0, len(symbols), chunk):
        syms = symbols[i:i+chunk]
        req = StockBarsRequest(
            symbol_or_symbols=syms,
            timeframe=tf,
            start=start_dt,
            end=end_dt,
            adjustment="raw",
            feed="iex",
            limit=10000,
        )
        resp = client.get_stock_bars(req)
        if hasattr(resp, "df") and not resp.df.empty:
            frames.append(resp.df)
        next_token = getattr(resp, "next_page_token", None)
        while next_token:
            req = StockBarsRequest(
                symbol_or_symbols=syms,
                timeframe=tf,
                start=start_dt,
                end=end_dt,
                adjustment="raw",
                feed="iex",
                limit=10000,
                page_token=next_token,
            )
            resp = client.get_stock_bars(req)
            if hasattr(resp, "df") and not resp.df.empty:
                frames.append(resp.df)
            next_token = getattr(resp, "next_page_token", None)
    df = pd.concat(frames) if frames else pd.DataFrame()
    # Normalize index timezone to Eastern
    if not df.empty:
        if isinstance(df.index, pd.MultiIndex):
            # Ensure timestamp level is tz-aware and convert to Eastern
            try:
                ts = df.index.get_level_values("timestamp")
                if ts.tz is None:
                    df = df.tz_localize("UTC", level="timestamp")
            except Exception:
                # If level not named, skip localization
                pass
            try:
                df = df.tz_convert("US/Eastern", level="timestamp")
            except Exception:
                pass
        else:
            try:
                if df.index.tz is None:
                    df = df.tz_localize("UTC")
            except Exception:
                pass
            try:
                df = df.tz_convert("US/Eastern")
            except Exception:
                pass
    return df


def fetch_daily_bars(client: StockHistoricalDataClient, symbols: List[str], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    chunk = 150
    for i in range(0, len(symbols), chunk):
        syms = symbols[i:i+chunk]
        req = StockBarsRequest(
            symbol_or_symbols=syms,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            adjustment="raw",
            feed="iex",
            limit=10000,
        )
        resp = client.get_stock_bars(req)
        if hasattr(resp, "df") and not resp.df.empty:
            frames.append(resp.df)
        next_token = getattr(resp, "next_page_token", None)
        while next_token:
            req = StockBarsRequest(
                symbol_or_symbols=syms,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                adjustment="raw",
                feed="iex",
                limit=10000,
                page_token=next_token,
            )
            resp = client.get_stock_bars(req)
            if hasattr(resp, "df") and not resp.df.empty:
                frames.append(resp.df)
            next_token = getattr(resp, "next_page_token", None)
    df = pd.concat(frames) if frames else pd.DataFrame()
    if not df.empty:
        if isinstance(df.index, pd.MultiIndex):
            try:
                ts = df.index.get_level_values("timestamp")
                if ts.tz is None:
                    df = df.tz_localize("UTC", level="timestamp")
            except Exception:
                pass
            try:
                df = df.tz_convert("US/Eastern", level="timestamp")
            except Exception:
                pass
        else:
            try:
                if df.index.tz is None:
                    df = df.tz_localize("UTC")
            except Exception:
                pass
            try:
                df = df.tz_convert("US/Eastern")
            except Exception:
                pass
    return df


def compute_premarket_change(minute_df: pd.DataFrame, daily_df: pd.DataFrame, as_of: datetime) -> Tuple[pd.Series, pd.Series]:
    # Pre-market last price vs previous close; premarket last 60m volume sum
    # minute_df: MultiIndex (timestamp, symbol) columns [open, high, low, close, volume, trade_count, vwap]
    eastern = pytz.timezone("US/Eastern")
    last60_start = as_of - timedelta(minutes=60)
    rth_open = eastern.localize(datetime.combine(as_of.date(), time(9, 30)))
    ts_index = minute_df.index.get_level_values("timestamp")
    pmask = ts_index < rth_open  # before RTH
    last_mask = (ts_index >= last60_start) & (ts_index <= as_of)
    # latest premarket close price per symbol
    pm_df = minute_df[pmask]
    latest_pm = pm_df.groupby("symbol")["close"].last()
    # prev close from daily: pivot to wide then take last available day per symbol
    pct_change = pd.Series(np.nan, index=latest_pm.index, dtype=float)
    try:
        closes = daily_df["close"].unstack("symbol")
        # Use the last available completed daily bar within the range
        prev_close_series = closes.dropna(how="all").tail(1).T.iloc[:, 0]
        # Align indices
        common = latest_pm.index.intersection(prev_close_series.index)
        pct_change.loc[common] = (latest_pm.loc[common] / prev_close_series.loc[common] - 1.0) * 100.0
    except Exception:
        pass
    # volume last 60 minutes
    last60 = minute_df[last_mask]
    vol_60m = last60.groupby("symbol")["volume"].sum()
    return pct_change, vol_60m


def compute_gap_vs_atr(daily_df: pd.DataFrame, as_of_date: datetime, pm_last_price: pd.Series) -> pd.Series:
    # Use prior 30 trading days for ATR14; gap = |pm_last - prev_close| / ATR14
    closes = daily_df["close"].unstack("symbol")
    highs = daily_df["high"].unstack("symbol")
    lows = daily_df["low"].unstack("symbol")
    atr14 = pd.DataFrame({sym: compute_atr(highs[sym], lows[sym], closes[sym], 14) for sym in closes.columns})
    atr14_last = atr14.dropna(how="all").tail(1).T.iloc[:, 0]
    prev_close = closes.dropna(how="all").tail(2).iloc[-2]
    gap_vs_atr = pd.Series(np.nan, index=pm_last_price.index, dtype=float)
    idx = pm_last_price.index.intersection(prev_close.index).intersection(atr14_last.index)
    gap_vs_atr.loc[idx] = (pm_last_price.loc[idx] - prev_close.loc[idx]).abs() / atr14_last.loc[idx].replace(0, np.nan)
    return gap_vs_atr


def compute_daily_indicators(daily_df: pd.DataFrame) -> pd.DataFrame:
    closes = daily_df["close"].unstack("symbol")
    rsi14 = closes.apply(lambda s: compute_rsi(s, 14))
    sma20 = closes.rolling(window=20, min_periods=20).mean()
    sma50 = closes.rolling(window=50, min_periods=50).mean()
    latest_close = closes.tail(1).T.iloc[:, 0]
    latest_rsi = rsi14.tail(1).T.iloc[:, 0]
    latest_sma20 = sma20.tail(1).T.iloc[:, 0]
    latest_sma50 = sma50.tail(1).T.iloc[:, 0]
    df = pd.DataFrame({
        "close_daily": latest_close,
        "rsi14_daily": latest_rsi,
        "sma20": latest_sma20,
        "sma50": latest_sma50,
    })
    df["above_sma20"] = (df["close_daily"] > df["sma20"]).astype(float)
    df["above_sma50"] = (df["close_daily"] > df["sma50"]).astype(float)
    return df


def compute_5min_rsi(minute5_df: pd.DataFrame) -> pd.Series:
    close_5 = minute5_df["close"].unstack("symbol")
    rsi5 = close_5.apply(lambda s: compute_rsi(s, 14))
    return rsi5.tail(1).T.iloc[:, 0]


def compute_corr_with_proxies(minute_df: pd.DataFrame, spy_series: pd.Series, qqq_series: pd.Series, as_of: datetime) -> pd.DataFrame:
    # Use SPY/QQQ closing prices over last 60 minutes to compute per-symbol correlations
    last60_start = as_of - timedelta(minutes=60)
    mask = (minute_df.index.get_level_values("timestamp") >= last60_start) & (minute_df.index.get_level_values("timestamp") <= as_of)
    df = minute_df[mask]
    wide = df["close"].unstack("symbol")
    returns = wide.pct_change().dropna(how="all")
    # Align SPY/QQQ returns to index
    spy_rets = spy_series.loc[last60_start:as_of].pct_change().reindex(returns.index, method="nearest")
    qqq_rets = qqq_series.loc[last60_start:as_of].pct_change().reindex(returns.index, method="nearest")
    out = pd.DataFrame(index=returns.columns, columns=["spy_corr_60m", "qqq_corr_60m"], dtype=float)
    for sym in returns.columns:
        try:
            if spy_rets.notna().any():
                out.loc[sym, "spy_corr_60m"] = float(np.corrcoef(returns[sym].dropna(), spy_rets.dropna().reindex_like(returns[sym]).fillna(0))[0, 1])
            if qqq_rets.notna().any():
                out.loc[sym, "qqq_corr_60m"] = float(np.corrcoef(returns[sym].dropna(), qqq_rets.dropna().reindex_like(returns[sym]).fillna(0))[0, 1])
        except Exception:
            pass
    return out


def fetch_overnight_news_flags(news_client: NewsClient, symbols: List[str], date_dt: datetime) -> pd.Series:
    # From previous close (16:00 ET previous day) to 09:20 ET
    eastern = pytz.timezone("US/Eastern")
    prev_day = (date_dt - timedelta(days=1)).date()
    start_dt = eastern.localize(datetime.combine(prev_day, time(16, 0)))
    end_dt = eastern.localize(datetime.combine(date_dt.date(), time(9, 20)))
    flags = pd.Series(0, index=pd.Index(symbols, name="symbol"), dtype=int)
    # Alpaca News API rate limits; fetch in small batches
    batch = 50
    for i in range(0, len(symbols), batch):
        syms = symbols[i:i+batch]
        try:
            req = NewsRequest(symbols=syms, start=start_dt, end=end_dt, limit=100)
            news = news_client.get_news(req)
            # news is an iterable; count per symbol
            counts: Dict[str, int] = {}
            for item in news:
                for sym in getattr(item, "symbols", []) or []:
                    counts[sym] = counts.get(sym, 0) + 1
            for sym, cnt in counts.items():
                if sym in flags.index:
                    flags.loc[sym] = int(cnt > 0)
        except Exception:
            continue
    return flags


def build_metrics(symbols: List[str], date_dt: datetime) -> pd.DataFrame:
    stock_client, news_client = get_alpaca_clients()

    # Normalize symbols for Alpaca API, maintain mapping for output
    norm_symbols, orig_to_norm, norm_to_orig = normalize_symbols_for_alpaca(symbols)

    # Time windows
    eastern = pytz.timezone("US/Eastern")
    start_pm = eastern.localize(datetime.combine(date_dt.date(), time(4, 0)))
    as_of = eastern.localize(datetime.combine(date_dt.date(), time(9, 20))) if date_dt.date() == eastern_now().date() else eastern.localize(datetime.combine(date_dt.date(), time(9, 20)))
    end_pm = as_of

    # Fetch data
    minute_df = fetch_minute_bars(stock_client, norm_symbols, start_pm, end_pm, tf=TimeFrame.Minute)
    minute5_df = fetch_minute_bars(stock_client, norm_symbols, start_pm, end_pm, tf=TimeFrame(5, TimeFrameUnit.Minute))
    daily_df = fetch_daily_bars(stock_client, norm_symbols, start_pm - timedelta(days=70), end_pm)

    # Helpers to standardize indices and map symbols back to original
    def _standardize_df(in_df: pd.DataFrame, default_symbol: str) -> pd.DataFrame:
        df_local = in_df
        if df_local.empty:
            return df_local
        if isinstance(df_local.index, pd.MultiIndex):
            names = df_local.index.names
            # Ensure we have a 'timestamp' level; if order is (symbol, timestamp), swap to (timestamp, symbol)
            if "timestamp" in names and names[0] != "timestamp":
                try:
                    df_local = df_local.swaplevel(names.index("timestamp"), 0)
                except Exception:
                    pass
            # After swap, force names
            df_local.index.names = ["timestamp", "symbol"]
        else:
            # Single index; assume it's timestamp and add symbol level
            sym = default_symbol
            df_local.set_index([df_local.index, pd.Index([sym]*len(df_local), name="symbol")], inplace=True)
            df_local.index.names = ["timestamp", "symbol"]
        # Map normalized symbol back to original
        try:
            df_reset = df_local.reset_index()
            df_reset["symbol"] = df_reset["symbol"].map(norm_to_orig).fillna(df_reset["symbol"])
            df_local = df_reset.set_index(["timestamp", "symbol"]).sort_index()
        except Exception:
            pass
        return df_local

    # Reassign standardized dataframes
    minute_df = _standardize_df(minute_df, norm_symbols[0] if norm_symbols else "UNK")
    minute5_df = _standardize_df(minute5_df, norm_symbols[0] if norm_symbols else "UNK")
    daily_df = _standardize_df(daily_df, norm_symbols[0] if norm_symbols else "UNK")

    # P0: Premarket % change, Premarket volume (hourly)
    pm_pct_change, vol_60m = compute_premarket_change(minute_df, daily_df, as_of)

    # Gap vs ATR
    # pm_last_price from minute_df
    pm_last_price = minute_df.groupby("symbol")["close"].last()
    gap_vs_atr = compute_gap_vs_atr(daily_df, as_of, pm_last_price)

    # P1: Daily RSI(14), SMA(20/50) position
    daily_inds = compute_daily_indicators(daily_df)

    # P2: 5-min RSI(14)
    rsi5 = compute_5min_rsi(minute5_df)

    # P2: Sector % change via ETFs (global snapshot, not per-symbol mapping)
    sector_etfs = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
    etf_df = fetch_minute_bars(stock_client, sector_etfs, start_pm, end_pm, tf=TimeFrame.Minute)
    # reuse daily for prev close
    etf_daily = fetch_daily_bars(stock_client, sector_etfs, start_pm - timedelta(days=40), end_pm)
    etf_pm_last = etf_df.groupby("symbol")["close"].last() if not etf_df.empty else pd.Series(dtype=float)
    etf_prev_close = etf_daily.groupby("symbol")["close"].shift(1).groupby(level=0).last() if not etf_daily.empty else pd.Series(dtype=float)
    etf_pct_change = pd.Series(np.nan, index=sector_etfs, dtype=float)
    idx_etf = etf_pm_last.index.intersection(etf_prev_close.index)
    etf_pct_change.loc[idx_etf] = (etf_pm_last.loc[idx_etf] / etf_prev_close.loc[idx_etf] - 1.0) * 100.0

    # P0: ES/NQ correlation (proxy via SPY/QQQ) using IEX feed
    spy_series = etf_df.xs("SPY", level="symbol")["close"] if not etf_df.empty and "SPY" in etf_df.index.get_level_values("symbol") else pd.Series(dtype=float)
    qqq_series = etf_df.xs("QQQ", level="symbol")["close"] if not etf_df.empty and "QQQ" in etf_df.index.get_level_values("symbol") else pd.Series(dtype=float)
    corr_df = compute_corr_with_proxies(minute_df, spy_series, qqq_series, as_of) if not spy_series.empty and not qqq_series.empty else pd.DataFrame()

    # P3: Options IV (placeholder NaN unless later enabled)
    options_iv = pd.Series(np.nan, index=pd.Index(symbols, name="symbol"), dtype=float)

    # Overnight news flag
    # Use normalized symbols for API, map back inside the function
    news_flag = fetch_overnight_news_flags(news_client, norm_symbols, date_dt).rename(index=norm_to_orig)

    # Assemble
    out = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
    out["premarket_pct_change"] = pm_pct_change.reindex(out.index)
    out["premarket_volume_60m"] = vol_60m.reindex(out.index)
    out["gap_vs_atr14"] = gap_vs_atr.reindex(out.index)
    out["rsi14_daily"] = daily_inds["rsi14_daily"].reindex(out.index)
    out["sma20"] = daily_inds["sma20"].reindex(out.index)
    out["sma50"] = daily_inds["sma50"].reindex(out.index)
    out["above_sma20"] = daily_inds["above_sma20"].reindex(out.index)
    out["above_sma50"] = daily_inds["above_sma50"].reindex(out.index)
    out["rsi14_5m"] = rsi5.reindex(out.index)
    out["news_overnight_flag"] = news_flag.reindex(out.index)
    out["spy_corr_60m"] = corr_df["spy_corr_60m"].reindex(out.index) if not corr_df.empty else np.nan
    out["qqq_corr_60m"] = corr_df["qqq_corr_60m"].reindex(out.index) if not corr_df.empty else np.nan
    out["options_iv"] = options_iv.reindex(out.index)

    # Attach ETF sector snapshot as separate columns (same for all rows)
    for etf, val in etf_pct_change.items():
        out[f"etf_{etf}_premarket_pct_change"] = float(val) if pd.notna(val) else np.nan

    return out


def main():
    parser = argparse.ArgumentParser(description="Build premarket metrics via Alpaca API (scripts2).")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (US/Eastern). Defaults to today.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols.")
    parser.add_argument("--symbols-csv", default=None, help="CSV with 'symbol' column. Defaults to data/premarket/<date>.csv then S&P500 list.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of symbols (0 = no cap).")
    parser.add_argument("--out", default=None, help="Output TSV path. Defaults to data/premarket/<date>_alpaca_metrics.tsv")
    args = parser.parse_args()

    date_dt = parse_date(args.date)
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.symbols_csv:
        df = pd.read_csv(args.symbols_csv)
        if "symbol" not in df.columns or df.empty:
            raise RuntimeError("Provided --symbols-csv must have a 'symbol' column.")
        symbols = df["symbol"].astype(str).dropna().unique().tolist()
    else:
        symbols = default_symbols(date_dt)

    if args.limit and args.limit > 0:
        symbols = symbols[: args.limit]

    if not symbols:
        raise RuntimeError("No symbols to process.")

    out_df = build_metrics(symbols, date_dt)
    out_path = args.out or os.path.join(DATA_DIR, f"{date_dt.strftime('%Y-%m-%d')}_alpaca_metrics.tsv")
    out_df.reset_index().to_csv(out_path, index=False, sep="\t", float_format="%.4f")
    print(f"Wrote Alpaca premarket metrics: {out_path} (rows={len(out_df)})")

    # Build 9:20 watchlist (top gainers/losers with volume filter)
    try:
        # Ensure 'symbol' is a column (out_df uses symbol as index)
        df = out_df.reset_index().copy()
        # Coerce numeric and handle NaNs
        if "premarket_pct_change" in df.columns:
            df["premarket_pct_change"] = pd.to_numeric(df["premarket_pct_change"], errors="coerce")
        if "premarket_volume_60m" in df.columns:
            df["premarket_volume_60m"] = pd.to_numeric(df["premarket_volume_60m"], errors="coerce").fillna(0)

        min_volume = 10000
        # Keep only rows with a usable pct_change
        df_valid = df[df.get("premarket_pct_change").notna()] if "premarket_pct_change" in df.columns else pd.DataFrame()

        # Apply volume filter when available; if it empties the set, fall back to unfiltered
        if "premarket_volume_60m" in df_valid.columns:
            df_filtered = df_valid[df_valid["premarket_volume_60m"] >= min_volume]
            if df_filtered.empty:
                df_filtered = df_valid
        else:
            df_filtered = df_valid

        if not df_filtered.empty and "premarket_pct_change" in df_filtered.columns:
            df_sorted = df_filtered.sort_values("premarket_pct_change", ascending=False)
            top_gainers = df_sorted.head(15).copy()
            top_gainers["gainer_loser"] = "gainer"
            top_losers = df_sorted.tail(15).sort_values("premarket_pct_change").copy()
            top_losers["gainer_loser"] = "loser"
            watchlist = pd.concat([top_gainers, top_losers], ignore_index=True)
            # Avoid duplicate symbols (can happen with <30 total rows)
            if "symbol" in watchlist.columns:
                watchlist = watchlist.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
            else:
                watchlist = watchlist.reset_index(drop=True)

            watchlist_path = os.path.join(os.path.dirname(out_path), f"{date_dt.strftime('%Y-%m-%d')}_premarket_watchlist.tsv")
            watchlist.to_csv(watchlist_path, sep="\t", index=False, float_format="%.4f")
            print("Watchlist created:")
            cols_to_show = [c for c in ["symbol", "premarket_pct_change", "premarket_volume_60m", "gainer_loser"] if c in watchlist.columns]
            if cols_to_show:
                print(watchlist[cols_to_show].to_string(index=False))
            print(f"Wrote watchlist: {watchlist_path} (rows={len(watchlist)})")
        else:
            available_cols = list(out_df.columns)
            print("Skipped watchlist creation: required columns missing or no rows.")
            print(f"Available columns: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}")
    except Exception as e:
        print(f"Failed to create watchlist: {e}")


if __name__ == "__main__":
    main()


