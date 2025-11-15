#!/usr/bin/env python3
"""
FULL METRIC BUILDER — with 10:15 support, micro-trend, SPY/QQQ context, wick & volume.

- Premarket: 04:00–09:30
- Opening:  09:30–10:30

Outputs per symbol:
    price_930, price_1000, price_1015, price_1030
    return_1000_to_1030_pct
    return_1015_to_1030_pct
    return_1000_to_1015_pct
    vol_930_1000, vol_1000_1015, vol_spike_1000_1015_over_930_1000
    upper_wick_930_1015_pct, lower_wick_930_1015_pct
    spy_ret_930_1000_pct, spy_ret_1000_1015_pct
    qqq_ret_930_1000_pct, qqq_ret_1000_1015_pct

Plus everything from:
    calc_premarket_metrics, calc_volume_metrics, calc_opening_metrics, calc_daily_technicals
"""

import os
import sys
import argparse
from datetime import datetime, date, time, timedelta
import pandas as pd
import pytz

# -------------------------------------------------------------------
# Ensure project path
# -------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

from fetchers.alpaca_client import fetch_bars
from fetchers.yahoo_client import fetch_daily_history_batch

from calculators.premarket_metrics import calc_premarket_metrics
from calculators.volume_metrics import calc_volume_metrics
from calculators.opening_metrics import calc_opening_metrics
from calculators.technicals import calc_daily_technicals


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

SP500_PATH = "data/indices/sp500.csv"
METRICS_BASE_DIR = "data/metrics"
ET = pytz.timezone("US/Eastern")

MARKET_ETFS = ["SPY", "QQQ"]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_sp500(path):
    df = pd.read_csv(path)
    return df["symbol"].dropna().unique().tolist()


def dt(d: date, h: int, m: int):
    return ET.localize(datetime.combine(d, time(h, m)))


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def extract_price(df: pd.DataFrame, hh: int, mm: int):
    """
    Extract the last close of the given minute bar (hh:mm).
    df index: MultiIndex [symbol, timestamp] OR just timestamp for per-symbol slice.
    """
    if df is None or df.empty:
        return None
    if isinstance(df.index, pd.MultiIndex):
        t = df.index.get_level_values("timestamp")
    else:
        t = df.index
    mask = (t.hour == hh) & (t.minute == mm)
    sub = df[mask]
    if sub.empty:
        return None
    return float(sub.iloc[-1]["close"])


def compute_micro_features(df_op_sym: pd.DataFrame):
    """
    SAFE, NO-LEAKAGE version.

    Computes:
        - vol_930_1000
        - vol_1000_1015
        - vol_spike_1000_1015_over_930_1000
        - upper_wick_930_1015_pct
        - lower_wick_930_1015_pct

    STRICT RULES:
        • Only use bars with timestamps < 10:15:00 ET
        • No inclusion of the 10:15 bar if timestamps represent bar-start
        • Convert all timestamps to US/Eastern explicitly
        • No future highs/lows can slip into the wick calculation
    """

    # ----------------------------
    # 0. Handle empty input
    # ----------------------------
    if df_op_sym is None or df_op_sym.empty:
        return {
            "vol_930_1000": None,
            "vol_1000_1015": None,
            "vol_spike_1000_1015_over_930_1000": None,
            "upper_wick_930_1015_pct": None,
            "lower_wick_930_1015_pct": None,
        }

    # ----------------------------
    # 1. Ensure timestamps are properly localized to ET
    # ----------------------------
    if isinstance(df_op_sym.index, pd.MultiIndex):
        t = df_op_sym.index.get_level_values("timestamp")
    else:
        t = df_op_sym.index

    # Convert to timezone-aware
    if t.tz is None:
        # Assume timestamps are UTC if naive
        t = t.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        t = t.tz_convert("US/Eastern")

    df = df_op_sym.copy()
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.set_levels([df.index.levels[0], t], level="timestamp")
    else:
        df.index = t

    # ----------------------------
    # 2. Define precise time windows
    # ----------------------------
    # Windows strictly before prediction cutoff
    # Get a reference date from the first timestamp
    ref_date = t[0].date() if len(t) > 0 else date.today()
    t_930  = ET.localize(datetime.combine(ref_date, time(9, 30)))
    t_1000 = ET.localize(datetime.combine(ref_date, time(10, 0)))
    t_1015 = ET.localize(datetime.combine(ref_date, time(10, 15)))

    mask_930_1000 = (t >= t_930) & (t < t_1000)
    mask_1000_1015 = (t >= t_1000) & (t < t_1015)
    mask_930_1015 = (t >= t_930) & (t < t_1015)

    # ----------------------------
    # 3. Volume windows (safe)
    # ----------------------------
    vol_930_1000 = float(df.loc[mask_930_1000, "volume"].sum()) if mask_930_1000.any() else None
    vol_1000_1015 = float(df.loc[mask_1000_1015, "volume"].sum()) if mask_1000_1015.any() else None

    if vol_930_1000 and vol_930_1000 > 0:
        vol_spike = vol_1000_1015 / vol_930_1000 if vol_1000_1015 is not None else None
    else:
        vol_spike = None

    # ----------------------------
    # 4. Wick structure — SAFE: no 10:15 bar leakage
    # ----------------------------
    if mask_930_1015.any():
        sub = df.loc[mask_930_1015]

        # Strict high/low using only bars < 10:15
        high_ = float(sub["high"].max())
        low_  = float(sub["low"].min())

        # Critical: price_1015 substitute = last CLOSE before 10:15
        sub_before_1015 = df.loc[t < t_1015]
        if not sub_before_1015.empty:
            price_1015 = float(sub_before_1015["close"].iloc[-1])
        else:
            price_1015 = None

        if price_1015 is not None and high_ > low_:
            rng = high_ - low_
            upper = (high_ - price_1015) / rng
            lower = (price_1015 - low_) / rng
        else:
            upper, lower = None, None
    else:
        upper, lower = None, None

    # ----------------------------
    # 5. Return safe dictionary
    # ----------------------------
    return {
        "vol_930_1000": vol_930_1000,
        "vol_1000_1015": vol_1000_1015,
        "vol_spike_1000_1015_over_930_1000": vol_spike,
        "upper_wick_930_1015_pct": upper,
        "lower_wick_930_1015_pct": lower,
    }



def compute_etf_intraday(etf_df: pd.DataFrame):
    """
    Compute SPY/QQQ 09:30→10:00 and 10:00→10:15 returns (shared by all symbols).
    """
    if etf_df is None or etf_df.empty:
        return {
            "spy_ret_930_1000_pct": None,
            "spy_ret_1000_1015_pct": None,
            "qqq_ret_930_1000_pct": None,
            "qqq_ret_1000_1015_pct": None,
        }

    out = {
        "spy_ret_930_1000_pct": None,
        "spy_ret_1000_1015_pct": None,
        "qqq_ret_930_1000_pct": None,
        "qqq_ret_1000_1015_pct": None,
    }

    for etf in ["SPY", "QQQ"]:
        if etf not in etf_df.index.get_level_values(0):
            continue
        df_e = etf_df.loc[etf]
        if df_e.empty:
            continue

        p_930 = extract_price(df_e, 9, 30)
        p_1000 = extract_price(df_e, 10, 0)
        p_1015 = extract_price(df_e, 10, 15)

        if p_930 and p_1000:
            ret_930_1000 = (p_1000 - p_930) / p_930 * 100.0
        else:
            ret_930_1000 = None

        if p_1000 and p_1015:
            ret_1000_1015 = (p_1015 - p_1000) / p_1000 * 100.0
        else:
            ret_1000_1015 = None

        if etf == "SPY":
            out["spy_ret_930_1000_pct"] = ret_930_1000
            out["spy_ret_1000_1015_pct"] = ret_1000_1015
        elif etf == "QQQ":
            out["qqq_ret_930_1000_pct"] = ret_930_1000
            out["qqq_ret_1000_1015_pct"] = ret_1000_1015

    return out


# -------------------------------------------------------------------
# MAIN (single-day)
# -------------------------------------------------------------------

def main(target_date=None, symbols=None):
    if target_date is None:
        target_date = date.today()

    # Symbol universe
    if symbols is None:
        print("[INIT] Loading S&P500...")
        symbols = load_sp500(SP500_PATH)
    else:
        if isinstance(symbols, str):
            symbols = [symbols]
        print(f"[INIT] Using {len(symbols)} provided symbol(s)...")

    # Time windows
    pm_start = dt(target_date, 4, 0)
    pm_end   = dt(target_date, 9, 30)

    op_start = dt(target_date, 9, 30)
    op_end   = dt(target_date, 10, 30)   # includes 10:15 and 10:30

    # -------------------------------------------------------------------
    # 1) PREMARKET 04:00–09:30
    # -------------------------------------------------------------------
    print(f"[{target_date}] [FETCH] Premarket (04:00–09:30)...")
    pm_chunks = []
    for chunk in chunk_list(symbols, 150):
        df = fetch_bars(chunk, pm_start, pm_end)
        if df is not None and not df.empty:
            pm_chunks.append(df)
    pm_df = pd.concat(pm_chunks) if pm_chunks else pd.DataFrame()
    pm_index = pm_df.index.get_level_values(0) if not pm_df.empty else []

    # -------------------------------------------------------------------
    # 2) OPENING 09:30–10:30
    # -------------------------------------------------------------------
    print(f"[{target_date}] [FETCH] Opening (09:30–10:30)...")
    op_chunks = []
    for chunk in chunk_list(symbols, 150):
        df = fetch_bars(chunk, op_start, op_end)
        if df is not None and not df.empty:
            op_chunks.append(df)
    op_df = pd.concat(op_chunks) if op_chunks else pd.DataFrame()
    op_index = op_df.index.get_level_values(0) if not op_df.empty else []

    # -------------------------------------------------------------------
    # 3) MARKET ETFS (SPY/QQQ)
    # -------------------------------------------------------------------
    print(f"[{target_date}] [FETCH] SPY/QQQ (09:30–10:30)...")
    etf_df = fetch_bars(MARKET_ETFS, op_start, op_end)
    etf_intraday = compute_etf_intraday(etf_df) if etf_df is not None and not etf_df.empty else {
        "spy_ret_930_1000_pct": None,
        "spy_ret_1000_1015_pct": None,
        "qqq_ret_930_1000_pct": None,
        "qqq_ret_1000_1015_pct": None,
    }

    # -------------------------------------------------------------------
    # 4) DAILY HISTORY
    # -------------------------------------------------------------------
    print(f"[{target_date}] [FETCH] Daily history...")
    daily_dict = fetch_daily_history_batch(symbols)

    # -------------------------------------------------------------------
    # 5) Compute metrics per symbol
    # -------------------------------------------------------------------
    print(f"[{target_date}] [PROCESS] Computing metrics...")
    results = []

    for sym in symbols:
        df_pm = pm_df.loc[sym] if sym in pm_index else pd.DataFrame()
        df_op = op_df.loc[sym] if sym in op_index else pd.DataFrame()
        df_daily = daily_dict.get(sym, pd.DataFrame())

        pre = calc_premarket_metrics(df_pm)
        vol = calc_volume_metrics(df_pm, df_daily)
        tech = calc_daily_technicals(df_daily)
        openm = calc_opening_metrics(df_op, pre.get("premarket_vwap"))

        # Prices
        price_930 = extract_price(df_op, 9, 30)
        price_1000 = extract_price(df_op, 10, 0)
        price_1015 = extract_price(df_op, 10, 15)
        price_1030 = extract_price(df_op, 10, 30)

        # Returns
        if price_1000 and price_1030:
            ret_1000_1030 = (price_1030 - price_1000) / price_1000 * 100.0
        else:
            ret_1000_1030 = None

        if price_1015 and price_1030:
            ret_1015_1030 = (price_1030 - price_1015) / price_1015 * 100.0
        else:
            ret_1015_1030 = None

        if price_1000 and price_1015:
            ret_1000_1015 = (price_1015 - price_1000) / price_1000 * 100.0
        else:
            ret_1000_1015 = None

        # Micro-structure features (volume, wick)
        micro = compute_micro_features(df_op)

        row = {
            "symbol": sym,
            **pre,
            **vol,
            **openm,
            **tech,
            # prices / returns
            "price_930": price_930,
            "price_1000": price_1000,
            "price_1015": price_1015,
            "price_1030": price_1030,
            "return_1000_to_1030_pct": ret_1000_1030,
            "return_1015_to_1030_pct": ret_1015_1030,
            "return_1000_to_1015_pct": ret_1000_1015,
            # micro features
            **micro,
            # market context
            **etf_intraday,
        }

        results.append(row)

    # -------------------------------------------------------------------
    # 6) Save metrics.tsv
    # -------------------------------------------------------------------
    date_str = target_date.strftime("%Y-%m-%d")
    out_dir = os.path.join(METRICS_BASE_DIR, f"{target_date.year:04d}", f"{target_date.month:02d}", date_str)
    out_path = os.path.join(out_dir, "metrics.tsv")

    print(f"[{target_date}] [SAVE] → {out_path}")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, sep="\t", index=False)

    print(f"[{target_date}] [DONE]")


# -------------------------------------------------------------------
# Range Runner
# -------------------------------------------------------------------

def run_range(start_date: date, end_date: date, symbols=None):
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:
            print(f"\n======= Running {cur} =======")
            try:
                main(target_date=cur, symbols=symbols)
            except Exception as e:
                print(f"[ERROR] {cur}: {e}")
        else:
            print(f"[SKIP] {cur} (weekend)")
        cur += timedelta(days=1)
    print("\n[COMPLETE]")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build metrics with micro features + SPY/QQQ context.")
    parser.add_argument("-d", "--date")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("-s", "--symbol", action="append")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.date and not args.start:
        d = date.fromisoformat(args.date)
        symbols = args.symbol if args.symbol else None
        main(target_date=d, symbols=symbols)
        sys.exit(0)

    if args.start:
        s = date.fromisoformat(args.start)
        e = date.fromisoformat(args.end) if args.end else s
        symbols = args.symbol if args.symbol else None
        run_range(s, e, symbols)
        sys.exit(0)

    main()
