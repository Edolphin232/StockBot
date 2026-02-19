#!/usr/bin/env python3
"""
SPY Debit Spread Scanner — Evening Scan Tool
=============================================
Checks three conditions for a 14-21 DTE debit spread entry:
  1. IV Rank below threshold (options are cheap)
  2. Higher timeframe trend (price vs 21 EMA on daily)
  3. Pullback to a key level (21 EMA, VWAP proxy, support/resistance)

Run after market close each evening. Logs results to CSV.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION — Adjust these to your preferences
# ─────────────────────────────────────────────
TICKER = "SPY"
IV_RANK_THRESHOLD = 35          # Max IV rank to consider entry (0-100)
EMA_PERIOD = 21                 # Trend EMA period
PULLBACK_TOLERANCE = 0.003      # How close price must be to a level (0.3%)
SUPPORT_RES_LOOKBACK = 20       # Days to look back for support/resistance
SUPPORT_RES_TOUCHES = 2         # Min touches to confirm a level
LEVEL_CLUSTER_PCT = 0.003       # How close levels must be to cluster (0.3%)
CSV_LOG_PATH = "spy_scan_log.csv"


def get_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLCV data."""
    print(f"  Fetching {ticker} price data...")
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def calculate_iv_rank(ticker: str) -> dict:
    """
    Calculate IV Rank using options-implied data from yfinance.
    
    IV Rank = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV) * 100
    
    We estimate current IV from the nearest-expiry options chain,
    and approximate historical IV range from price-based realized vol.
    """
    print("  Calculating IV rank...")
    
    tk = yf.Ticker(ticker)
    
    # Get current implied volatility from nearest expiration
    expirations = tk.options
    if not expirations:
        return {"iv_rank": None, "current_iv": None, "error": "No options data"}
    
    # Find expiration closest to 14-21 days out
    today = datetime.now().date()
    target_dte = 14
    best_exp = None
    best_diff = 999
    
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        diff = (exp_date - today).days
        if 7 <= diff <= 30 and abs(diff - target_dte) < best_diff:
            best_diff = abs(diff - target_dte)
            best_exp = exp_str
    
    if not best_exp:
        best_exp = expirations[0]
    
    chain = tk.option_chain(best_exp)
    calls = chain.calls
    
    # Get ATM implied volatility (strike closest to current price)
    current_price = tk.info.get("regularMarketPrice") or tk.info.get("previousClose")
    if current_price is None:
        hist = tk.history(period="5d")
        current_price = hist["Close"].iloc[-1]
    
    calls = calls[calls["impliedVolatility"] > 0].copy()
    if calls.empty:
        return {"iv_rank": None, "current_iv": None, "error": "No valid IV data"}
    
    calls["dist"] = abs(calls["strike"] - current_price)
    atm_row = calls.loc[calls["dist"].idxmin()]
    current_iv = atm_row["impliedVolatility"]
    
    # Approximate IV rank using historical realized volatility as proxy
    # for the 52-week IV range (imperfect but best we can do with free data)
    hist = yf.download(ticker, period="1y", interval="1d", progress=False)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    
    hist["returns"] = hist["Close"].pct_change()
    
    # Rolling 21-day realized vol, annualized
    hist["realized_vol"] = hist["returns"].rolling(21).std() * np.sqrt(252)
    hist = hist.dropna(subset=["realized_vol"])
    
    rv_high = hist["realized_vol"].max()
    rv_low = hist["realized_vol"].min()
    
    if rv_high == rv_low:
        iv_rank = 50.0
    else:
        iv_rank = ((current_iv - rv_low) / (rv_high - rv_low)) * 100
        iv_rank = max(0, min(100, iv_rank))
    
    return {
        "iv_rank": round(iv_rank, 1),
        "current_iv": round(current_iv * 100, 1),
        "rv_high": round(rv_high * 100, 1),
        "rv_low": round(rv_low * 100, 1),
        "expiration_used": best_exp,
    }


def check_trend(df: pd.DataFrame, ema_period: int = 21) -> dict:
    """Check if price is above or below the EMA for trend direction."""
    print("  Checking trend...")
    df["ema"] = df["Close"].ewm(span=ema_period, adjust=False).mean()
    
    last_close = float(df["Close"].iloc[-1])
    ema_value = float(df["ema"].iloc[-1])
    
    # Also check slope of EMA for trend strength
    ema_prev = float(df["ema"].iloc[-5])
    ema_slope = (ema_value - ema_prev) / ema_prev * 100
    
    if last_close > ema_value:
        trend = "BULLISH"
    else:
        trend = "BEARISH"
    
    return {
        "trend": trend,
        "close": round(last_close, 2),
        "ema_value": round(ema_value, 2),
        "ema_slope_pct": round(ema_slope, 3),
        "distance_from_ema_pct": round((last_close - ema_value) / ema_value * 100, 2),
    }


def find_support_resistance(df: pd.DataFrame, lookback: int = 20, min_touches: int = 2, cluster_pct: float = 0.003) -> list:
    """
    Identify support/resistance levels from recent price action.
    Uses swing highs/lows and clusters nearby levels.
    """
    recent = df.tail(lookback).copy()
    levels = []
    
    highs = recent["High"].values
    lows = recent["Low"].values
    closes = recent["Close"].values
    
    # Find swing highs (local maxima)
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            levels.append(float(highs[i]))
    
    # Find swing lows (local minima)
    for i in range(1, len(lows) - 1):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            levels.append(float(lows[i]))
    
    # Add prior day high/low as potential levels
    levels.append(float(recent["High"].iloc[-2]))
    levels.append(float(recent["Low"].iloc[-2]))
    
    if not levels:
        return []
    
    # Cluster nearby levels
    levels.sort()
    clustered = []
    current_cluster = [levels[0]]
    
    for i in range(1, len(levels)):
        if (levels[i] - current_cluster[0]) / current_cluster[0] <= cluster_pct:
            current_cluster.append(levels[i])
        else:
            if len(current_cluster) >= min_touches:
                clustered.append(round(np.mean(current_cluster), 2))
            current_cluster = [levels[i]]
    
    if len(current_cluster) >= min_touches:
        clustered.append(round(np.mean(current_cluster), 2))
    
    # Also add single strong levels (prior day high/low always count)
    prior_high = round(float(recent["High"].iloc[-2]), 2)
    prior_low = round(float(recent["Low"].iloc[-2]), 2)
    for level in [prior_high, prior_low]:
        if not any(abs(level - c) / c < cluster_pct for c in clustered):
            clustered.append(level)
    
    return sorted(clustered)


def check_pullback(df: pd.DataFrame, trend_info: dict, tolerance: float = 0.003) -> dict:
    """
    Check if price has pulled back to any key level:
      - 21 EMA
      - VWAP proxy (cumulative volume-weighted price of recent session)
      - Support/resistance levels
    """
    print("  Checking pullback levels...")
    
    last_close = trend_info["close"]
    ema_value = trend_info["ema_value"]
    pullback_levels = []
    
    # 1. Check pullback to 21 EMA
    ema_distance = abs(last_close - ema_value) / ema_value
    if ema_distance <= tolerance:
        pullback_levels.append({
            "level": "21 EMA",
            "value": ema_value,
            "distance_pct": round(ema_distance * 100, 2),
        })
    
    # 2. VWAP proxy — use anchored VWAP from last 5 sessions
    recent = df.tail(5).copy()
    typical_price = (recent["High"] + recent["Low"] + recent["Close"]) / 3
    vwap = float((typical_price * recent["Volume"]).sum() / recent["Volume"].sum())
    vwap = round(vwap, 2)
    
    vwap_distance = abs(last_close - vwap) / vwap
    if vwap_distance <= tolerance:
        pullback_levels.append({
            "level": "VWAP (5-day anchored)",
            "value": vwap,
            "distance_pct": round(vwap_distance * 100, 2),
        })
    
    # 3. Support/resistance levels
    sr_levels = find_support_resistance(
        df,
        lookback=SUPPORT_RES_LOOKBACK,
        min_touches=SUPPORT_RES_TOUCHES,
        cluster_pct=LEVEL_CLUSTER_PCT,
    )
    
    for level in sr_levels:
        dist = abs(last_close - level) / level
        if dist <= tolerance:
            pullback_levels.append({
                "level": f"S/R zone",
                "value": level,
                "distance_pct": round(dist * 100, 2),
            })
    
    # Also report nearby levels even if outside tolerance (for awareness)
    nearby_levels = []
    for level in sr_levels:
        dist = abs(last_close - level) / level
        if tolerance < dist <= tolerance * 3:
            nearby_levels.append({
                "level": f"S/R zone (nearby)",
                "value": level,
                "distance_pct": round(dist * 100, 2),
            })
    
    if ema_distance > tolerance and ema_distance <= tolerance * 3:
        nearby_levels.append({
            "level": "21 EMA (nearby)",
            "value": ema_value,
            "distance_pct": round(ema_distance * 100, 2),
        })
    
    return {
        "pullback_detected": len(pullback_levels) > 0,
        "levels_hit": pullback_levels,
        "levels_nearby": nearby_levels,
        "vwap_value": vwap,
        "sr_levels": sr_levels,
    }


def generate_signal(iv_data: dict, trend_info: dict, pullback_info: dict) -> dict:
    """Combine all conditions into a final signal."""
    
    iv_ok = iv_data["iv_rank"] is not None and iv_data["iv_rank"] <= IV_RANK_THRESHOLD
    trend_clear = trend_info["trend"] in ["BULLISH", "BEARISH"]
    pullback_ok = pullback_info["pullback_detected"]
    
    conditions_met = sum([iv_ok, trend_clear, pullback_ok])
    
    if conditions_met == 3:
        signal = "SETUP ACTIVE"
        direction = "CALL debit spread" if trend_info["trend"] == "BULLISH" else "PUT debit spread"
    elif conditions_met == 2:
        signal = "WATCH — 2 of 3 conditions met"
        direction = "CALL side" if trend_info["trend"] == "BULLISH" else "PUT side"
    else:
        signal = "NO SETUP"
        direction = "N/A"
    
    return {
        "signal": signal,
        "direction": direction,
        "conditions_met": conditions_met,
        "iv_pass": iv_ok,
        "trend_pass": trend_clear,
        "pullback_pass": pullback_ok,
    }


def print_report(iv_data: dict, trend_info: dict, pullback_info: dict, signal: dict):
    """Print a clean terminal report."""
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    print("\n")
    print("=" * 60)
    print(f"  SPY DEBIT SPREAD SCANNER — {now}")
    print("=" * 60)
    
    # Signal
    if signal["signal"] == "SETUP ACTIVE":
        icon = ">>>"
    elif "WATCH" in signal["signal"]:
        icon = " >>"
    else:
        icon = "   "
    
    print(f"\n  {icon} {signal['signal']}")
    if signal["direction"] != "N/A":
        print(f"      Direction: {signal['direction']}")
    print(f"      Conditions met: {signal['conditions_met']} / 3")
    
    # Condition 1: IV Rank
    print("\n" + "-" * 60)
    status = "PASS" if signal["iv_pass"] else "FAIL"
    print(f"  [{status}] IV RANK")
    if iv_data["iv_rank"] is not None:
        print(f"      IV Rank:      {iv_data['iv_rank']}  (threshold: <{IV_RANK_THRESHOLD})")
        print(f"      Current IV:   {iv_data['current_iv']}%")
        print(f"      52w RV range: {iv_data['rv_low']}% — {iv_data['rv_high']}%")
        print(f"      Expiration:   {iv_data['expiration_used']}")
    else:
        print(f"      Error: {iv_data.get('error', 'Unknown')}")
    
    # Condition 2: Trend
    print("\n" + "-" * 60)
    status = "PASS" if signal["trend_pass"] else "FAIL"
    print(f"  [{status}] TREND")
    print(f"      Direction:    {trend_info['trend']}")
    print(f"      SPY Close:    ${trend_info['close']}")
    print(f"      21 EMA:       ${trend_info['ema_value']}")
    print(f"      Distance:     {trend_info['distance_from_ema_pct']}%")
    print(f"      EMA slope:    {trend_info['ema_slope_pct']}% (5-day)")
    
    # Condition 3: Pullback
    print("\n" + "-" * 60)
    status = "PASS" if signal["pullback_pass"] else "FAIL"
    print(f"  [{status}] PULLBACK")
    
    if pullback_info["levels_hit"]:
        print(f"      Levels hit:")
        for lvl in pullback_info["levels_hit"]:
            print(f"        - {lvl['level']}: ${lvl['value']} ({lvl['distance_pct']}% away)")
    else:
        print(f"      No levels hit within {PULLBACK_TOLERANCE * 100}% tolerance")
    
    if pullback_info["levels_nearby"]:
        print(f"      Nearby levels (approaching):")
        for lvl in pullback_info["levels_nearby"]:
            print(f"        - {lvl['level']}: ${lvl['value']} ({lvl['distance_pct']}% away)")
    
    print(f"\n      VWAP (5-day): ${pullback_info['vwap_value']}")
    if pullback_info["sr_levels"]:
        print(f"      S/R levels:   {', '.join(f'${l}' for l in pullback_info['sr_levels'])}")
    
    # Action
    print("\n" + "=" * 60)
    if signal["signal"] == "SETUP ACTIVE":
        print(f"  ACTION: Plan {signal['direction']} entry for tomorrow AM")
        print(f"  TIMING: Enter 30-45 min after open with limit order")
        print(f"  RISK:   Max 2-3% of account per trade")
    elif "WATCH" in signal["signal"]:
        print(f"  ACTION: Monitor tomorrow — setup may complete")
        missing = []
        if not signal["iv_pass"]:
            missing.append("IV rank too high")
        if not signal["pullback_pass"]:
            missing.append("no pullback to level yet")
        print(f"  MISSING: {', '.join(missing)}")
    else:
        print(f"  ACTION: No trade. Check again tomorrow evening.")
    print("=" * 60)
    print()


def log_to_csv(iv_data: dict, trend_info: dict, pullback_info: dict, signal: dict):
    """Append scan results to CSV log."""
    
    row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "signal": signal["signal"],
        "direction": signal["direction"],
        "conditions_met": signal["conditions_met"],
        "spy_close": trend_info["close"],
        "iv_rank": iv_data.get("iv_rank"),
        "current_iv": iv_data.get("current_iv"),
        "trend": trend_info["trend"],
        "ema_value": trend_info["ema_value"],
        "distance_from_ema_pct": trend_info["distance_from_ema_pct"],
        "pullback_detected": pullback_info["pullback_detected"],
        "levels_hit": "; ".join(
            f"{l['level']}=${l['value']}" for l in pullback_info["levels_hit"]
        ) if pullback_info["levels_hit"] else "none",
        "vwap": pullback_info["vwap_value"],
    }
    
    df = pd.DataFrame([row])
    
    file_exists = os.path.exists(CSV_LOG_PATH)
    df.to_csv(CSV_LOG_PATH, mode="a", header=not file_exists, index=False)
    print(f"  Logged to {CSV_LOG_PATH}")


def main():
    print("\n  SPY Debit Spread Scanner")
    print("  " + "-" * 30)
    
    # Fetch data
    df = get_price_data(TICKER)
    if df.empty or len(df) < 50:
        print("  ERROR: Could not fetch sufficient price data.")
        return
    
    # Run checks
    iv_data = calculate_iv_rank(TICKER)
    trend_info = check_trend(df, EMA_PERIOD)
    pullback_info = check_pullback(df, trend_info, PULLBACK_TOLERANCE)
    signal = generate_signal(iv_data, trend_info, pullback_info)
    
    # Output
    print_report(iv_data, trend_info, pullback_info, signal)
    log_to_csv(iv_data, trend_info, pullback_info, signal)


if __name__ == "__main__":
    main()