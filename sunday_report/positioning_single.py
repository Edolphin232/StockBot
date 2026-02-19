#!/usr/bin/env python3
"""
Earnings Options Trade Setup Analyzer
Combines dealer gamma, put/call, IV, and price acceptance into a complete trade plan
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import math
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# ----------------------------
# Setup
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
TRADIER_KEY = os.getenv("TRADIER_API_KEY")

if not FINNHUB_KEY:
    raise RuntimeError("FINNHUB_API_KEY not set")
if not TRADIER_KEY:
    raise RuntimeError("TRADIER_API_KEY not set")

# Data directory path
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FINNHUB = "https://finnhub.io/api/v1"
TRADIER = "https://api.tradier.com/v1"
TRADIER_HEADERS = {"Authorization": f"Bearer {TRADIER_KEY}", "Accept": "application/json"}

MAX_RETRIES = 3
REQUEST_TIMEOUT = 20

# ----------------------------
# API wrappers
# ----------------------------
def finnhub_get(path, params=None):
    params = dict(params or {})
    params["token"] = FINNHUB_KEY
    for _ in range(MAX_RETRIES):
        r = requests.get(FINNHUB + path, params=params, timeout=REQUEST_TIMEOUT)
        if r.ok:
            return r.json()
        time.sleep(1)
    r.raise_for_status()

def tradier_get(path, params=None):
    for _ in range(MAX_RETRIES):
        r = requests.get(TRADIER + path, headers=TRADIER_HEADERS, params=params or {}, timeout=REQUEST_TIMEOUT)
        if r.ok:
            return r.json()
        time.sleep(1)
    r.raise_for_status()

# ----------------------------
# Price context (prior value)
# ----------------------------
def get_price_context(symbol):
    data = tradier_get("/markets/history", {"symbol": symbol, "interval": "daily"})
    bars = data["history"]["day"]
    if isinstance(bars, dict):
        bars = [bars]
    last_bar = bars[-1]
    prev_bar = bars[-2]
    last_price = float(last_bar["close"])
    prev_typical = (float(prev_bar["high"]) + float(prev_bar["low"]) + float(prev_bar["close"])) / 3
    return last_price, prev_typical

# ----------------------------
# Expiration selection
# ----------------------------
def pick_expiration_after(symbol, earnings_date):
    ed = datetime.strptime(earnings_date, "%Y-%m-%d").date()
    dates = tradier_get("/markets/options/expirations", {"symbol": symbol})["expirations"]["date"]
    if isinstance(dates, str):
        dates = [dates]
    for d in dates:
        if datetime.strptime(d, "%Y-%m-%d").date() >= ed:
            return d
    return dates[0]

# ----------------------------
# Positioning (Put/Call + GEX)
# ----------------------------
def get_positioning(symbol, expiration, spot):
    chain = tradier_get("/markets/options/chains", {"symbol": symbol, "expiration": expiration, "greeks": "true"})
    options = chain["options"]["option"]
    if isinstance(options, dict):
        options = [options]

    calls = puts = 0
    gamma = 0
    total_options = len(options)
    skipped_no_oi = 0
    skipped_no_gamma = 0
    processed_calls = 0
    processed_puts = 0

    for o in options:
        oi = o.get("open_interest", 0)
        g = o.get("greeks", {}).get("gamma", 0)
        if oi == 0:
            skipped_no_oi += 1
            continue
        if g == 0:
            skipped_no_gamma += 1
            continue
        if o["option_type"] == "call":
            calls += oi
            gamma -= g * oi * 100 * spot
            processed_calls += 1
        else:
            puts += oi
            gamma += g * oi * 100 * spot
            processed_puts += 1

    put_call = puts / calls if calls else 1
    gex = gamma / 1e6
    
    # Log diagnostic info if GEX is near zero
    if abs(gex) < 1:
        logger.info(f"GEX near zero for {symbol}: total_options={total_options}, "
                   f"processed_calls={processed_calls}, processed_puts={processed_puts}, "
                   f"skipped_no_oi={skipped_no_oi}, skipped_no_gamma={skipped_no_gamma}, "
                   f"raw_gamma={gamma:.2f}")
    
    return put_call, gex

# ----------------------------
# ATM IV + expected move
# ----------------------------
import math

def get_atm_iv_and_em(symbol, expiration, spot):
    chain = tradier_get("/markets/options/chains", {
        "symbol": symbol,
        "expiration": expiration,
        "greeks": "true"
    })

    options = chain.get("options", {}).get("option", [])
    if isinstance(options, dict):
        options = [options]
    if not options:
        raise ValueError("No options chain returned")

    # find ATM strike
    strikes = sorted({float(o["strike"]) for o in options if o.get("strike")})
    atm_strike = min(strikes, key=lambda k: abs(k - spot))

    call = put = None
    for o in options:
        if float(o.get("strike", 0)) == atm_strike:
            if o["option_type"] == "call":
                call = o
            elif o["option_type"] == "put":
                put = o

    if not call or not put:
        raise ValueError("ATM call or put missing")

    # --- Try implied volatility first ---
    iv = None
    if call.get("implied_volatility"):
        iv = float(call["implied_volatility"])
    elif put.get("implied_volatility"):
        iv = float(put["implied_volatility"])

    # --- Fallback: infer IV from straddle ---
    if iv is None:
        call_mid = (float(call["bid"]) + float(call["ask"])) / 2
        put_mid = (float(put["bid"]) + float(put["ask"])) / 2
        straddle = call_mid + put_mid

        # time to expiration
        exp = datetime.strptime(expiration, "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        T = max((exp - today).days, 1) / 365

        # expected move ≈ straddle
        em = straddle
        iv = em / (spot * math.sqrt(T))

    else:
        exp = datetime.strptime(expiration, "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        T = max((exp - today).days, 1) / 365
        em = spot * iv * math.sqrt(T)

    return iv, em, int(T * 365)


# ----------------------------
# Strike selection
# ----------------------------
def pick_strike(symbol, expiration, spot, bias):
    chain = tradier_get("/markets/options/chains", {"symbol": symbol, "expiration": expiration})
    options = chain["options"]["option"]
    if isinstance(options, dict):
        options = [options]

    want = "call" if bias == "CALL" else "put"
    strikes = sorted({float(o["strike"]) for o in options if o["option_type"] == want})

    if bias == "CALL":
        above = [k for k in strikes if k >= spot]
        return above[0] if above else min(strikes, key=lambda k: abs(k - spot))
    else:
        below = [k for k in strikes if k <= spot]
        return below[-1] if below else min(strikes, key=lambda k: abs(k - spot))

# ----------------------------
# Stops + targets
# ----------------------------
def compute_levels(bias, price, ref, em):
    buf = 0.25 * em
    if bias == "CALL":
        return ref - buf, price + 0.6 * em, price + 1.0 * em
    if bias == "PUT":
        return ref + buf, price - 0.6 * em, price - 1.0 * em
    return None, None, None

def explain_decision(bias, gex, put_call, price, ref, iv, expected_move, spot):
    """
    Generate a complete human-readable explanation of:
    - directional bias
    - entry timing
    - IV / pricing risk
    - structure (option vs spread)
    """

    parts = []

    # --- Dealer positioning ---
    if bias == "CALL":
        parts.append(
            f"Dealer gamma is strongly negative (GEX {gex:+.1f}M), meaning dealers are short calls and would need to buy stock if price rises."
        )
    elif bias == "PUT":
        parts.append(
            f"Dealer gamma is strongly positive (GEX {gex:+.1f}M), meaning dealers are short puts and would need to sell stock if price falls."
        )
    else:
        parts.append(
            f"Dealer gamma is near neutral (GEX {gex:+.1f}M), so there is no strong hedging pressure to amplify moves."
        )

    # --- Sentiment skew ---
    if put_call < 0.7:
        parts.append(f"Put/Call OI ({put_call:.2f}) shows traders are positioned bullish.")
    elif put_call > 1.3:
        parts.append(f"Put/Call OI ({put_call:.2f}) shows traders are positioned defensively.")
    else:
        parts.append(f"Put/Call OI ({put_call:.2f}) is balanced.")

    # --- Price acceptance ---
    if price > ref:
        parts.append(
            f"Price ({price:.2f}) is trading above prior value ({ref:.2f}), indicating buyers are in control."
        )
    else:
        parts.append(
            f"Price ({price:.2f}) is trading below prior value ({ref:.2f}), indicating sellers are in control."
        )

    # --- IV / expected move ---
    em_pct = expected_move / spot
    if em_pct < 0.025:
        parts.append(
            f"The market is only pricing a {em_pct*100:.1f}% move via options, which is likely too small to overcome IV crush."
        )
    else:
        parts.append(
            f"Options are pricing a ±{em_pct*100:.1f}% move into earnings, so a directional break could pay."
        )

    if iv > 0.65:
        parts.append(
            f"Implied volatility is elevated ({iv*100:.0f}%), so defined-risk spreads are preferred over naked options."
        )
    else:
        parts.append(
            f"Implied volatility ({iv*100:.0f}%) is reasonable for taking directional options."
        )

    return " ".join(parts)



# ----------------------------
# Main analysis
# ----------------------------
def main():
    symbol = sys.argv[1].upper()
    earnings_path = DATA_DIR / "weekly_earnings.csv"
    df = pd.read_csv(earnings_path)
    earnings_date = df[df["Ticker"] == symbol].iloc[0]["Date"]

    try:
        quote_data = finnhub_get("/quote", {"symbol": symbol})
        spot = quote_data.get("c") if quote_data else None
    except Exception as e:
        logger.error(f"Failed to get spot price: {e}")
        spot = None

    try:
        expiration = pick_expiration_after(symbol, earnings_date)
    except Exception as e:
        logger.error(f"Failed to get expiration: {e}")
        expiration = None

    try:
        if spot is not None and expiration is not None:
            put_call, gex = get_positioning(symbol, expiration, spot)
        else:
            put_call, gex = None, None
    except Exception as e:
        logger.error(f"Failed to get positioning: {e}")
        put_call, gex = None, None

    try:
        price, ref = get_price_context(symbol)
    except Exception as e:
        logger.error(f"Failed to get price context: {e}")
        price, ref = None, None

    try:
        if spot is not None and expiration is not None:
            iv, em, dte = get_atm_iv_and_em(symbol, expiration, spot)
        else:
            iv, em, dte = None, None, None
    except Exception as e:
        logger.error(f"Failed to get IV and expected move: {e}")
        iv, em, dte = None, None, None

    if gex is not None:
        if gex < -10:
            bias = "CALL"
        elif gex > 10:
            bias = "PUT"
        else:
            bias = "AVOID"
    else:
        bias = "AVOID"

    if bias in ("CALL", "PUT") and spot is not None and expiration is not None:
        try:
            strike = pick_strike(symbol, expiration, spot, bias)
            if price is not None and ref is not None and em is not None:
                stop, tp1, tp2 = compute_levels(bias, price, ref, em)
            else:
                stop, tp1, tp2 = None, None, None
        except Exception as e:
            logger.error(f"Failed to get strike/levels: {e}")
            strike = stop = tp1 = tp2 = None
    else:
        strike = stop = tp1 = tp2 = None

    if price is not None and ref is not None:
        action = "BUY NOW" if (bias == "CALL" and price > ref) or (bias == "PUT" and price < ref) else "WAIT"
    else:
        action = "WAIT"

    if all(v is not None for v in [bias, gex, put_call, price, ref, iv, em, spot]):
        explanation = explain_decision(bias, gex, put_call, price, ref, iv, em, spot)
    else:
        explanation = "Unable to generate explanation due to missing data"

    # Format contract string
    if strike is not None and expiration is not None:
        contract_str = f"{symbol} {expiration} {strike}{'C' if bias=='CALL' else 'P'}"
    else:
        contract_str = "N/A"
    
    # Helper functions to format values safely
    def fmt_float(val, fmt=".2f"):
        if val is None:
            return "N/A"
        try:
            return f"{val:{fmt}}"
        except (TypeError, ValueError):
            return "N/A"
    
    def fmt_int(val):
        if val is None:
            return "N/A"
        try:
            return str(int(val))
        except (TypeError, ValueError):
            return "N/A"
    
    def fmt_iv(val):
        if val is None or val == 0:
            return "N/A"
        try:
            return f"{val * 100:.1f}%"
        except (TypeError, ValueError):
            return "N/A"
    
    print(f"""
        Ticker:     {symbol}
        Earnings:   {earnings_date}
        Expiration: {expiration or 'N/A'}
        Spot:       {fmt_float(spot)}
        Price:      {fmt_float(price)}
        Ref Price:  {fmt_float(ref)}

        Put/Call:   {fmt_float(put_call)}
        GEX ($M):   {fmt_float(gex, '+.3f') if gex is not None else 'N/A'}
        ATM IV:     {fmt_iv(iv)}
        Exp Move:   ±{fmt_float(em)}
        DTE:        {fmt_int(dte)}

        Bias:       {bias}
        Action:     {action}

        Contract:   {contract_str}
        Stop:       {fmt_float(stop)}
        TP1:        {fmt_float(tp1)}
        TP2:        {fmt_float(tp2)}

        Explanation: {explanation}
        """)

if __name__ == "__main__":
    main()
