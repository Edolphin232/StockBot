#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

INPUT_PATH  = "data/metrics/premarket_metrics.tsv"
OUTPUT_PATH = "data/metrics/ranked_candidates.tsv"


# -------------------------------------------------------------------
# Load & basic filters
# -------------------------------------------------------------------

def load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df


def apply_base_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to:
      - decent premarket liquidity
      - price high enough for options
      - enough volatility (ATR%)
    """
    df = df.copy()

    # avoid divide-by-zero
    df["atr_pct"] = df["atr14"] / df["price_930"].replace(0, np.nan)

    cond = (
        (df["premarket_volume_60m"] >= 10_000) &        # liquid in premarket
        (df["price_930"] >= 10) &                       # avoid penny/illiquid
        (df["atr_pct"].between(0.02, 0.10))             # 2–10% daily ATR
    )

    filtered = df[cond].reset_index(drop=True)
    return filtered


# -------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------

def score_row(row):
    """
    Compute a single directional score:
      >0  => CALL bias
      <0  => PUT bias
      ~0  => NEUTRAL

    Also returns a short natural-language explanation.
    """
    bull = 0
    bear = 0
    reasons = []

    pm = row.get("premarket_pct_change", np.nan)
    op = row.get("move_930_to_1000_pct", np.nan)
    vs_pm_vwap = row.get("price_1000_vs_premarket_vwap", np.nan)
    vs_open_vwap = row.get("price_1000_vs_opening_vwap", np.nan)
    hh = row.get("higher_highs_9_30_to_10_00", np.nan)
    hl = row.get("higher_lows_9_30_to_10_00", np.nan)
    rsi = row.get("rsi14_daily", np.nan)
    atr_pct = row.get("atr_pct", np.nan)

    # --- 1. Premarket direction (minor weight) ---
    if not np.isnan(pm):
        if pm > 0.5:
            bull += 1
            reasons.append("gapped up in premarket")
        if pm > 1.5:
            bull += 1
            reasons.append("strong premarket gap up")
        if pm < -0.5:
            bear += 1
            reasons.append("gapped down in premarket")
        if pm < -1.5:
            bear += 1
            reasons.append("strong premarket gap down")

    # --- 2. Opening follow-through 9:30–10:00 (strongest signal) ---
    if not np.isnan(op):
        if op > 0.3:
            bull += 3
            reasons.append("strong upside move 9:30–10:00")
        if op > 1.0:
            bull += 2
            reasons.append("very strong upside continuation after open")
        if op < -0.3:
            bear += 3
            reasons.append("strong downside move 9:30–10:00")
        if op < -1.0:
            bear += 2
            reasons.append("very strong downside continuation after open")

    # --- 3. VWAP alignment (who is in control) ---
    if not np.isnan(vs_pm_vwap):
        if vs_pm_vwap > 0:
            bull += 1
            reasons.append("trading above premarket VWAP")
        elif vs_pm_vwap < 0:
            bear += 1
            reasons.append("trading below premarket VWAP")

    if not np.isnan(vs_open_vwap):
        if vs_open_vwap > 0:
            bull += 1
            reasons.append("trading above opening VWAP")
        elif vs_open_vwap < 0:
            bear += 1
            reasons.append("trading below opening VWAP")

    # --- 4. Intraday structure (higher highs / higher lows) ---
    if not np.isnan(hh) and not np.isnan(hl):
        # require at least a small edge to avoid noise
        if hh >= hl + 2:
            bull += 1
            reasons.append("more higher highs than higher lows intraday")
        elif hl >= hh + 2:
            bear += 1
            reasons.append("more higher lows broken than made intraday")

    # --- 5. Daily RSI context ---
    if not np.isnan(rsi):
        if 40 <= rsi <= 65:
            # trending but not cooked – only helps if already bullish
            if bull > bear:
                bull += 1
                reasons.append("daily RSI supports bullish trend")
        if rsi > 70:
            bear += 1
            reasons.append("daily RSI overbought (extended upside)")
        if rsi < 35:
            bear += 1
            reasons.append("daily RSI weak/oversold")
        if rsi < 30:
            bear += 1
            reasons.append("daily RSI deeply oversold (strong downtrend risk)")

    # --- 6. Volatility bonus (only in the winning direction) ---
    if not np.isnan(atr_pct) and 0.02 <= atr_pct <= 0.10:
        if bull > bear:
            bull += 1
            reasons.append("good intraday volatility for upside")
        elif bear > bull:
            bear += 1
            reasons.append("good intraday volatility for downside")

    final_score = bull - bear

    # Keep explanation short-ish
    if not reasons:
        explanation = "no strong directional edge"
    else:
        # de-duplicate and cap length
        uniq = []
        for r in reasons:
            if r not in uniq:
                uniq.append(r)
        explanation = "; ".join(uniq[:6])

    return final_score, explanation


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    scores = df.apply(score_row, axis=1, result_type="expand")
    df["score"] = scores[0]
    df["reason"] = scores[1]

    # Directional bias label
    def bias_from_score(s):
        if s > 0:
            return "CALL"
        elif s < 0:
            return "PUT"
        else:
            return "NEUTRAL"

    df["bias"] = df["score"].apply(bias_from_score)
    df["abs_score"] = df["score"].abs()

    return df


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    df = load_metrics(INPUT_PATH)
    print(f"[INFO] Loaded {len(df)} rows from {INPUT_PATH}")

    df = apply_base_filters(df)
    print(f"[INFO] After base filters: {len(df)} rows")

    if df.empty:
        print("[WARN] No rows after filtering; exiting.")
        return

    df = rank_candidates(df)

    # Save full table
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"[INFO] Saved ranked candidates to {OUTPUT_PATH}")

    # Top 10 CALL and PUT candidates
    calls = df[df["bias"] == "CALL"].sort_values("score", ascending=False).head(10)
    puts  = df[df["bias"] == "PUT"].sort_values("score", ascending=True).head(10)

    print("\n=== Top 10 CALL candidates ===")
    if calls.empty:
        print("  (none)")
    else:
        print(calls[["symbol", "score", "premarket_pct_change",
                     "move_930_to_1000_pct", "price_1000_vs_premarket_vwap",
                     "rsi14_daily", "atr14", "reason"]])

    print("\n=== Top 10 PUT candidates ===")
    if puts.empty:
        print("  (none)")
    else:
        print(puts[["symbol", "score", "premarket_pct_change",
                    "move_930_to_1000_pct", "price_1000_vs_premarket_vwap",
                    "rsi14_daily", "atr14", "reason"]])


if __name__ == "__main__":
    main()
