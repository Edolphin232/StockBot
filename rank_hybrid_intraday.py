#!/usr/bin/env python3
"""
Hybrid intraday ranking engine (rules + ML fused).

- Uses ONLY information available up to 10:15 (no leakage)
- Combines:
    * Soft rule signals (liquidity, ATR%, SPY/QQQ alignment, wicks, volume spike)
    * Rule-based direction & quality scores
    * ML continuation probability p_up (10:15 → 10:30)
- Produces:
    * meta_score       → main ranking key
    * bias             → CALL / PUT / NEUTRAL / REJECT
    * p_up             → ML probability of up move
    * ml_direction     → [-1, 1] from p_up
    * rule_direction   → [-1, 1]
    * rule_quality     → [0.3, 2.0]

Output TSV per date:
    data/metrics/YYYY/MM/YYYY-MM-DD/ranked_hybrid.tsv
"""

import os
import sys
import json
import argparse
from datetime import date, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(PROJECT_ROOT, "data", "metrics")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model_weights.json")

# Base filters  (LOOSENED for balanced regime)
MIN_PRICE = 5.0
MAX_PRICE = 1000.0
MIN_PREMARKET_VOL = 10000          # was 20_000
MIN_ATR_PCT = 0.01            # was 0.01
MAX_ATR_PCT = 0.08                # slightly wider

# Rule thresholds
SPY_MIN_ALIGN = 0.0       # not used as hard gate here
VOL_SPIKE_MIN = 0.55       # only affects quality now (no hard reject)
WICK_MAX = 0.6            # wick still penalizes quality, not eligibility
ORB_MIN = 0.0035           # 0.2% min ORB (looser than 0.4%)
ORB_MAX = 0.035            # 6% max ORB (looser than 4%)

# Meta-score thresholds → bias (LOOSER)
CALL_SCORE_THRESHOLD = 0.18
PUT_SCORE_THRESHOLD = -0.18

# Confidence scaling
CONF_SCALE = 1.8

# Cached model
_MODEL_CACHE: Optional[Dict[str, Any]] = None


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def metrics_path_for_date(d: date) -> str:
    year = f"{d.year:04d}"
    month = f"{d.month:02d}"
    dstr = d.isoformat()
    return os.path.join(METRICS_DIR, year, month, dstr, "metrics.tsv")


def load_model_weights() -> Dict[str, Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Missing model file: {MODEL_PATH}")
    with open(MODEL_PATH, "r") as f:
        _MODEL_CACHE = json.load(f)
    print(f"[MODEL] Loaded model from {MODEL_PATH}")
    return _MODEL_CACHE


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def g(row: pd.Series, col: str, default=None):
    if col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return val


# -------------------------------------------------------------------------
# 1) Base filters (price / liquidity / ATR%) — LOOSENED
# -------------------------------------------------------------------------

def apply_base_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    price_930 = pd.to_numeric(df.get("price_930", np.nan), errors="coerce")
    pm_vol = pd.to_numeric(df.get("premarket_volume_60m", 0), errors="coerce")
    atr14 = pd.to_numeric(df.get("atr14", np.nan), errors="coerce")

    df["atr_pct"] = atr14 / price_930.replace(0, np.nan)

    df["price_ok"] = (price_930 >= MIN_PRICE) & (price_930 <= MAX_PRICE)
    df["liquidity_ok"] = pm_vol >= MIN_PREMARKET_VOL
    df["atr_ok"] = df["atr_pct"].between(MIN_ATR_PCT, MAX_ATR_PCT)

    mask = df["price_ok"] & df["liquidity_ok"] & df["atr_ok"]
    out = df[mask].reset_index(drop=True)

    print(f"[BASE] raw={len(df)} after_base={len(out)}")
    return out


# -------------------------------------------------------------------------
# 2) ML directional component (p_up)
# -------------------------------------------------------------------------

def ml_directional_score(df: pd.DataFrame, model: Dict[str, Any]) -> pd.Series:
    feats = model["features"]
    w = model["weights"]
    b = model["bias"]
    mu = model["scaler_mean"]
    scale = model["scaler_scale"]

    lin = np.full(len(df), b, dtype=float)

    for feat in feats:
        x = pd.to_numeric(df.get(feat, 0.0), errors="coerce").astype(float)
        x = x.fillna(mu.get(feat, 0.0))
        s = scale.get(feat, 1.0)
        if s == 0:
            s = 1.0
        z = (x - mu.get(feat, 0.0)) / s
        lin += z * w[feat]

    return sigmoid(lin)


# -------------------------------------------------------------------------
# 3) Rule-based scoring (direction + quality) per row
# -------------------------------------------------------------------------

def rule_scores(row: pd.Series) -> Dict[str, Any]:
    """
    Compute:
        rule_dir_raw     → unbounded directional score (+ bull, - bear)
        rule_qual_raw    → unbounded quality score
        hard_reject      → True only for truly unusable rows
        rule_notes       → text for debugging
    """
    notes = []

    # Core fields
    price_930 = g(row, "price_930")
    price_1000 = g(row, "price_1000")
    price_1015 = g(row, "price_1015")
    orb_hi = g(row, "opening_range_high")
    orb_lo = g(row, "opening_range_low")
    move_930_1000 = g(row, "move_930_to_1000_pct", 0.0)
    rsi = g(row, "rsi14_daily")
    sma20 = g(row, "sma20")
    sma50 = g(row, "sma50")
    atr_pct = g(row, "atr_pct")
    hh = g(row, "higher_highs_9_30_to_10_00", 0.0)
    hl = g(row, "higher_lows_9_30_to_10_00", 0.0)

    # VWAP relationships (may be pct or absolute)
    vwap_open = g(row, "price_1000_vs_opening_vwap_pct",
                  g(row, "price_1000_vs_opening_vwap", 0.0))
    vwap_pm = g(row, "price_1000_vs_premarket_vwap_pct",
                g(row, "price_1000_vs_premarket_vwap", 0.0))

    # Micro features (10:00–10:15)
    ret_1000_1015 = g(row, "return_1000_to_1015_pct", 0.0)
    vol_spike = g(row, "vol_spike_1000_1015_over_930_1000")
    upper_wick = g(row, "upper_wick_930_1015_pct")
    lower_wick = g(row, "lower_wick_930_1015_pct")

    # Market context
    spy_1000_1015 = g(row, "spy_ret_1000_1015_pct", 0.0)
    qqq_1000_1015 = g(row, "qqq_ret_1000_1015_pct", 0.0)

    # ORB stats
    orb_range_frac = None
    if price_930 and orb_hi and orb_lo:
        orb_range_frac = (orb_hi - orb_lo) / price_930

    # ----- Hard reject logic (ONLY for fatal issues) -----
    hard_reject = False

    # Missing essential prices → can't simulate; mark as REJECT
    if price_930 is None or price_1000 is None or price_1015 is None or price_1015 == 0:
        hard_reject = True
        notes.append("missing_price")
        return dict(rule_dir_raw=0.0, rule_qual_raw=0.0,
                    hard_reject=hard_reject, rule_notes=",".join(notes))

    # NOTE: ORB + volume spike are NO LONGER hard rejects.
    # They will be treated as quality adjustments below.
    if orb_range_frac is not None:
        if orb_range_frac < ORB_MIN:
            notes.append("orb_small")
        elif orb_range_frac > ORB_MAX:
            notes.append("orb_large")

    if vol_spike is not None and vol_spike < VOL_SPIKE_MIN:
        notes.append("low_vol_spike")

    # ----- Directional rule score -----
    rule_dir_raw = 0.0

    # Trend from 9:30–10:00
    if move_930_1000 > 0.3:
        rule_dir_raw += 1.0
    if move_930_1000 > 0.7:
        rule_dir_raw += 0.5
    if move_930_1000 < -0.3:
        rule_dir_raw -= 1.0
    if move_930_1000 < -0.7:
        rule_dir_raw -= 0.5

    # Intraday micro trend 10:00–10:15
    if ret_1000_1015 > 0.15:
        rule_dir_raw += 1.0
    if ret_1000_1015 > 0.4:
        rule_dir_raw += 0.5
    if ret_1000_1015 < -0.15:
        rule_dir_raw -= 1.0
    if ret_1000_1015 < -0.4:
        rule_dir_raw -= 0.5

    # VWAP alignment
    if vwap_open > 0:
        rule_dir_raw += 0.5
    if vwap_open > 0.3:
        rule_dir_raw += 0.5
    if vwap_open < 0:
        rule_dir_raw -= 0.5
    if vwap_open < -0.3:
        rule_dir_raw -= 0.5

    if vwap_pm > 0:
        rule_dir_raw += 0.5
    if vwap_pm < 0:
        rule_dir_raw -= 0.5

    # HH / HL structure
    if hh > hl:
        rule_dir_raw += 0.5
    if hl > hh:
        rule_dir_raw -= 0.5

    # Daily trend (20/50 SMA + RSI)
    if sma20 and sma50 and sma20 > sma50:
        rule_dir_raw += 0.5
    if sma20 and sma50 and sma20 < sma50:
        rule_dir_raw -= 0.5

    if price_930 and sma20 and price_930 > sma20:
        rule_dir_raw += 0.25
    if price_930 and sma20 and price_930 < sma20:
        rule_dir_raw -= 0.25

    if rsi and rsi > 60:
        rule_dir_raw += 0.5
    if rsi and rsi < 40:
        rule_dir_raw -= 0.5

    # Market alignment (SPY/QQQ 10:00–10:15)
    avg_mkt = 0.5 * spy_1000_1015 + 0.5 * qqq_1000_1015
    if avg_mkt > 0.15:
        rule_dir_raw += 0.5
    if avg_mkt < -0.15:
        rule_dir_raw -= 0.5

    # Wick logic → quality penalty, not eligibility
    wick_penalty = 0.0
    if upper_wick is not None and upper_wick > WICK_MAX:
        wick_penalty -= 1.0
        notes.append("large_upper_wick")
    if lower_wick is not None and lower_wick > WICK_MAX:
        wick_penalty -= 1.0
        notes.append("large_lower_wick")

    # ----- Quality rule score -----
    rule_qual_raw = 0.0

    # ATR regime: some volatility is good, too much is bad
    if atr_pct and atr_pct > 0.015:
        rule_qual_raw += 0.5
    if atr_pct and atr_pct > 0.04:
        rule_qual_raw -= 0.25  # too wild

    # ORB sweet spot around 1–3%
    if orb_range_frac is not None:
        if 0.01 <= orb_range_frac <= 0.03:
            rule_qual_raw += 0.75
        elif 0.005 <= orb_range_frac < 0.01 or 0.03 < orb_range_frac <= 0.05:
            rule_qual_raw += 0.25
        else:
            rule_qual_raw -= 0.25

    # Volume spike is good quality
    if vol_spike is not None:
        if vol_spike > 1.0:
            rule_qual_raw += 0.75
        elif vol_spike > 0.7:
            rule_qual_raw += 0.25
        else:
            rule_qual_raw -= 0.25

    # Wick penalty applied to quality
    rule_qual_raw += wick_penalty

    return dict(
        rule_dir_raw=rule_dir_raw,
        rule_qual_raw=rule_qual_raw,
        hard_reject=hard_reject,
        rule_notes=",".join(notes),
    )


# -------------------------------------------------------------------------
# 4) Hybrid fusion: rules + ML → meta scores
# -------------------------------------------------------------------------

def apply_hybrid_scoring(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    model = load_model_weights()

    # ML layer
    df["p_up"] = ml_directional_score(df, model)
    df["ml_direction"] = 2.0 * df["p_up"] - 1.0  # [-1, 1]

    # Rule layer
    rule_res = df.apply(rule_scores, axis=1, result_type="expand")
    df["rule_dir_raw"] = rule_res["rule_dir_raw"]
    df["rule_qual_raw"] = rule_res["rule_qual_raw"]
    df["hard_reject"] = rule_res["hard_reject"]
    df["rule_notes"] = rule_res["rule_notes"]

    # Normalize rule direction to [-1, 1] via tanh
    df["rule_direction"] = np.tanh(df["rule_dir_raw"] / 3.0)

    # Normalize quality to [0.3, 2.0]
    df["rule_quality"] = 1.0 + 0.6 * np.tanh(df["rule_qual_raw"] / 3.0)
    df["rule_quality"] = df["rule_quality"].clip(0.3, 2.0)

    # Fusion coefficients
    alpha_dir = 0.6   # ML direction weight
    beta_dir = 0.4    # Rule direction weight

    # Meta-direction: blend ML + rules
    df["meta_direction"] = alpha_dir * df["ml_direction"] + beta_dir * df["rule_direction"]
    df["meta_direction"] = df["meta_direction"].clip(-1.0, 1.0)

    # Meta-quality = just rule_quality for now
    df["meta_quality"] = df["rule_quality"]

    # Final meta score
    df["meta_score"] = df["meta_direction"] * df["meta_quality"]

    # Apply hard rejections: zero out score, mark bias=REJECT later
    df.loc[df["hard_reject"], ["meta_score", "meta_direction"]] = 0.0

    # Confidence
    df["confidence"] = np.minimum(1.0, np.abs(df["meta_score"]) / CONF_SCALE)

    # Directional bias
    bias = []
    for _, r in df.iterrows():
        if r.get("hard_reject", False):
            bias.append("REJECT")
        else:
            s = r["meta_score"]
            if s >= CALL_SCORE_THRESHOLD:
                bias.append("CALL")
            elif s <= PUT_SCORE_THRESHOLD:
                bias.append("PUT")
            else:
                bias.append("NEUTRAL")
    df["bias"] = bias

    df["abs_meta_score"] = df["meta_score"].abs()
    return df


# -------------------------------------------------------------------------
# 5) Main ranking functions
# -------------------------------------------------------------------------

def rank_for_date(d: date) -> pd.DataFrame:
    path = metrics_path_for_date(d)
    if not os.path.exists(path):
        print(f"[WARN] metrics file missing for {d}: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, sep="\t")
    if df.empty:
        print(f"[WARN] metrics empty for {d}")
        return df

    print(f"[RANK] {d} raw_rows={len(df)}")
    df = apply_base_filters(df)
    if df.empty:
        print(f"[INFO] No rows after base filters for {d}")
        return df

    df = apply_hybrid_scoring(df)

    # Columns useful for backtesting:
    keep = [
        "symbol",
        "bias",
        "meta_score",
        "abs_meta_score",
        "confidence",
        "p_up",
        "ml_direction",
        "rule_direction",
        "rule_quality",
        "rule_dir_raw",
        "rule_qual_raw",
        "hard_reject",
        "rule_notes",
        # prices/returns
        "price_930",
        "price_1000",
        "price_1015",
        "price_1030",
        "return_1015_to_1030_pct",
        "return_1000_to_1015_pct",
        # context
        "premarket_pct_change",
        "move_930_to_1000_pct",
        "premarket_volume_60m",
        "relative_volume_20d",
        "atr14",
        "atr_pct",
        "spy_ret_1000_1015_pct",
        "qqq_ret_1000_1015_pct",
        "vol_spike_1000_1015_over_930_1000",
        "upper_wick_930_1015_pct",
        "lower_wick_930_1015_pct",
    ]

    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out = out.sort_values("abs_meta_score", ascending=False).reset_index(drop=True)

    print(f"[RANK] {d} ranked_rows={len(out)} "
          f"calls={sum(out['bias'] == 'CALL')} puts={sum(out['bias'] == 'PUT')} "
          f"neut={sum(out['bias'] == 'NEUTRAL')} rej={sum(out['bias'] == 'REJECT')}")
    return out


def run_single_date(d: date):
    print(f"=== Ranking {d} ===")
    ranked = rank_for_date(d)
    if ranked.empty:
        print(f"[INFO] Nothing to save for {d}")
        return

    out_dir = os.path.dirname(metrics_path_for_date(d))
    out_path = os.path.join(out_dir, "ranked_hybrid.tsv")
    os.makedirs(out_dir, exist_ok=True)
    ranked.to_csv(out_path, sep="\t", index=False)
    print(f"[SAVE] → {out_path}")


def run_range(start_date: date, end_date: date):
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:
            try:
                run_single_date(cur)
            except Exception as e:
                print(f"[ERROR] {cur}: {e}")
        else:
            print(f"[SKIP] {cur} (weekend)")
        cur += timedelta(days=1)
    print("[DONE]")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Hybrid ML + rules intraday ranking.")
    p.add_argument("-d", "--date")
    p.add_argument("--start")
    p.add_argument("--end")
    args = p.parse_args()

    if args.date and not args.start:
        d = date.fromisoformat(args.date)
        run_single_date(d)
        sys.exit(0)

    if args.start:
        s = date.fromisoformat(args.start)
        e = date.fromisoformat(args.end) if args.end else s
        run_range(s, e)
        sys.exit(0)

    run_single_date(date.today())


if __name__ == "__main__":
    main()
