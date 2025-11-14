#!/usr/bin/env python3
import os
import sys
import argparse
import time
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
import pytz

# Ensure project root is on sys.path when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranker.news_fetcher import fetch_news
from ranker.scorer import SentimentScorer
from ranker.config import NEWS_LOOKBACK_DAYS, NEWS_MAX_ARTICLES


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def score_news(symbol: str, scorer: SentimentScorer, days: int, max_articles: int) -> tuple[float, int]:
    articles = fetch_news(symbol, days=days, max_articles=max_articles)
    if not articles:
        return (0.0, 0)
    scores = scorer.score_articles(articles)
    if scores:
        abs_threshold = 0.05
        significant = [(s, abs(s)) for s in scores if abs(s) >= abs_threshold]
        if significant:
            num = sum(s * w for s, w in significant)
            den = sum(w for _, w in significant) or 1.0
            sentiment = float(num / den)
        else:
            sentiment = float(sum(scores) / len(scores))
    else:
        sentiment = 0.0
    return (sentiment, len(articles))


def calculate_signal_score(row: pd.Series) -> float:
    """
    Composite score using daily indicators (MACD, Bollinger Bands, RSI) 
    validated by pre-market momentum and volume.
    
    Weights:
    - Pre-market movement: ±40 points (35-40%)
    - MACD (daily 12/26/9): ±25 points (20-25%)
    - Bollinger Bands (daily 20, 2-std): ±25 points (20-25%)
    - RSI (daily 14): ±15 points (15-20%)
    - Volume ratio: ±10 points (10-15%)
    
    Raw range: -115 to +115, normalized to -100 to +100.
    """
    score = 0.0
    
    # ============================================================================
    # PRE-MARKET MOVEMENT (±40 points) - 35-40% weight
    # ============================================================================
    pct_change = _safe_float(row.get("premarket_pct_change"))
    if not np.isnan(pct_change):
        if pct_change >= 5:
            score += 40
        elif pct_change >= 3:
            score += 28
        elif pct_change >= 1.5:
            score += 15
        elif pct_change >= 0.5:
            score += 5
        elif pct_change <= -5:
            score -= 40
        elif pct_change <= -3:
            score -= 28
        elif pct_change <= -1.5:
            score -= 15
        elif pct_change <= -0.5:
            score -= 5
    
    # ============================================================================
    # MACD HISTOGRAM (±25 points) - 20-25% weight
    # Daily MACD 12/26/9 - trend momentum validator
    # ============================================================================
    macd_hist = _safe_float(row.get("macd_hist"))
    if not np.isnan(macd_hist):
        # Strong bullish momentum
        if macd_hist > 0.5:
            score += 25
        # Moderate bullish momentum
        elif macd_hist > 0.1:
            score += 12
        # Strong bearish momentum
        elif macd_hist < -0.5:
            score -= 25
        # Moderate bearish momentum
        elif macd_hist < -0.1:
            score -= 12
    
    # ============================================================================
    # BOLLINGER BANDS (±25 points) - 20-25% weight
    # Daily 20-period, 2 standard deviations
    # ============================================================================
    price = _safe_float(row.get("premarket_price"))
    bb_lower = _safe_float(row.get("bb_lower"))
    bb_upper = _safe_float(row.get("bb_upper"))
    
    if not np.isnan(price) and not np.isnan(bb_lower) and not np.isnan(bb_upper):
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_position = (price - bb_lower) / bb_range  # 0 = lower band, 1 = upper band
            
            # Price at lower band (oversold)
            if bb_position < 0.15:
                score += 25
            elif bb_position < 0.35:
                score += 12
            # Price at upper band (overbought)
            elif bb_position > 0.85:
                score -= 25
            elif bb_position > 0.65:
                score -= 12
    
    # ============================================================================
    # RSI (±15 points) - 15-20% weight
    # Daily 14-period RSI
    # ============================================================================
    rsi = _safe_float(row.get("rsi14"))
    if not np.isnan(rsi):
        # Strong oversold
        if rsi < 25:
            score += 15
        # Moderate oversold
        elif rsi < 35:
            score += 8
        # Strong overbought
        elif rsi > 75:
            score -= 15
        # Moderate overbought
        elif rsi > 65:
            score -= 8
    
    # ============================================================================
    # VOLUME RATIO (±10 points) - 10-15% weight
    # Pre-market volume / 20-day average
    # Volume amplifies other signals; minimal standalone contribution
    # ============================================================================
    avg_vol = _safe_float(row.get("avg_volume_20d"))
    prev_vol = _safe_float(row.get("prev_volume_1d"))
    
    if not np.isnan(avg_vol) and not np.isnan(prev_vol) and avg_vol > 0:
        vol_ratio = prev_vol / avg_vol
        
        # Very high volume
        if vol_ratio > 2.5:
            score += 10
        # High volume
        elif vol_ratio > 1.75:
            score += 5
        # Very low volume
        elif vol_ratio < 0.4:
            score -= 10
        # Low volume
        elif vol_ratio < 0.6:
            score -= 5
    
    # ============================================================================
    # NORMALIZE TO -100 to +100 RANGE
    # ============================================================================
    # Raw range is -115 to +115
    normalized_score = (score / 115.0) * 100.0
    
    # Clamp to ensure we stay within bounds
    normalized_score = np.clip(normalized_score, -100, 100)
    
    return float(normalized_score)



def consolidate(df: pd.DataFrame) -> pd.DataFrame:
    # Volume ratio
    if "prev_volume_1d" in df.columns and "avg_volume_20d" in df.columns:
        avg = pd.to_numeric(df["avg_volume_20d"], errors="coerce").replace(0, np.nan)
        prev = pd.to_numeric(df["prev_volume_1d"], errors="coerce")
        df["volume_ratio"] = prev / avg
    cols = ["symbol", "signal_score", "signal_label", "news_sentiment", "volume_ratio"]
    return df[[c for c in cols if c in df.columns]].copy()


def _default_premarket_csv_path() -> str:
    eastern = pytz.timezone("US/Eastern")
    date_str = datetime.now(tz=eastern).strftime("%Y-%m-%d")
    return os.path.join(PROJECT_ROOT, "data", "premarket", f"{date_str}.csv")


def main():
    parser = argparse.ArgumentParser(description="Analyze all stocks: compute score for all, then fetch news for selected top/bottom, and write outputs.")
    parser.add_argument("--out-full", default=None, help="Path to write full analysis TSV (default: <input>_analysis.tsv).")
    parser.add_argument("--out-top", default=None, help="Path to write top/bottom selection TSV (default: <input>_topbottom.tsv).")
    # Intraday-friendly defaults: focus on a wider selection, recent news, few articles
    parser.add_argument("--top", type=int, default=10, help="Number of top and bottom to keep (default: 10).")
    parser.add_argument("--days", type=int, default=1, help="News lookback window in days (default: 1).")
    parser.add_argument("--max-articles", type=int, default=8, help="Max news articles per symbol (default: 8).")
    # Use a proper flag; do NOT use type=bool which treats any non-empty string as True
    parser.add_argument("--news", action="store_true", help="Fetch news for selected top/bottom.")
    parser.add_argument(
        "--min-volume-ratio",
        type=float,
        default=0.1,
        help="Minimum prev/avg volume ratio for selection (top/bottom only; default: 0.1).",
    )
    args = parser.parse_args()

    csv_path = _default_premarket_csv_path()
    df = pd.read_csv(csv_path)
    if df.empty or "symbol" not in df.columns:
        raise RuntimeError("Input CSV must contain a 'symbol' column.")

    # Compute volume ratio for reference; do NOT filter the full dataset
    if "prev_volume_1d" in df.columns and "avg_volume_20d" in df.columns:
        avg = pd.to_numeric(df["avg_volume_20d"], errors="coerce").replace(0, np.nan)
        prev = pd.to_numeric(df["prev_volume_1d"], errors="coerce")
        df["volume_ratio"] = prev / avg

    # 1) Compute scores for all (no news yet)
    df["signal_score"] = df.apply(calculate_signal_score, axis=1)
    # Labels (match get_recommendation thresholds)
    def label(score: float) -> str:
        if score >= 70:
            return "Strong Buy"
        if score >= 40:
            return "Buy"
        if score >= 15:
            return "Slight Buy"
        if score >= -15:
            return "Neutral"
        if score >= -40:
            return "Slight Sell"
        if score >= -70:
            return "Sell"
        return "Strong Sell"
    df["signal_label"] = df["signal_score"].apply(label)

    # 2) Select top and bottom N by score (apply volume-ratio trim only to selection)
    sorted_all = df.sort_values("signal_score", ascending=False).reset_index(drop=True)
    base = sorted_all
    if "volume_ratio" in base.columns and args.min_volume_ratio is not None and np.isfinite(args.min_volume_ratio):
        base = base[base["volume_ratio"] >= float(args.min_volume_ratio)]
    top_sel = base.head(args.top).copy()
    bot_sel = base.tail(args.top).copy()
    selected = pd.concat([top_sel, bot_sel], ignore_index=True)

    # 3) Fetch news only for selected symbols if --news flag is present
    if args.news:
        scorer = SentimentScorer()
        selected_symbols = selected["symbol"].astype(str).tolist()
        articles_by_symbol = []
        counts = []
        total = len(selected_symbols)
        for idx, sym in enumerate(tqdm(selected_symbols, desc="Fetch news (selected)", unit="stk")):
            arts = fetch_news(sym, days=args.days, max_articles=args.max_articles)
            articles_by_symbol.append(arts)
            counts.append(len(arts))
            if idx < total - 1:
                time.sleep(1.02)
        sentiments = scorer.score_articles_multi(articles_by_symbol)
        selected["news_sentiment"] = sentiments
        selected["news_count"] = counts

    # 4) Attach selected news back to main df (others remain NaN)
    df = df.merge(
        selected[["symbol", "news_sentiment", "news_count"] if args.news else ["symbol"]],
        on="symbol",
        how="left",
    )

    # 5) Consolidate and save full (with news for selected only)
    full_out = consolidate(df)
    out_full_path = args.out_full or csv_path.replace(".csv", "_analysis.tsv")
    full_out.to_csv(out_full_path, index=False, sep="\t", float_format="%.2f")

    # 6) Save top/bottom selection (now includes news for those rows)
    topbottom = consolidate(selected).sort_values("signal_score", ascending=False).reset_index(drop=True)
    out_tb_path = args.out_top or csv_path.replace(".csv", "_topbottom.tsv")
    topbottom.to_csv(out_tb_path, index=False, sep="\t", float_format="%.2f")

    # Print quick summary
    print(f"Wrote full analysis: {out_full_path} (rows={len(full_out)})")
    print(f"Wrote top/bottom selection: {out_tb_path} (rows={len(topbottom)})")


if __name__ == "__main__":
    main()


