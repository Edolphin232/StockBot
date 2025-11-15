#!/usr/bin/env python3
"""
Hybrid intraday PnL simulator + parameter sweep for ranked_hybrid.tsv outputs.

Sniper profile (C):
    - Very selective
    - High thresholds on confidence, meta_score, and ML probability
    - 1–3 trades/week target (depending on data)
    - Designed to avoid fantasy PnL while still using your 4x option proxy

Assumes:
    - metrics: data/metrics/YYYY/MM/YYYY-MM-DD/metrics.tsv
    - hybrid ranks: data/metrics/YYYY/MM/YYYY-MM-DD/ranked_hybrid.tsv
    - ranking is produced by rank_hybrid_intraday.py

Core idea:
    - Use 'meta_score' + 'bias' + 'confidence' + 'p_up'
    - Enter at 10:15, exit at 10:30
    - Underlying returns from 'return_1015_to_1030_pct'
    - CALL: profit if price up, PUT: profit if price down
    - Option-like returns via leverage factor (e.g. 4x)
"""

import os
import sys
import argparse
from datetime import date, timedelta
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(PROJECT_ROOT, "data", "metrics")


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def ranked_path_for_date(d: date) -> str:
    year = f"{d.year:04d}"
    month = f"{d.month:02d}"
    dstr = d.isoformat()
    return os.path.join(METRICS_DIR, year, month, dstr, "ranked_hybrid.tsv")


def load_ranked_for_date(d: date) -> Optional[pd.DataFrame]:
    path = ranked_path_for_date(d)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return None
    df["trade_date"] = d
    return df


# ---------------------------------------------------------------------
# Core PnL simulation logic
# ---------------------------------------------------------------------

def compute_p_signal(row: pd.Series) -> float:
    """
    Directional ML confidence for this trade:
        - CALL → use p_up
        - PUT  → use 1 - p_up (p_down)
        - NEUTRAL/REJECT → 0 (we don't want to trade those in sniper mode)
    """
    bias = row.get("bias", "NEUTRAL")
    p_up = float(row.get("p_up", 0.5))

    if bias == "CALL":
        return p_up
    elif bias == "PUT":
        return 1.0 - p_up
    else:
        return 0.0


def select_trades_for_day(
    df: pd.DataFrame,
    min_conf: float,
    min_p_signal: float,
    min_abs_meta_score: float,
    top_n: int,
    allowed_biases: List[str],
) -> pd.DataFrame:
    """
    Filter and pick trades for a single day from ranked_hybrid.tsv.

    Sniper mode:
        - Only CALL / PUT (no NEUTRAL / REJECT)
        - High thresholds on confidence, meta_score, and directional ML probability
        - Then take top N by abs_meta_score
    """

    df = df.copy()
    if df.empty:
        print("[INFO] No trades for this day.")
        return df

    # Restrict to directional trades (CALL/PUT) within allowed_biases
    df = df[df["bias"].isin(allowed_biases)]
    if df.empty:
        return df

    # Compute p_signal per row (direction-aware probability)
    df["p_signal"] = df.apply(compute_p_signal, axis=1)

    # Hard sniper filters
    if "confidence" in df.columns:
        df = df[df["confidence"] >= min_conf]

    if "abs_meta_score" in df.columns:
        df = df[df["abs_meta_score"] >= min_abs_meta_score]

    df = df[df["p_signal"] >= min_p_signal]

    if df.empty:
        return df

    # Now sort by abs_meta_score and take top-N
    df = df.sort_values("abs_meta_score", ascending=False)
    df = df.head(top_n)

    return df


def compute_trade_returns(
    trades: pd.DataFrame,
    option_leverage: float = 1.0,
    per_trade_cost: float = 0.0,
) -> pd.DataFrame:
    """
    Given selected trades (CALL & PUT), compute returns:

    - underlying_ret: return_1015_to_1030_pct / 100
    - direction-adjusted underlying_ret based on bias
    - option_ret: underlying_return * option_leverage - per_trade_cost

    Assumes:
        'bias' in {"CALL", "PUT"}
        'return_1015_to_1030_pct' exists.
    """
    trades = trades.copy()

    if "return_1015_to_1030_pct" not in trades.columns:
        raise RuntimeError("Column 'return_1015_to_1030_pct' missing in ranked_hybrid.tsv")

    underlying_ret = trades["return_1015_to_1030_pct"].astype(float) / 100.0

    dir_mult = np.where(
        trades["bias"] == "CALL",
        1.0,
        np.where(trades["bias"] == "PUT", -1.0, 0.0),
    )

    trades["underlying_ret"] = underlying_ret * dir_mult
    trades["option_ret"] = trades["underlying_ret"] * option_leverage - per_trade_cost

    return trades


def simulate_pnl_over_range(
    start: date,
    end: date,
    min_conf: float,
    min_p_signal: float,
    min_abs_meta_score: float,
    top_n: int,
    allowed_biases: List[str],
    option_leverage: float = 1.0,
    per_trade_cost: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main PnL simulation across a date range.
    Returns:
        trades_df: all trades with per-trade returns
        daily_pnl_df: daily aggregate PnL (sum/mean/count)
    """

    all_trades = []

    cur = start
    while cur <= end:
        if cur.weekday() >= 5:
            cur += timedelta(days=1)
            continue

        df_day = load_ranked_for_date(cur)
        if df_day is None or df_day.empty:
            cur += timedelta(days=1)
            continue

        picks = select_trades_for_day(
            df_day,
            min_conf=min_conf,
            min_p_signal=min_p_signal,
            min_abs_meta_score=min_abs_meta_score,
            top_n=top_n,
            allowed_biases=allowed_biases,
        )

        if picks.empty:
            cur += timedelta(days=1)
            continue

        picks = compute_trade_returns(
            picks,
            option_leverage=option_leverage,
            per_trade_cost=per_trade_cost,
        )
        all_trades.append(picks)

        cur += timedelta(days=1)

    if not all_trades:
        return pd.DataFrame(), pd.DataFrame()

    trades_df = pd.concat(all_trades, ignore_index=True)

    daily_pnl_df = (
        trades_df.groupby("trade_date")["option_ret"]
        .agg(["count", "mean", "sum"])
        .reset_index()
    )

    return trades_df, daily_pnl_df


# ---------------------------------------------------------------------
# PnL statistics + equity curve
# ---------------------------------------------------------------------

def max_drawdown(equity: np.ndarray) -> float:
    """
    Compute max drawdown on equity curve (array of cumulative equity).
    Assumes equity[0] is starting equity value.
    """
    if len(equity) == 0:
        return np.nan

    peak = equity[0]
    max_dd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def summarize_pnl(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_ret": np.nan,
            "median_ret": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "profit_factor": np.nan,
            "expectancy": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    r = trades["option_ret"].astype(float)

    n = len(r)
    wins = r[r > 0]
    losses = r[r < 0]

    win_rate = len(wins) / n if n else np.nan
    avg_ret = r.mean()
    median_ret = r.median()
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0

    gross_win = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.nan

    expectancy = avg_ret

    if r.std(ddof=0) > 0:
        sharpe = r.mean() / r.std(ddof=0) * np.sqrt(n)
    else:
        sharpe = np.nan

    eq = 1.0 + r.cumsum().values
    dd = max_drawdown(eq)

    return {
        "n_trades": int(n),
        "win_rate": float(win_rate),
        "avg_ret": float(avg_ret),
        "median_ret": float(median_ret),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
    }


def print_pnl_summary(stats: Dict[str, Any], daily_pnl: pd.DataFrame):
    print("=== OVERALL PNL STATS (option_ret) ===")
    print(f"Trades:          {stats['n_trades']}")
    print(f"Win rate:        {stats['win_rate']:.3f}")
    print(f"Avg return:      {stats['avg_ret']:.4f}")
    print(f"Median return:   {stats['median_ret']:.4f}")
    print(f"Avg win:         {stats['avg_win']:.4f}")
    print(f"Avg loss:        {stats['avg_loss']:.4f}")
    print(f"Profit factor:   {stats['profit_factor']:.2f}")
    print(f"Expectancy/trade {stats['expectancy']:.4f}")
    print(f"Sharpe (per-trade) {stats['sharpe']:.2f}")
    print(f"Max drawdown:    {stats['max_drawdown']:.3f}")

    if not daily_pnl.empty:
        print("\n=== BY DAY SUMMARY (option_ret) ===")
        print(daily_pnl.set_index("trade_date"))


def save_equity_curve(trades: pd.DataFrame, out_path: str):
    """
    Save a simple equity curve by trade index.
    """
    if trades.empty:
        print("[INFO] No trades, skipping equity curve save.")
        return

    r = trades["option_ret"].astype(float)
    eq = 1.0 + r.cumsum()
    df_eq = pd.DataFrame({
        "trade_index": np.arange(len(eq)),
        "equity": eq.values,
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_eq.to_csv(out_path, index=False)
    print(f"[SAVE] Equity curve → {out_path}")


# ---------------------------------------------------------------------
# Parameter sweep (SNIPER GRID)
# ---------------------------------------------------------------------

def sweep_hyperparams(
    start: date,
    end: date,
    allowed_biases: List[str],
    option_leverage: float = 1.0,
    per_trade_cost: float = 0.0,
    min_trades_threshold: int = 5,
):
    """
    Grid search over:
        min_conf
        min_p_signal
        min_abs_meta_score
        top_n_per_day

    Sniper regime:
        - High min_conf
        - High min_p_signal
        - Moderate-high meta_score
        - top_n small (1–2)
    """

    min_conf_grid = [0.5, 0.6, 0.7]
    min_p_signal_grid = [0.65, 0.7, 0.75, 0.8]
    min_abs_meta_score_grid = [0.25, 0.3, 0.35]
    top_n_grid = [1, 2]

    results = []

    for min_conf in min_conf_grid:
        for min_p_signal in min_p_signal_grid:
            for min_abs_meta_score in min_abs_meta_score_grid:
                for top_n in top_n_grid:
                    trades, _ = simulate_pnl_over_range(
                        start, end,
                        min_conf=min_conf,
                        min_p_signal=min_p_signal,
                        min_abs_meta_score=min_abs_meta_score,
                        top_n=top_n,
                        allowed_biases=allowed_biases,
                        option_leverage=option_leverage,
                        per_trade_cost=per_trade_cost,
                    )
                    stats = summarize_pnl(trades)
                    if stats["n_trades"] < min_trades_threshold:
                        continue
                    results.append({
                        "min_conf": min_conf,
                        "min_p_signal": min_p_signal,
                        "min_abs_meta_score": min_abs_meta_score,
                        "top_n": top_n,
                        **stats,
                    })

    if not results:
        print("[INFO] No configs produced enough trades.")
        return

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(
        ["profit_factor", "expectancy"], ascending=[False, False]
    )

    print("\n=== PARAMETER SWEEP RESULTS (top 25 by PF) ===")
    print(df_res.head(25).to_string(index=False))

    out_path = os.path.join(PROJECT_ROOT, "backtest_results", "sweep_hybrid_sniper.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_res.to_csv(out_path, index=False)
    print(f"[SAVE] Sweep results → {out_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid intraday PnL simulator + param sweep (sniper).")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--mode", choices=["single", "sweep"], default="single")
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--min-p-signal", type=float, default=0.6,
                   help="Directional ML probability threshold (p_up for CALL, 1-p_up for PUT).")
    p.add_argument("--min-abs-meta-score", type=float, default=0.2)
    p.add_argument("--top-n", type=int, default=2)
    p.add_argument("--bias", choices=["both", "calls", "puts"], default="both")
    p.add_argument("--option-leverage", type=float, default=4.0,
                  help="Multiplier from underlying_ret → option_ret (e.g. 4.0).")
    p.add_argument("--per-trade-cost", type=float, default=0.0,
                  help="Fixed cost per trade (in return units, e.g. 0.002 = 0.2%).")
    p.add_argument("--save-equity", action="store_true",
                  help="If set, saves equity curve CSV for the chosen config.")
    p.add_argument("--min-trades-threshold", type=int, default=5,
                  help="Minimum trades per config in sweep mode.")
    return p.parse_args()


def main():
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if args.bias == "both":
        allowed_biases = ["CALL", "PUT"]
    elif args.bias == "calls":
        allowed_biases = ["CALL"]
    else:
        allowed_biases = ["PUT"]

    if args.mode == "sweep":
        sweep_hyperparams(
            start, end,
            allowed_biases=allowed_biases,
            option_leverage=args.option_leverage,
            per_trade_cost=args.per_trade_cost,
            min_trades_threshold=args.min_trades_threshold,
        )
        return

    # Single-config run
    trades, daily_pnl = simulate_pnl_over_range(
        start, end,
        min_conf=args.min_conf,
        min_p_signal=args.min_p_signal,
        min_abs_meta_score=args.min_abs_meta_score,
        top_n=args.top_n,
        allowed_biases=allowed_biases,
        option_leverage=args.option_leverage,
        per_trade_cost=args.per_trade_cost,
    )

    stats = summarize_pnl(trades)
    print_pnl_summary(stats, daily_pnl)

    if args.save_equity:
        out_path = os.path.join(
            PROJECT_ROOT,
            "backtest_results",
            f"equity_sniper_{args.start}_to_{args.end}.csv",
        )
        save_equity_curve(trades, out_path)


if __name__ == "__main__":
    main()
