#!/usr/bin/env python3
"""
Generate top 3 CALLS and PUTS with a minimal, human-friendly readout.

Fields included:
    - symbol
    - bias (CALL/PUT)
    - meta_score
    - confidence
    - reasoning (compact summary from rules + ML)

NO FUTURE DATA INCLUDED.
"""

import os
import sys
import argparse
from datetime import date, timedelta
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(PROJECT_ROOT, "data", "metrics")
SIGNALS_DIR = os.path.join(PROJECT_ROOT, "data", "signals")


def ranked_path_for_date(d: date) -> str:
    return os.path.join(
        METRICS_DIR,
        f"{d.year:04d}",
        f"{d.month:02d}",
        d.isoformat(),
        "ranked_hybrid.tsv",
    )


def load_ranked(d: date) -> pd.DataFrame:
    path = ranked_path_for_date(d)
    if not os.path.exists(path):
        raise RuntimeError(f"Missing ranked_hybrid.tsv for {d}")
    return pd.read_csv(path, sep="\t")


def reasoning_from_row(row) -> str:
    """
    Creates a compact reasoning line from available rule & ML fields.
    """
    parts = []

    # ML tendency
    if "p_up" in row:
        if row["p_up"] > 0.6:
            parts.append("ML↑")
        elif row["p_up"] < 0.4:
            parts.append("ML↓")

    # Rule direction
    if "rule_direction" in row:
        if row["rule_direction"] > 0.35:
            parts.append("rules↑")
        elif row["rule_direction"] < -0.35:
            parts.append("rules↓")

    # Notes from rule system
    rn = row.get("rule_notes", "")
    if isinstance(rn, str) and rn:
        parts.append(rn.replace(",", "|"))

    return " ".join(parts) if parts else "n/a"


def print_signals(df, title, d):
    if df.empty:
        print(f"\n=== {title} for {d} : NONE ===")
        return

    # Build display table
    out = pd.DataFrame({
        "symbol": df["symbol"],
        "bias": df["bias"],
        "meta_score": df["meta_score"].round(4),
        "confidence": df["confidence"].round(3),
        "reason": df.apply(reasoning_from_row, axis=1),
    })

    print(f"\n=== {title} for {d} ===")
    print(out.to_string(index=False))


def save_signals(df_calls, df_puts, d):
    out_dir = os.path.join(SIGNALS_DIR, d.isoformat())
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "top_signals.tsv")

    merged = pd.concat([
        df_calls.assign(direction="CALL"),
        df_puts.assign(direction="PUT")
    ], ignore_index=True)

    merged.to_csv(path, sep="\t", index=False)
    print(f"[SAVED] {path}")


def process_day(d: date):
    print(f"\n[DAY] {d}")
    df = load_ranked(d)
    if df.empty:
        print("[WARN] ranked_hybrid.tsv empty")
        return

    # Pick top 3 each direction
    calls = df[df["bias"] == "CALL"].sort_values("abs_meta_score", ascending=False).head(3)
    puts  = df[df["bias"] == "PUT"].sort_values("abs_meta_score", ascending=False).head(3)

    print_signals(calls, "TOP 3 CALLS", d)
    print_signals(puts, "TOP 3 PUTS", d)

    save_signals(calls, puts, d)


def run_range(start: date, end: date):
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            process_day(cur)
        else:
            print(f"[SKIP] {cur} (weekend)")
        cur += timedelta(days=1)
    print("\n[DONE]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date")
    p.add_argument("--start")
    p.add_argument("--end")
    args = p.parse_args()

    if args.date:
        process_day(date.fromisoformat(args.date))
        return

    if args.start:
        s = date.fromisoformat(args.start)
        e = date.fromisoformat(args.end) if args.end else s
        run_range(s, e)
        return

    # default: today
    process_day(date.today())


if __name__ == "__main__":
    main()
