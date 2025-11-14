#!/usr/bin/env python3
import os
import sys
import argparse
from typing import List
from pathlib import Path
import pandas as pd


# Ensure project root is on sys.path when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def indices_dir() -> Path:
    d = Path(PROJECT_ROOT) / "data" / "indices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def csv_path(universe: str) -> Path:
    name = universe.lower()
    mapping = {"sp500": "sp500.csv", "nasdaq100": "nasdaq100.csv", "dji": "dji.csv"}
    return indices_dir() / mapping.get(name, f"{name}.csv")


def write_csv(universe: str, tickers: List[str]) -> str:
    out = csv_path(universe)
    df = pd.DataFrame({"symbol": list(dict.fromkeys([t.upper() for t in tickers]))})
    df.to_csv(out, index=False)
    return str(out)


def fetch_sp500() -> List[str]:
    # Prefer DataHub CSV; fallback to Wikipedia
    try:
        df = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv", usecols=["Symbol"])
        vals = df["Symbol"].dropna().astype(str).tolist()
        return [v.strip().upper() for v in vals if v]
    except Exception:
        pass
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        for t in tables:
            if "Symbol" in t.columns:
                vals = t["Symbol"].dropna().astype(str).tolist()
                return [v.strip().upper() for v in vals if v]
    except Exception:
        pass
    return []


def fetch_nasdaq100() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            for col in ("Ticker", "Symbol", "Ticker symbol"):
                if col in t.columns:
                    vals = t[col].dropna().astype(str).tolist()
                    vals = [v.strip().upper() for v in vals if v]
                    vals = [v for v in vals if v.isalnum() and len(v) <= 5]
                    return list(dict.fromkeys(vals))
    except Exception:
        pass
    return []


def fetch_dji() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        for t in tables:
            for col in ("Symbol", "Ticker"):
                if col in t.columns:
                    vals = t[col].dropna().astype(str).tolist()
                    vals = [v.strip().upper() for v in vals if v]
                    vals = [v for v in vals if v.isalnum() and len(v) <= 5]
                    return list(dict.fromkeys(vals))
    except Exception:
        pass
    return []


def save_universe(universe: str, refresh: bool) -> str:
    path = csv_path(universe)
    if path.exists() and not refresh:
        return str(path)
    if universe == "sp500":
        tickers = fetch_sp500()
    elif universe == "nasdaq100":
        tickers = fetch_nasdaq100()
    elif universe == "dji":
        tickers = fetch_dji()
    else:
        raise SystemExit(f"Unknown universe: {universe}")
    if not tickers:
        raise SystemExit(f"Failed to fetch universe: {universe}")
    return write_csv(universe, tickers)


def main():
    parser = argparse.ArgumentParser(description="Save index universes to CSV files in data/indices/")
    parser.add_argument(
        "--universes",
        type=str,
        default="sp500,nasdaq100,dji",
        help="Comma-separated list of universes (sp500,nasdaq100,dji)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch and overwrite CSVs even if they already exist",
    )
    args = parser.parse_args()
    universes: List[str] = [u.strip().lower() for u in args.universes.split(",") if u.strip()]
    for u in universes:
        path = save_universe(u, refresh=args.refresh)
        print(f"{u}: {path}")


if __name__ == "__main__":
    main()

