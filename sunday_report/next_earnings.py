import yfinance as yf
import pandas as pd
import numpy as np
import sys
from math import sqrt

def get_earnings_intel(ticker):
    stock = yf.Ticker(ticker)

    # --- Next Earnings ---
    cal = stock.earnings_dates
    if cal is None or cal.empty:
        print("No earnings calendar available")
        return

    now = pd.Timestamp.now(tz=cal.index.tz)
    future = cal[cal.index > now]
    if future.empty:
        print("No future earnings found")
        return

    next_date = future.index[0]
    eps_est = future.iloc[0].get("EPS Estimate", None)

    print(f"\n📊 {ticker.upper()} Earnings Intel")
    print("──────────────────────────────────")
    print("Next earnings:", next_date.strftime("%Y-%m-%d %H:%M %Z"))
    if eps_est:
        print("EPS Estimate:", eps_est)

    session = "BMO" if next_date.hour < 12 else "AMC"
    print("Session:", session)

    # --- Historical EPS beats ---
    earnings = stock.earnings_history

    if earnings is not None and not earnings.empty:
        earnings = earnings.tail(8).copy()
        earnings["surprise"] = earnings["epsActual"] - earnings["epsEstimate"]
        earnings["surprise_pct"] = earnings["surprise"] / earnings["epsEstimate"] * 100

        print("\nLast 8 earnings:")
        for _, row in earnings.iterrows():
            beat = "BEAT" if row["surprise"] > 0 else "MISS"
            print(
                f"{row.name.date()}  "
                f"Act: {row['epsActual']:.2f}  "
                f"Est: {row['epsEstimate']:.2f}  "
                f"{beat}  "
                f"{row['surprise_pct']:+.1f}%"
            )

        avg_surprise = earnings["surprise_pct"].mean()
        print("\nAverage EPS surprise:", f"{avg_surprise:+.2f}%")

    # --- Expected move from options ---
    try:
        options = stock.option_chain()
        calls = options.calls
        puts = options.puts

        # Pick ATM options
        price = stock.history(period="1d")["Close"].iloc[-1]
        calls["dist"] = abs(calls["strike"] - price)
        atm_call = calls.sort_values("dist").iloc[0]

        iv = atm_call["impliedVolatility"]
        dte = (next_date - now).days / 365

        exp_move = price * iv * sqrt(dte)

        print("\nOptions-implied move:")
        print(f"IV: {iv*100:.1f}%")
        print(f"Expected ± move: ${exp_move:.2f}  ({exp_move/price*100:.1f}%)")

    except:
        print("\nOptions data unavailable")

    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python earnings_intel.py TICKER")
        sys.exit(1)

    get_earnings_intel(sys.argv[1])
