import os
import math
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# ===========================
# API KEYS
# ===========================

from dotenv import load_dotenv
load_dotenv()

# Data directory path
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
TRADIER_KEY = os.getenv("TRADIER_API_KEY")

# ===========================
# Finnhub setup (price)
# ===========================

FINNHUB = "https://finnhub.io/api/v1"

def finnhub(endpoint, params={}):
    params["token"] = FINNHUB_KEY
    r = requests.get(FINNHUB + endpoint, params=params)
    r.raise_for_status()
    return r.json()

# ===========================
# Tradier setup (options)
# ===========================

TRADIER = "https://api.tradier.com/v1"
HEADERS = {
    "Authorization": f"Bearer {TRADIER_KEY}",
    "Accept": "application/json"
}

def tradier(endpoint, params={}):
    r = requests.get(TRADIER + endpoint, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()

# ===========================
# Load cached earnings
# ===========================

earnings_path = DATA_DIR / "weekly_earnings.csv"
df = pd.read_csv(earnings_path)
top = df.sort_values("ImpactScore", ascending=False).head(20)

rows = []

for _, row in top.iterrows():
    sym = row["Ticker"]
    print("Processing", sym)

    try:
        # ---------------------------
        # Spot price
        # ---------------------------
        quote = finnhub("/quote", {"symbol": sym})
        S = quote["c"]

        # ---------------------------
        # Get nearest expiration
        # ---------------------------
        exps = tradier("/markets/options/expirations", {"symbol": sym})
        exp = exps["expirations"]["date"][0]

        # ---------------------------
        # Get option chain
        # ---------------------------
        chain = tradier("/markets/options/chains", {
            "symbol": sym,
            "expiration": exp,
            "greeks": "true"
        })["options"]["option"]

        calls = 0
        puts = 0
        total_gamma = 0

        for o in chain:
            oi = o.get("open_interest", 0)
            gamma = o.get("greeks", {}).get("gamma", 0)
            
            # Skip options with no open interest or no gamma (matches positioning_single.py)
            if oi == 0 or gamma == 0:
                continue
            
            option_type = o["option_type"]

            if option_type == "call":
                calls += oi
                total_gamma -= gamma * oi * 100 * S
            else:
                puts += oi
                total_gamma += gamma * oi * 100 * S

        put_call = puts / calls if calls > 0 else 1
        gex = total_gamma / 1e6

        squeeze = (
            0.7 * put_call -
            0.3 * gex
        )

        # Get weekday from date
        date_obj = datetime.strptime(row["Date"], "%Y-%m-%d")
        weekday = date_obj.strftime("%A")
        
        rows.append({
            "Ticker": sym,
            "Date": row["Date"],
            "Weekday": weekday,
            "ImpactScore": row["ImpactScore"],
            "PutCall": round(put_call, 2),
            "GEX": round(gex, 1),
            "SqueezeScore": round(squeeze, 2)
        })

    except Exception as e:
        print(sym, "error:", e)

# ===========================
# Output
# ===========================

out = pd.DataFrame(rows)

# Convert Date to datetime for sorting
out["Date"] = pd.to_datetime(out["Date"])

# Add weekday number for sorting (Monday=0, Sunday=6)
out["WeekdayNum"] = out["Date"].dt.dayofweek

# Sort by weekday (Monday first) then by Date, then by SqueezeScore
out = out.sort_values(["WeekdayNum", "Date", "SqueezeScore"], ascending=[True, True, False])

# Drop the temporary weekday number column
out = out.drop(columns=["WeekdayNum"])

# Convert Date back to string format for display
out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

print("\nEarnings positioning & squeeze risk:\n")
print(out.to_string(index=False))

# Save to data folder
positioning_path = DATA_DIR / "weekly_positioning.csv"
out.to_csv(positioning_path, index=False)

print(f"\nSaved to {positioning_path}")
