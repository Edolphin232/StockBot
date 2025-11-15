import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------

def safe_mean(x):
    x = [v for v in x if v is not None and not np.isnan(v)]
    return np.mean(x) if x else None


# -----------------------------------------------------------
# MAIN ANALYTICS FUNCTION
# -----------------------------------------------------------

def analyze_backtest(df: pd.DataFrame, title="Backtest Performance"):
    """
    Compute detailed analytics and produce charts.
    """

    # -------------------------------------------------------
    # CLEAN DATA
    # -------------------------------------------------------
    df = df.copy()

    # Build unified return columns - check which exist
    call_col = "call_close_ret" if "call_close_ret" in df.columns else None
    put_col = "put_close_ret" if "put_close_ret" in df.columns else None
    
    if call_col:
        df[call_col] = pd.to_numeric(df[call_col], errors="coerce")
    if put_col:
        df[put_col] = pd.to_numeric(df[put_col], errors="coerce")

    # Combined return per day (call or put if present)
    ret_cols = [c for c in [call_col, put_col] if c is not None]
    if ret_cols:
        df["best_ret"] = df[ret_cols].max(axis=1, skipna=True)
        # Check if we have any actual return data
        if df["best_ret"].notna().sum() == 0:
            print("[WARN] Return columns exist but all values are missing/None")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return None
    else:
        print("[WARN] No return columns found in backtest results")
        print(f"[INFO] Available columns: {list(df.columns)}")
        return None

    # Equity curve
    df["equity"] = (1 + df["best_ret"].fillna(0)).cumprod()


    # -------------------------------------------------------
    # SUMMARY STATS
    # -------------------------------------------------------
    total_days   = len(df)
    total_trades = df["best_ret"].notna().sum()
    exposure     = total_trades / total_days

    wins = df[df["best_ret"] > 0]
    losses = df[df["best_ret"] < 0]

    win_rate = len(wins) / total_trades if total_trades else 0

    avg_win  = safe_mean(wins["best_ret"])
    avg_loss = safe_mean(losses["best_ret"])

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if avg_win and avg_loss else None

    profit_factor = (wins["best_ret"].sum() / abs(losses["best_ret"].sum())) if len(losses) else None


    # -------------------------------------------------------
    # PRINT SUMMARY
    # -------------------------------------------------------
    print("\n===============================")
    print("      Backtest Summary")
    print("===============================")
    print(f"Total Trading Days:      {total_days}")
    print(f"Trades Executed:         {total_trades}")
    print(f"Exposure (% days traded): {exposure:.2%}")
    print(f"Win Rate:                {win_rate:.2%}")
    print(f"Average Win:             {avg_win:.4f}")
    print(f"Average Loss:            {avg_loss:.4f}")
    print(f"Expectancy Per Trade:    {expectancy:.4f}")
    print(f"Profit Factor:           {profit_factor:.3f}")
    print("===============================\n")


    # -------------------------------------------------------
    # CHARTS
    # -------------------------------------------------------

    # 1. Equity Curve
    plt.figure(figsize=(12,4))
    plt.plot(df["date"], df["equity"])
    plt.title(f"Equity Curve — {title}")
    plt.ylabel("Equity (starting at 1.0)")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Histogram of Returns
    plt.figure(figsize=(10,4))
    plt.hist(df["best_ret"].dropna(), bins=20)
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. CALL vs PUT performance scatter (only if both exist)
    if call_col and put_col:
        plt.figure(figsize=(6,6))
        plt.scatter(df[call_col], df[put_col], alpha=0.6)
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.title("CALL vs PUT Close Returns")
        plt.xlabel("CALL Return")
        plt.ylabel("PUT Return")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    # 4. Daily Performance Bar Chart
    plt.figure(figsize=(12,4))
    plt.bar(df["date"], df["best_ret"].fillna(0))
    plt.title("Daily Best Return (Call or Put)")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "total_days": total_days,
        "total_trades": total_trades,
        "exposure": exposure,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }

