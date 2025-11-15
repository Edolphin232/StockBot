import numpy as np

def calc_volume_metrics(df_pm, df_daily):
    if df_pm is None or df_pm.empty:
        return {
            "premarket_volume_60m": np.nan,
            "relative_volume_20d": np.nan
        }

    sixty = df_pm.tail(60)["volume"].sum()

    if df_daily.empty:
        rel = np.nan
    else:
        avg20 = df_daily["volume"].tail(20).mean()
        rel = df_pm["volume"].sum() / avg20 if avg20 > 0 else np.nan

    return {
        "premarket_volume_60m": sixty,
        "relative_volume_20d": rel,
    }
