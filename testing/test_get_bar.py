import pytz
from datetime import datetime

import fetchers.alpaca_client as alpaca

EASTERN = pytz.timezone("US/Eastern")

start = EASTERN.localize(datetime(2026, 2, 17, 9, 30))  # Feb 1 2025 9:30 AM ET
end   = EASTERN.localize(datetime(2026, 2, 17, 16, 0))  # Feb 17 2026 4:00 PM ET

df = alpaca.fetch_bars("SPY", start, end, timeframe="1m")

print(df)