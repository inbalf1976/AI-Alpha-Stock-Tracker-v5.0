"""
VOLUME LAG DIAGNOSTIC
=======================
Checks whether yfinance's daily Volume figure for ZW=F changes when
re-fetched later — i.e. whether the most recent daily bar's volume
is provisional/lagged and gets corrected once Yahoo fully processes
settlement data.

Run this any time. It prints the Volume for the last several daily
bars. If you run it again tomorrow and yesterday's figure (which
should now be "old enough" to be finalized) has changed significantly
from what an earlier run showed, that confirms the lag theory.

Usage:
  python3 volume_lag_check.py
"""

import yfinance as yf
from datetime import datetime, timedelta

TICKER = "ZWU26.CBT"

end = datetime.now()
start = end - timedelta(days=20)
df = yf.Ticker(TICKER).history(start=start, end=end, interval='1d', auto_adjust=False)

print(f"Checked at: {datetime.now().isoformat()}")
print(f"\nLast 10 daily bars for {TICKER}:")
print(f"{'Date':<12} {'Volume':>10}")
print("-" * 24)
for idx, row in df.tail(10).iterrows():
    print(f"{idx.date()!s:<12} {row['Volume']:>10.0f}")

print(f"\n20-day rolling average volume: {df['Volume'].tail(20).mean():.0f}")
print(f"\nSave this output. Run again tomorrow and compare the volume")
print(f"shown for the SAME dates (especially the most recent 1-2 bars).")
print(f"If those numbers grow significantly between runs, it confirms")
print(f"the most recent bar's volume is provisional/lagged when the")
print(f"main monitor script runs at 1-4 AM Israel time.")
