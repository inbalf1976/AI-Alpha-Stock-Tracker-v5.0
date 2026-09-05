"""
FIND SIGNIFICANT PRICE MOVES
==============================
Added 2026-09-04. Different, more reliable direction than
verify_historical_events.py: instead of asking Gemini to recall a
specific historical event AND its price reaction (where the reaction
side turned out wrong 5 times out of 10 when checked), this starts
from our OWN real, already-verified price history — the same 2-year
window backtest.py already uses — and finds the biggest real moves.
The price side here is 100% ground truth, no recall risk at all, since
it's just our own data. Gemini's job becomes much narrower and more
checkable: research what likely caused a KNOWN, EXACT, VERIFIED move,
rather than recall a specific magnitude from memory.

Prints the top single-day and top 5-day moves (by absolute % change)
over the lookback window, with exact dates and magnitudes — feed this
list to Gemini as "what happened around these specific dates" rather
than starting from Gemini's memory of events.

Read-only — never writes any file. Safe to run any time.

Usage:
  python3 find_significant_price_moves.py
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf

IL = ZoneInfo("Asia/Jerusalem")
TICKER = "ZW=F"
LOOKBACK_DAYS = 730  # matches backtest.py's LOOKBACK_DAYS, same window the model/backtest use
TOP_N = 15


def fetch_price_history():
    end = datetime.now(IL)
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = yf.Ticker(TICKER).history(start=start, end=end, interval='1d', auto_adjust=False)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def main():
    print(f"Fetching {LOOKBACK_DAYS}-day ZW=F daily history (same window backtest.py uses)...")
    df = fetch_price_history()
    print(f"Fetched {len(df)} daily bars.\n")

    df['pct_1d'] = df['Close'].pct_change() * 100
    df['pct_5d'] = df['Close'].pct_change(periods=5) * 100

    print("=" * 90)
    print(f"TOP {TOP_N} SINGLE-DAY MOVES (by absolute %)")
    print("=" * 90)
    top_1d = df.reindex(df['pct_1d'].abs().sort_values(ascending=False).index).head(TOP_N)
    for date, row in top_1d.iterrows():
        direction = "UP" if row['pct_1d'] > 0 else "DOWN"
        print(f"  {date.date().isoformat()}  {row['pct_1d']:+6.2f}%  ({direction})  "
              f"close={row['Close']:.2f}c  high={row['High']:.2f}c  low={row['Low']:.2f}c")

    print(f"\n{'=' * 90}")
    print(f"TOP {TOP_N} FIVE-TRADING-DAY MOVES (by absolute %)")
    print("=" * 90)
    top_5d = df.reindex(df['pct_5d'].abs().sort_values(ascending=False).index).head(TOP_N)
    for date, row in top_5d.iterrows():
        direction = "UP" if row['pct_5d'] > 0 else "DOWN"
        window_start = date - timedelta(days=7)  # approx, for display only
        print(f"  {date.date().isoformat()} (5-day window ending here)  {row['pct_5d']:+6.2f}%  "
              f"({direction})  close={row['Close']:.2f}c")

    print(f"\n{'=' * 90}")
    print("Next step: give these exact dates to Gemini and ask what real news/events")
    print("happened around each one — the move itself is already verified (it's our")
    print("own data), so Gemini only needs to research the CAUSE, not recall a")
    print("magnitude. Much narrower, more checkable task than the historical-events")
    print("approach, which got 5 of 10 specific price claims wrong when checked.")


if __name__ == "__main__":
    main()
