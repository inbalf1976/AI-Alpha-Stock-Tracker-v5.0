"""
price_source_diagnostic.py — ONE-OFF diagnostic, not part of the
regular pipeline. Run this manually to see, in a single moment, what
yfinance actually returns for both the generic continuous symbol
(ZW=F) and the specific front-month contract (ZWU26.CBT) — side by
side, with real timestamps — so we can compare them fairly instead of
guessing from secondhand snippets captured at different times from
different platforms.

Also fetches recent intraday history for both, to see whether either
symbol's data looks obviously stale (e.g. a last-updated timestamp
that's hours old) or genuinely live.

Usage: python3 price_source_diagnostic.py
"""

import yfinance as yf
from datetime import datetime, timezone

def check_symbol(symbol):
    print(f"\n{'='*60}")
    print(f"SYMBOL: {symbol}")
    print(f"{'='*60}")

    try:
        t = yf.Ticker(symbol)
        fast = t.fast_info
        print("fast_info fields:")
        for key in ['last_price', 'lastPrice', 'open', 'day_high', 'day_low',
                    'previous_close', 'regular_market_previous_close']:
            try:
                val = fast.get(key)
                if val is not None:
                    print(f"  {key}: {val}")
            except Exception:
                pass
    except Exception as e:
        print(f"  fast_info FAILED: {e}")

    try:
        intraday = t.history(period='1d', interval='1m')
        if not intraday.empty:
            last_row = intraday.iloc[-1]
            last_ts = intraday.index[-1]
            print(f"\n1-minute intraday — last bar:")
            print(f"  Timestamp: {last_ts}")
            print(f"  Close: {last_row['Close']}")
            print(f"  (compare this timestamp to right now — if it's more than "
                  f"a few minutes old during active trading hours, that symbol's "
                  f"data may be lagging)")
        else:
            print("\n1-minute intraday: EMPTY (no data returned)")
    except Exception as e:
        print(f"\n1-minute intraday FAILED: {e}")

    try:
        daily = t.history(period='5d', interval='1d')
        if not daily.empty:
            print(f"\nLast 5 daily bars:")
            print(daily[['Open', 'High', 'Low', 'Close', 'Volume']].to_string())
    except Exception as e:
        print(f"\nDaily history FAILED: {e}")


print(f"Diagnostic run at: {datetime.now(timezone.utc).isoformat()} UTC")
check_symbol("ZW=F")
check_symbol("ZWU26.CBT")

print(f"\n{'='*60}")
print("Compare the 'last_price'/'lastPrice' and intraday timestamps")
print("above for both symbols. If they disagree significantly AND")
print("both have recent timestamps, that's a real, current data")
print("discrepancy worth escalating (e.g. to Yahoo Finance's own")
print("data quality, since both come from the same provider).")
print(f"{'='*60}")
