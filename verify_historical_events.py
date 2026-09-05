"""
VERIFY HISTORICAL EVENTS
==========================
Added 2026-09-04. Gemini supplied 10 real, publicly documented
historical wheat-news events (research prompt + results in this
project's conversation history) to backfill news_signal_scored.json
ahead of waiting weeks for live categorized data to accumulate.

Before trusting ANY of it: Gemini's specific price-percentage claims
are not independently verified. LLMs can state confident, specific-
sounding numbers that are approximate, rounded, or occasionally wrong
— even when the underlying event itself is real and well-documented.
Same discipline used everywhere else in this project (verify against
real data before trusting a claim, e.g. every commit/log check done
by hand this session) — this script checks each event's claimed price
reaction against REAL ZW=F daily data. Only events where real data
roughly confirms the claim should be used as backfill; anything that
doesn't match gets flagged here and should be dropped, not silently
kept.

This is read-only — it only prints a comparison table, never writes
any file. Safe to run any time.

Usage:
  python3 verify_historical_events.py
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf

IL = ZoneInfo("Asia/Jerusalem")
TICKER = "ZW=F"

# Each event: entry_date is the day the news would have first been
# knowable (start of the event, not its resolution) — matching when
# our live system would have flagged it, not the end state.
EVENTS = [
    {
        'name': 'Suez Canal — Ever Given grounding',
        'category': 'confirmed_physical_disruption',
        'entry_date': '2021-03-23',
        'check_days_after': 6,
        'gemini_claim': 'Minimal/flat, <1-1.5% move',
    },
    {
        'name': 'Black Sea ports — Russian naval blockade begins',
        'category': 'confirmed_physical_disruption',
        'entry_date': '2022-02-24',
        'check_days_after': 15,  # ~3 weeks, since claimed peak was "March 2022"
        'gemini_claim': '+40-50% surge (from ~$8.00 to ~$13.50/bu peak)',
    },
    {
        'name': 'Kerch Strait naval incident',
        'category': 'speculative_tension',
        'entry_date': '2018-11-26',
        'check_days_after': 2,
        'gemini_claim': '+1.5-2.5% spike, reverted within 48h',
    },
    {
        'name': 'Black Sea Grain Initiative suspension threat',
        'category': 'speculative_tension',
        'entry_date': '2022-10-31',
        'check_days_after': 4,
        'gemini_claim': '+6% spike then rapid reversal by Nov 4',
    },
    {
        'name': 'Goldman Sachs "commodity supercycle" call',
        'category': 'self_fulfilling_sentiment',
        'entry_date': '2021-01-15',  # approximate, Gemini only gave month
        'check_days_after': 60,
        'gemini_claim': 'Gradual rise ~$6.00 to $7.00+/bu over Q1 2021',
    },
    {
        'name': 'SovEcon "gargantuan Russian crop" forecasts',
        'category': 'self_fulfilling_sentiment',
        'entry_date': '2022-08-01',  # approximate, Gemini said "August 2022"
        'check_days_after': 20,
        'gemini_claim': 'Fell from ~$9.00 to ~$7.50/bu by late August',
    },
    {
        'name': '2010 Russian heatwave/crop desiccation',
        'category': 'weather_crop_damage',
        'entry_date': '2010-06-15',
        'check_days_after': 45,
        'gemini_claim': '+60-70% rally, ~$4.50 to ~$8.40/bu peak',
    },
    {
        'name': '2021 North American spring wheat drought',
        'category': 'weather_crop_damage',
        'entry_date': '2021-06-15',
        'check_days_after': 60,
        'gemini_claim': 'CBOT soft red rose "modestly" (MGEX spring wheat moved more, different ticker)',
    },
    {
        'name': '2010 Russian grain export ban',
        'category': 'other_fundamental',
        'entry_date': '2010-08-05',
        'check_days_after': 1,
        'gemini_claim': '+8% single-day limit-up spike',
    },
    {
        'name': 'August 2021 WASDE shock (Russian production cut)',
        'category': 'other_fundamental',
        'entry_date': '2021-08-12',
        'check_days_after': 1,
        'gemini_claim': '+5-6% intraday surge',
    },
]


def fetch_full_history():
    """
    Wide fetch covering the earliest (2010) through today — ZW=F daily
    data is available on Yahoo back this far, unlike hourly data which
    has a much shorter lookback limit.
    """
    start = datetime(2010, 1, 1)
    end = datetime.now(IL)
    df = yf.Ticker(TICKER).history(start=start, end=end, interval='1d', auto_adjust=False)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def main():
    print("Fetching full ZW=F daily history (2010-present)...")
    df = fetch_full_history()
    print(f"Fetched {len(df)} daily bars.\n")

    print("=" * 100)
    print("HISTORICAL EVENT VERIFICATION — real data vs. Gemini's claim")
    print("=" * 100)

    for ev in EVENTS:
        entry_date = datetime.fromisoformat(ev['entry_date']).date()
        before = df[df.index.date <= entry_date]
        after_window = df[df.index.date > entry_date].head(ev['check_days_after'])

        print(f"\n--- {ev['name']} ({ev['category']}) ---")
        print(f"  Entry date: {ev['entry_date']}  |  Checking +{ev['check_days_after']} trading days")
        print(f"  Gemini claimed: {ev['gemini_claim']}")

        if before.empty:
            print("  REAL DATA: no bars available before this date (likely predates Yahoo's ZW=F history)")
            continue
        if after_window.empty:
            print("  REAL DATA: no bars available after this date")
            continue

        entry_bar = before.iloc[-1]
        entry_price = float(entry_bar['Close'])
        entry_actual_date = before.index[-1].date().isoformat()

        # Same-day intraday range — added 2026-09-04 after event #9 (2010
        # export ban) showed a real spike ON the announcement day itself
        # that a next-day-only check completely missed. Shows whether a
        # same-day announcement caused an intraday move even if it
        # reversed by that day's own close.
        entry_open = float(entry_bar['Open'])
        entry_high = float(entry_bar['High'])
        entry_low = float(entry_bar['Low'])
        intraday_high_pct = (entry_high - entry_open) / entry_open * 100
        intraday_low_pct = (entry_low - entry_open) / entry_open * 100

        final_price = float(after_window.iloc[-1]['Close'])
        final_actual_date = after_window.index[-1].date().isoformat()
        peak_high = float(after_window['High'].max())
        trough_low = float(after_window['Low'].min())
        pct_move = (final_price - entry_price) / entry_price * 100
        peak_pct = (peak_high - entry_price) / entry_price * 100
        trough_pct = (trough_low - entry_price) / entry_price * 100

        print(f"  SAME-DAY intraday ({entry_actual_date}): open {entry_open:.2f}c -> "
              f"high {intraday_high_pct:+.1f}% / low {intraday_low_pct:+.1f}% (from open)")
        print(f"  REAL DATA: entry {entry_actual_date} @ {entry_price:.2f}c (close) -> "
              f"{final_actual_date} @ {final_price:.2f}c  ({pct_move:+.1f}% net)")
        print(f"             window peak: {peak_pct:+.1f}%  |  window trough: {trough_pct:+.1f}%")
        print("  >>> COMPARE the line above to Gemini's claim by hand — does it roughly match?")
        print("      If not, drop this event rather than using it as backfill.")

    print("\n" + "=" * 100)
    print("Nothing here was written anywhere. Manually decide per-event which ones to")
    print("actually backfill into news_signal_scored.json based on the comparison above.")


if __name__ == "__main__":
    main()
