"""
SCORE NEWS SIGNALS — category-aware, abnormal-return adjusted
=================================================================
Companion to score_predictions.py, for the LLM news signal (from
news_scanner.py). Real, sourced institutional methodology (RavenPack/
LSEG-style categorization, event-study abnormal returns, EGARCH
volatility-vs-direction separation) confirmed 2026-09-04 that a single
blended BULLISH/BEARISH score across all news types is the wrong
design — see this file's git history / the 2026-09-04 conversation
for the full research. Rebuilt around three real, sourced ideas,
scaled down to match the sample size actually available (institutions
want N>=30 per category before trusting a result with a formal
t-test; we have far less, so this stays data-collection only — no
weighting decision should be made off these numbers yet):

1. CATEGORY SEPARATION — news_scanner.py's Gemini prompt now tags each
   entry with wheat_impact_category (confirmed_physical_disruption,
   speculative_tension, self_fulfilling_sentiment, weather_crop_damage,
   other_fundamental). Each category is scored SEPARATELY, never
   blended — "not every bombed boat matters," and blending categories
   with different real accuracy rates just averages them into noise.
   Entries logged before 2026-09-04 have no category (Gemini prompt
   change is not retroactive) and are bucketed as 'uncategorized'.

2. ABNORMAL RETURN, not raw price move — real event studies subtract
   a control/baseline return to isolate what the NEWS actually caused
   vs. what the broader market was doing anyway (institutions use
   BCOM/GSCI Grains; this project already tracks corn (ZC=F)
   correlation elsewhere, so corn's contemporaneous move is used here
   as a lightweight version of the same idea — NOT full GARCH, which
   would need far more data per category than exists yet to mean
   anything real).

3. SPECULATIVE TENSION scored as A VOLATILITY change, not a direction
   — real institutions price "tension, no confirmed disruption" as an
   options-volatility input, not a directional bet, precisely because
   it usually doesn't have a reliable direction. Scored here as
   whether realized volatility (simple rolling stdev of daily returns,
   not EGARCH) picked up in the following days — a coin-flip result
   on THIS category specifically would match real practice, not
   indicate a broken signal.

HORIZON: daily only (~2 trading days), matching the real institutional
standard of -1 to +3 days. The monthly horizon from an earlier version
of this file was removed 2026-09-04 — real event-study literature
explicitly treats weeks/months as "treacherous" (confounding variables
compound and make the measurement unreliable), which is a hard
correction, not a refinement, of the earlier design.

Deliberately does NOT write scoring fields back onto news_log.json —
that file is the shared raw scan log other things read. Scored results
go in their own file, news_signal_scored.json.

Usage:
  python3 score_news_signals.py
"""

import json
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf

IL = ZoneInfo("Asia/Jerusalem")
WHEAT_TICKER = "ZW=F"
CORN_TICKER = "ZC=F"
MOVE_THRESHOLD_PCT = 0.01   # 1% abnormal move counts as directional; below is FLAT
DAILY_OFFSET = 2            # ~2 trading days — matches real -1 to +3 day standard
VOL_WINDOW = 5              # trading days, for realized-volatility comparison

NEWS_LOG_FILE = Path("news_log.json")
SCORED_FILE = Path("news_signal_scored.json")

DIRECTIONAL_CATEGORIES = {
    'confirmed_physical_disruption', 'self_fulfilling_sentiment',
    'weather_crop_damage', 'other_fundamental',
}
VOLATILITY_CATEGORIES = {'speculative_tension'}


def normalize_signal(entry):
    """
    Same normalization as get_news_signal() in wheat_monitor_pro.py —
    including the 2026-09-06 startswith() fix for glued label+reason
    strings like "BULLISH - Severe European drought...", found via
    loss_forensics.py cross-referencing real losses against news_log.json
    (4 real directional entries were being silently dropped by an exact
    match). Keep these two functions in sync.
    """
    analysis = entry.get('llm_analysis')
    if not analysis:
        return None
    wheat_impact = analysis.get('wheat_impact')
    if isinstance(wheat_impact, dict):
        signal = wheat_impact.get('direction', 'NEUTRAL')
    else:
        signal = wheat_impact or 'NEUTRAL'
    signal = str(signal).upper().strip()
    if signal.startswith('BULLISH'):
        signal = 'BULLISH'
    elif signal.startswith('BEARISH'):
        signal = 'BEARISH'
    return signal if signal in ('BULLISH', 'BEARISH') else None


def normalize_category(entry):
    """
    Returns the wheat_impact_category, or 'uncategorized' for entries
    logged before 2026-09-04 (Gemini prompt change is not retroactive).
    """
    analysis = entry.get('llm_analysis') or {}
    cat = analysis.get('wheat_impact_category')
    known = DIRECTIONAL_CATEGORIES | VOLATILITY_CATEGORIES
    return cat if cat in known else 'uncategorized'


def load_scored():
    if not SCORED_FILE.exists():
        return {}
    return json.loads(SCORED_FILE.read_text())


def save_scored(scored):
    SCORED_FILE.write_text(json.dumps(scored, indent=2))


def fetch_price_history(earliest_needed=None):
    """Daily bars for both wheat and corn (for the abnormal-return baseline)."""
    end = datetime.now(IL)
    start = (earliest_needed - timedelta(days=10)) if earliest_needed else (end - timedelta(days=40))
    wheat = yf.Ticker(WHEAT_TICKER).history(start=start, end=end, interval='1d', auto_adjust=False)
    corn = yf.Ticker(CORN_TICKER).history(start=start, end=end, interval='1d', auto_adjust=False)
    wheat.index = wheat.index.tz_localize(None) if wheat.index.tz else wheat.index
    corn.index = corn.index.tz_localize(None) if corn.index.tz else corn.index
    return wheat, corn


def score_directional(entry_price, entry_date, signal, wheat_df, corn_df):
    """
    Abnormal-return version of the daily check: wheat's move minus
    corn's contemporaneous move over the same window, isolating what
    wheat did BEYOND the broader grain market — see module docstring
    point 2. Returns None if not enough trading days have elapsed yet.
    """
    future_wheat = wheat_df[wheat_df.index.date > entry_date]
    if len(future_wheat) < DAILY_OFFSET:
        return None

    target_date = future_wheat.index[DAILY_OFFSET - 1].date()
    wheat_final = float(future_wheat.iloc[DAILY_OFFSET - 1]['Close'])
    wheat_pct = (wheat_final - entry_price) / entry_price

    corn_before = corn_df[corn_df.index.date <= entry_date]
    corn_after = corn_df[corn_df.index.date == target_date]
    if corn_before.empty or corn_after.empty:
        corn_pct = 0.0  # no corn data available — fall back to raw wheat move
    else:
        corn_entry = float(corn_before.iloc[-1]['Close'])
        corn_final = float(corn_after.iloc[-1]['Close'])
        corn_pct = (corn_final - corn_entry) / corn_entry

    abnormal_pct = wheat_pct - corn_pct  # wheat's move BEYOND corn's move

    predicted_up = signal == 'BULLISH'
    actual_up = abnormal_pct > MOVE_THRESHOLD_PCT
    actual_down = abnormal_pct < -MOVE_THRESHOLD_PCT

    if not actual_up and not actual_down:
        outcome = 'FLAT'
    elif (predicted_up and actual_up) or (not predicted_up and actual_down):
        outcome = 'WIN'
    else:
        outcome = 'LOSS'

    return {'outcome': outcome, 'raw_wheat_pct': round(wheat_pct * 100, 2),
            'corn_baseline_pct': round(corn_pct * 100, 2),
            'abnormal_pct': round(abnormal_pct * 100, 2),
            'checked_date': target_date.isoformat()}


def score_volatility(entry_date, wheat_df):
    """
    For speculative_tension: did realized volatility (simple rolling
    stdev of daily returns) increase in the VOL_WINDOW trading days
    after the entry, vs. the VOL_WINDOW trading days before it? See
    module docstring point 3 for why this category is scored this way
    instead of directionally. Returns None if not enough data yet.
    """
    before = wheat_df[wheat_df.index.date <= entry_date].tail(VOL_WINDOW + 1)
    after = wheat_df[wheat_df.index.date > entry_date].head(VOL_WINDOW)
    if len(before) < VOL_WINDOW + 1 or len(after) < VOL_WINDOW:
        return None

    before_returns = before['Close'].pct_change().dropna().tolist()
    after_returns = after['Close'].pct_change().dropna().tolist()
    if len(before_returns) < 2 or len(after_returns) < 2:
        return None

    vol_before = statistics.stdev(before_returns)
    vol_after = statistics.stdev(after_returns)
    vol_increased = vol_after > vol_before * 1.1  # 10% buffer against noise

    return {'outcome': 'VOL_UP' if vol_increased else 'VOL_FLAT_OR_DOWN',
            'vol_before_pct': round(vol_before * 100, 3),
            'vol_after_pct': round(vol_after * 100, 3),
            'checked_date': after.index[-1].date().isoformat()}


def main():
    if not NEWS_LOG_FILE.exists():
        print(f"No {NEWS_LOG_FILE} found. Nothing to score.")
        return

    log = json.loads(NEWS_LOG_FILE.read_text())
    scored = load_scored()
    print(f"Loaded {len(log)} logged news scans.")

    def needs_work(entry):
        if 'timestamp' not in entry or 'llm_analysis' not in entry:
            return False
        if normalize_signal(entry) is None:
            return False
        return entry['timestamp'] not in scored or scored[entry['timestamp']].get('result') is None

    candidates = [e['timestamp'] for e in log if needs_work(e)]
    earliest_needed = datetime.fromisoformat(min(candidates)).astimezone(IL) if candidates else None

    wheat_df, corn_df = fetch_price_history(earliest_needed)
    print(f"Fetched {len(wheat_df)} wheat bars, {len(corn_df)} corn bars.\n")

    newly_scored = 0
    for entry in log:
        if 'timestamp' not in entry or 'llm_analysis' not in entry:
            continue
        ts_key = entry['timestamp']
        signal = normalize_signal(entry)
        if signal is None:
            continue

        if ts_key in scored and scored[ts_key].get('result') is not None:
            continue  # already resolved

        category = normalize_category(entry)
        key_phrase = (entry.get('llm_analysis', {}) or {}).get('key_risk') \
            or (entry.get('llm_analysis', {}) or {}).get('summary', '')[:80]

        entry_time = datetime.fromisoformat(ts_key).replace(tzinfo=None)
        entry_date = entry_time.date()
        at_or_before = wheat_df[wheat_df.index.date <= entry_date]
        if at_or_before.empty:
            continue
        entry_price = float(at_or_before.iloc[-1]['Close'])

        if category in VOLATILITY_CATEGORIES:
            result = score_volatility(entry_date, wheat_df)
        else:
            result = score_directional(entry_price, entry_date, signal, wheat_df, corn_df)

        if result is None:
            scored.setdefault(ts_key, {'signal': signal, 'category': category,
                                        'key_phrase': key_phrase, 'result': None})
            continue

        scored[ts_key] = {'signal': signal, 'category': category,
                           'key_phrase': key_phrase, 'result': result}
        newly_scored += 1
        print(f"  {ts_key[:16]} | {category:<28} | {signal:<8} -> {result['outcome']}")

    save_scored(scored)
    print(f"\nNewly scored this run: {newly_scored}")

    print("\n" + "=" * 60)
    print("LIVE NEWS SIGNAL ACCURACY — BY CATEGORY")
    print("=" * 60)
    print("(Every category below is DATA COLLECTION ONLY. Real")
    print(" institutional practice wants N>=30 per category with formal")
    print(" significance testing before acting on any of this — none of")
    print(" these categories are anywhere near that yet.)")

    all_categories = sorted(set(v['category'] for v in scored.values() if v.get('result')))
    for cat in all_categories:
        entries = [v for v in scored.values() if v['category'] == cat and v.get('result')]
        print(f"\n--- {cat} (n={len(entries)}) ---")
        if cat in VOLATILITY_CATEGORIES:
            vol_up = sum(1 for e in entries if e['result']['outcome'] == 'VOL_UP')
            print(f"  Volatility increased afterward: {vol_up}/{len(entries)} = "
                  f"{vol_up/len(entries):.1%}" if entries else "  (none)")
        else:
            directional = [e for e in entries if e['result']['outcome'] in ('WIN', 'LOSS')]
            flats = len(entries) - len(directional)
            if directional:
                wins = sum(1 for e in directional if e['result']['outcome'] == 'WIN')
                print(f"  Directional (abnormal-return adjusted): {wins}/{len(directional)} = "
                      f"{wins/len(directional):.1%}  (+ {flats} FLAT)")
            else:
                print("  (No directional results yet.)")


if __name__ == "__main__":
    main()
