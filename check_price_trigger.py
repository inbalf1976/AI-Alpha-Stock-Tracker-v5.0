"""
CHECK PRICE TRIGGER
=====================
Runs after every news scan (3x/day). Compares the current live price
against the price recorded at the last full wheat_monitor_pro.py run
(wheat_monitor_state.json's last_price). If the move exceeds
TRIGGER_PCT, sets a GitHub Actions output so the workflow can
immediately re-run the full monitor instead of waiting for the next
scheduled 1am cycle or a manual trigger.

This does NOT decide whether the move is "real" or what to do about
it — that judgment (seasonal override, ConvictionGate, weekly break
detection) still lives entirely in wheat_monitor_pro.py, unchanged.
This script's only job is to shorten the delay between "something
big just happened" and "the model actually looks at it."

TRIGGER_PCT is intentionally close to, but slightly below,
wheat_monitor_pro.py's own STOP_PCT (1.5%)/TARGET_PCT (2.5%) — the
goal is to wake the model up BEFORE a stop/target is quietly crossed
between scheduled runs, not just react after the fact.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import yfinance as yf

IL = ZoneInfo("Asia/Jerusalem")

TICKER = "ZW=F"
TRIGGER_PCT = 0.02  # 2% move since the last full monitor run

STATE_FILE = Path("wheat_monitor_state.json")

# ══════════════════════════════════════════════════════════════════════════
# ⚠️  DUPLICATED LOGIC — KEEP IN SYNC WITH wheat_monitor_pro.py  ⚠️
#
# FIX (2026-08-21): this script previously fetched live price straight
# from generic ZW=F, the same bug already fixed in wheat_monitor_pro.py
# on 2026-08-18 (ZW=F's continuous series can silently roll to the
# wrong contract month, a confirmed ~1.9% real price discrepancy).
# Since this script's whole job is deciding whether a ≥2% move
# happened, feeding it a price source with its own ~1-3% error made
# that decision unreliable — a false trigger from contract-roll noise,
# or a missed real move. Now mirrors wheat_monitor_pro.py's
# get_front_month_ticker() + get_live_price() fallback chain exactly.
# If WHEAT_MONTH_CODES / WHEAT_ROLL_BUFFER_DAYS / the roll rule ever
# change in wheat_monitor_pro.py, update this copy too — see
# bug_detector.py for the same pattern already in use.
# ══════════════════════════════════════════════════════════════════════════

WHEAT_MONTH_CODES = {3: 'H', 5: 'K', 7: 'N', 9: 'U', 12: 'Z'}
WHEAT_ROLL_BUFFER_DAYS = 5


def get_front_month_ticker(reference_date=None):
    """See wheat_monitor_pro.py's version of this function for the full
    rationale — this is an intentional duplicate, see KEEP IN SYNC note above."""
    ref = reference_date or datetime.now(IL)
    months = sorted(WHEAT_MONTH_CODES.keys())
    roll_cutoff_day = 15 - WHEAT_ROLL_BUFFER_DAYS

    year = ref.year
    for m in months:
        if ref.month < m:
            return f"ZW{WHEAT_MONTH_CODES[m]}{str(year)[-2:]}.CBT"
        if ref.month == m and ref.day < roll_cutoff_day:
            return f"ZW{WHEAT_MONTH_CODES[m]}{str(year)[-2:]}.CBT"
    return f"ZW{WHEAT_MONTH_CODES[3]}{str(year + 1)[-2:]}.CBT"


def get_live_price():
    front_month = get_front_month_ticker()
    sources = [front_month] if TICKER == front_month else [front_month, TICKER]

    for i, t in enumerate(sources):
        try:
            fast = yf.Ticker(t).fast_info
            live = fast.get('last_price') or fast.get('lastPrice')
            if live and live > 0:
                if i > 0:
                    print(f"⚠️ Live price from FALLBACK source ({t}) — front-month ({front_month}) fetch failed")
                return float(live)
        except Exception as e:
            print(f"Live price fetch failed ({t}): {e}")

    return None


def main():
    if not STATE_FILE.exists():
        print("No wheat_monitor_state.json yet — nothing to compare against. Skipping trigger check.")
        _write_output(False)
        return

    try:
        state = json.loads(STATE_FILE.read_text())
        last_price = state.get('last_price')
    except Exception as e:
        print(f"Failed to read state file: {e}")
        _write_output(False)
        return

    if not last_price:
        print("No last_price recorded yet. Skipping trigger check.")
        _write_output(False)
        return

    current_price = get_live_price()
    if current_price is None:
        print("Could not fetch current live price. Skipping trigger check.")
        _write_output(False)
        return

    move_pct = (current_price - last_price) / last_price
    print(f"Last recorded price: {last_price:.2f}c | Current live price: {current_price:.2f}c | "
          f"Move since last full run: {move_pct:+.2%}")

    triggered = abs(move_pct) >= TRIGGER_PCT
    if triggered:
        print(f"⚠️ TRIGGER: move ({move_pct:+.2%}) exceeds threshold ({TRIGGER_PCT:.0%}) — "
              f"re-running full monitor now instead of waiting for the next scheduled cycle.")
    else:
        print(f"No trigger — move within normal range.")

    _write_output(triggered, move_pct)


def _write_output(triggered, move_pct=None):
    gh_output = os.getenv('GITHUB_OUTPUT')
    if gh_output:
        with open(gh_output, 'a') as f:
            f.write(f"trigger={'true' if triggered else 'false'}\n")
            if move_pct is not None:
                f.write(f"move_pct={move_pct:+.2%}\n")


if __name__ == "__main__":
    main()
