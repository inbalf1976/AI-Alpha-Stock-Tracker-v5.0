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
from pathlib import Path
import yfinance as yf

TICKER = "ZW=F"
TRIGGER_PCT = 0.02  # 2% move since the last full monitor run

STATE_FILE = Path("wheat_monitor_state.json")


def get_live_price():
    try:
        fast = yf.Ticker(TICKER).fast_info
        live = fast.get('last_price') or fast.get('lastPrice')
        if live and live > 0:
            return float(live)
    except Exception as e:
        print(f"Live price fetch failed: {e}")
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
