"""
TEST FLIP HEURISTIC
======================
Added 2026-09-11. wheat_monitor_pro.py's weekly regeneration logic has
a rule, never backtested: "Win -> keep same direction, fresh forecast.
Loss -> the directional read was wrong, flip it." (see get_frozen_
weekly_plan()'s comment, ~line 936). Unlike every validated condition
in this system, this specific heuristic has never been checked against
real outcomes — it's just assumed to be sound trading practice.

This script tests it directly: for every REAL historical stop-hit that
triggered a flip (weekly_break_log.json), it builds a fair counter-
factual — what if the direction had NOT been flipped, using the exact
same stop/target distances (just mirrored to the original direction)
from the exact same real breach price and date — then walks REAL price
forward (daily Close, same convention as score_ranges.py) to see which
one would have actually won.

Read-only. Never writes anything. Reuses score_ranges.py's exact
_walk_outcome()/fetch_price_history() so the win/loss definition is
identical to what the rest of this project already uses.

Usage:
  python3 test_flip_heuristic.py
"""

import json
import importlib.util
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

IL = ZoneInfo("Asia/Jerusalem")

spec = importlib.util.spec_from_file_location("sr", "score_ranges.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def _direction_of(stop, target):
    return 'UP' if target > stop else 'DOWN'


def main():
    breaks = json.loads(Path('weekly_break_log.json').read_text())
    breaks.sort(key=lambda b: b['broken_at'])

    # Only LOSS-triggered breaks actually flip direction (see the real
    # rule this tests) — WIN-triggered breaks keep the same direction,
    # nothing to compare there.
    loss_breaks = [b for b in breaks if '(LOSS)' in b.get('reason', '')]
    print(f"Found {len(loss_breaks)} real historical LOSS-triggered flip events.\n")

    if not loss_breaks:
        print("Nothing to test yet.")
        return

    earliest = datetime.fromisoformat(loss_breaks[0]['broken_at'])
    price_df = sr.fetch_price_history()
    print(f"Fetched {len(price_df)} daily bars.\n")

    flip_wins = flip_losses = 0
    noflip_wins = noflip_losses = 0
    both_pending = 0

    print(f"{'Break date':20} {'Old dir':8} {'New dir (flip)':15} "
          f"{'Flip result':12} {'No-flip result':15}")
    print("-" * 90)

    for b in loss_breaks:
        old_stop, old_target = b['old_stop'], b['old_target']
        old_dir = _direction_of(old_stop, old_target)
        new_dir = 'DOWN' if old_dir == 'UP' else 'UP'
        breach_price = b['price_at_break']
        breach_date = datetime.fromisoformat(b['broken_at'])

        # Find the ACTUAL new stop/target that was set after this flip —
        # either the next break entry's old_stop/old_target (if this
        # segment later broke too) or the current live cache (if this
        # is the most recent segment, still open).
        idx = breaks.index(b)
        if idx + 1 < len(breaks):
            new_stop = breaks[idx + 1]['old_stop']
            new_target = breaks[idx + 1]['old_target']
        else:
            cache = json.loads(Path('weekly_range_cache.json').read_text())['weekly']
            new_stop, new_target = cache.get('stop'), cache.get('target')
        if new_stop is None or new_target is None:
            continue

        # Fair counterfactual: same risk/reward DISTANCES as the actual
        # flipped trade, same breach price/date, just mirrored back to
        # the ORIGINAL (non-flipped) direction.
        target_dist = abs(new_target - breach_price)
        stop_dist = abs(new_stop - breach_price)
        if old_dir == 'UP':
            cf_target = breach_price + target_dist
            cf_stop = breach_price - stop_dist
        else:
            cf_target = breach_price - target_dist
            cf_stop = breach_price + stop_dist

        flip_outcome, _ = sr._walk_outcome(new_stop, new_target, b['broken_at'], None, price_df)
        noflip_outcome, _ = sr._walk_outcome(cf_stop, cf_target, b['broken_at'], None, price_df)

        if flip_outcome == 'WIN':
            flip_wins += 1
        elif flip_outcome == 'LOSS':
            flip_losses += 1
        if noflip_outcome == 'WIN':
            noflip_wins += 1
        elif noflip_outcome == 'LOSS':
            noflip_losses += 1
        if flip_outcome is None and noflip_outcome is None:
            both_pending += 1

        print(f"{breach_date.date().isoformat():20} {old_dir:8} {new_dir:15} "
              f"{flip_outcome or 'pending':12} {noflip_outcome or 'pending':15}")

    print()
    print("=" * 60)
    print("REAL RESULT: does flipping actually outperform not flipping?")
    print("=" * 60)
    flip_total = flip_wins + flip_losses
    noflip_total = noflip_wins + noflip_losses
    if flip_total:
        print(f"FLIP (actual rule):    {flip_wins}/{flip_total} = {flip_wins/flip_total:.1%}")
    else:
        print("FLIP (actual rule):    no resolved results yet")
    if noflip_total:
        print(f"NO-FLIP (counterfactual): {noflip_wins}/{noflip_total} = {noflip_wins/noflip_total:.1%}")
    else:
        print("NO-FLIP (counterfactual): no resolved results yet")
    print(f"\n({both_pending} events still too recent to resolve either way)")
    print("\nSample size caveat: this is real data, but almost certainly")
    print("well under the N>=30 bar this project has used everywhere else.")
    print("Treat this as an early read, not a final verdict.")


if __name__ == "__main__":
    main()
