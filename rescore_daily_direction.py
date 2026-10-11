#!/usr/bin/env python3
"""
rescore_daily_direction.py  --  one-off, SAFE BY DEFAULT (created 2026-10-10)

The daily direction is now defined as "does THIS trading day close above where it started?"
and is scored against the same day's own bar (Close vs Open). This script re-scores every logged
call that way so old and new rows are comparable, and prints old-basis vs new-basis accuracy.

  * prediction_log.json is never modified
  * writes prediction_log.rescored_direction.json (a corrected COPY)
  * you check the printout, then copy the file over prediction_log.json yourself

Note: the OLD rows were produced by the old model (which mostly repeated the previous day's
direction), so this measures how those old calls would have scored on the new definition.
Needs network (daily wheat prices from Yahoo).
"""
import json
from pathlib import Path

import score_predictions as sp

LOG = Path("prediction_log.json")
OUT = Path("prediction_log.rescored_direction.json")
FIELDS = ("direction_scored", "direction_correct", "direction_check_date", "direction_basis", "direction_correct_vs_entry")


def acc(rows, key="direction_correct"):
    rows = [e for e in rows if e.get(key) is not None]
    return (sum(1 for e in rows if e[key]), len(rows))


def main():
    log = json.loads(LOG.read_text())
    old = [e for e in log if e.get("direction_scored")]
    ok, n = acc(old)
    print(f"OLD basis (next day's close vs the 03:05 price): {ok}/{n} right" + (f" = {ok / n:.0%}" if n else ""))
    for e in log:
        for k in FIELDS:
            e.pop(k, None)
    price_df = sp.fetch_price_history()
    scored = sp.score_daily_direction(log, price_df)
    new = [e for e in log if e.get("direction_basis") == "same_day_close_vs_open"]
    ok, n = acc(new)
    print(f"NEW basis (same day: close vs open):            {ok}/{n} right" + (f" = {ok / n:.0%}" if n else ""))
    ok2, n2 = acc(new, "direction_correct_vs_entry")
    print(f"for reference, same-day close vs the 03:05 price: {ok2}/{n2} right" + (f" = {ok2 / n2:.0%}" if n2 else ""))
    pending = [e["timestamp"][:10] for e in log if not e.get("direction_scored")]
    print(f"rows newly scored: {scored}   still waiting (day not finished): {pending}")
    OUT.write_text(json.dumps(log, indent=2))
    print(f"\nWrote {OUT} (a corrected copy). {LOG} was NOT modified.")


if __name__ == "__main__":
    main()
