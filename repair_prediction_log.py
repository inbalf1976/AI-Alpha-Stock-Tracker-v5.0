#!/usr/bin/env python3
"""
repair_prediction_log.py  --  one-off repair (created 2026-10-07), SAFE BY DEFAULT

Why: from 2026-09-03 the log stored the DAILY direction next to the WEEKLY plan's stop/target.
When the two sides disagreed (rows 2026-09-09..16) the old scorer applied the daily direction to
levels laid out for the opposite side and logged an instant "same_bar_ambiguous" LOSS with a
positive pnl. score_predictions.py is now fixed going forward; this script re-scores only the
already-scored rows that were hit by that bug.

What it does:
  * finds rows with scoring_method == 'real_setup' whose stop/target geometry points the opposite
    way from the logged `direction`
  * re-scores them with the fixed scorer (side taken from the levels)
  * prints old -> new for each row
  * writes prediction_log.repaired.json (a corrected COPY) - prediction_log.json is never touched

You review the printout and, if you agree, copy prediction_log.repaired.json over
prediction_log.json yourself.  Needs network (daily wheat prices from Yahoo).
"""
import json
from pathlib import Path

import score_predictions as sp

LOG = Path("prediction_log.json")
OUT = Path("prediction_log.repaired.json")
FIELDS = ("outcome", "exit_reason", "pnl_cents", "scoring_method")


def geometry(e):
    s, t, p = e.get("stop_price"), e.get("target_price"), e.get("entry_price")
    if s is None or t is None or p is None:
        return None
    return "UP" if t > p > s else "DOWN" if t < p < s else None


def repair(log, price_df):
    changes = []
    for e in log:
        if not (e.get("validated") and e.get("scoring_method") == "real_setup"):
            continue
        geo = geometry(e)
        if geo is None or geo == e.get("direction"):
            continue                                   # consistent row - leave alone
        probe = dict(e, setup_direction=geo)
        outcome, reason, pnl, method = sp.score_one_prediction(probe, price_df)
        if outcome is None:
            changes.append({"date": e["timestamp"][:10], "status": "SKIPPED (cannot re-score yet)"})
            continue
        before = {k: e.get(k) for k in FIELDS}
        for k, v in zip(FIELDS, (outcome, reason, pnl, method)):
            e[k] = v
        e["setup_direction"] = geo
        e["rescored_2026_10_07"] = True
        changes.append({"date": e["timestamp"][:10], "daily_direction": e["direction"], "weekly_setup_side": geo,
                        "old": f'{before["outcome"]} ({before["exit_reason"]}, {before["pnl_cents"]})',
                        "new": f"{outcome} ({reason}, {pnl})"})
    return changes


def main():
    log = json.loads(LOG.read_text())
    price_df = sp.fetch_price_history()
    changes = repair(log, price_df)
    print(f"rows checked: {len(log)}   rows re-scored/flagged: {len(changes)}\n")
    for c in changes:
        print(c)
    OUT.write_text(json.dumps(log, indent=2))
    print(f"\nWrote {OUT} (a corrected copy). {LOG} was NOT modified.")


if __name__ == "__main__":
    main()
