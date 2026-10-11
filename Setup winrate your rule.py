#!/usr/bin/env python3
"""
setup_winrate_your_rule.py  --  READ-ONLY analysis (created 2026-10-11)

Re-scores every weekly plan in weekly_break_log.json with the user's own rule:

    WIN  = the plan's TARGET is hit at any time while the plan is live, even if the stop was touched
           before it, and even if price keeps going past the target.
    LOSS = the stop is hit and the target is NEVER hit.
    (a plan that stays inside its range and touches neither is a WIN under her rule; those plans do
     not appear in weekly_break_log.json - it only records plans that broke - so her true rate can
     only be higher than what is computed here)

"While the plan is live" = from the plan's start (the previous break in the same ISO week, or the
start of that ISO week) to the END of that ISO week. A plan that breaks early still gets the rest of
its week to hit the target.

For every plan the script also shows the system's own verdict (first level touched) so the two
rules can be compared side by side.

Prices: hourly ZW=F via hunter_pattern_research.fetch() (glitch weeks removed). A plan whose window
touches a removed glitch period is reported as UNCERTAIN and left out of the rates.

Never writes repo state, never sends Telegram, never commits.
"""
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import hunter_pattern_research as hp

IL = ZoneInfo("Asia/Jerusalem")
LOG = "weekly_break_log.json"


def week_bounds(iso_key):
    y, w = iso_key.split("-W")
    start = datetime.fromisocalendar(int(y), int(w), 1).replace(tzinfo=IL)                 # Monday 00:00 IL
    end = datetime.fromisocalendar(int(y), int(w), 7).replace(hour=23, minute=59, tzinfo=IL)  # Sunday night IL
    return pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")


def touched(bars, direction, level, kind):
    """kind='target' or 'stop' for a plan going `direction`."""
    if bars.empty:
        return False
    up_plan = direction == "UP"
    if kind == "target":
        return bool((bars["High"] >= level).any()) if up_plan else bool((bars["Low"] <= level).any())
    return bool((bars["Low"] <= level).any()) if up_plan else bool((bars["High"] >= level).any())


def analyse(breaks, h):
    h = h.copy()
    h.index = h.index.tz_convert("UTC")
    bad = hp.bad_array(h)
    rows, last_break = [], {}
    for b in sorted(breaks, key=lambda x: pd.Timestamp(x["broken_at"])):
        key = b["iso_key"]
        wk_start, wk_end = week_bounds(key)
        broke = pd.Timestamp(b["broken_at"]).tz_convert("UTC")
        start = last_break.get(key, wk_start)
        last_break[key] = broke
        stop, target = b.get("old_stop"), b.get("old_target")
        if stop is None or target is None:
            continue
        direction = "UP" if target > stop else "DOWN"
        win = (h.index >= start) & (h.index <= wk_end)
        bars = h[win]
        sys_out = "WIN" if "WIN" in b["reason"].upper() else "LOSS"
        row = {"week": key, "broke_at": str(broke.tz_convert(IL))[:16], "direction": direction, "stop": stop,
               "target": target, "system_says": sys_out}
        if bars.empty:
            row["your_rule"] = "UNCERTAIN (no price data)"
        elif bad[win].any():
            row["your_rule"] = "UNCERTAIN (glitch weeks in window)"
        else:
            t_hit = touched(bars, direction, target, "target")
            s_hit = touched(bars, direction, stop, "stop")
            row["target_hit"], row["stop_hit"] = t_hit, s_hit
            row["your_rule"] = "WIN" if t_hit else "LOSS" if s_hit else "WIN (stayed in range)"
        rows.append(row)
    return rows


def summarize(rows):
    ok = [r for r in rows if not r["your_rule"].startswith("UNCERTAIN")]
    sys_w = sum(r["system_says"] == "WIN" for r in ok)
    you_w = sum(r["your_rule"].startswith("WIN") for r in ok)
    rescued = [r for r in ok if r["system_says"] == "LOSS" and r["your_rule"].startswith("WIN")]
    lost_by_you = [r for r in ok if r["system_says"] == "WIN" and r["your_rule"] == "LOSS"]
    by_month = {}
    for r in ok:
        m = r["broke_at"][:7]
        d = by_month.setdefault(m, {"plans": 0, "system_wins": 0, "your_rule_wins": 0})
        d["plans"] += 1
        d["system_wins"] += r["system_says"] == "WIN"
        d["your_rule_wins"] += r["your_rule"].startswith("WIN")
    n = len(ok)
    return {"plans_scored": n, "uncertain_excluded": len(rows) - n,
            "system_rule_(first_level_touched)": f"{sys_w}/{n} = {sys_w / n:.0%}" if n else None,
            "YOUR_RULE_(target_hit_anytime_in_week)": f"{you_w}/{n} = {you_w / n:.0%}" if n else None,
            "system_losses_that_hit_target_later": len(rescued),
            "system_wins_that_your_rule_calls_loss": len(lost_by_you),
            "by_month": by_month,
            "note": "plans that never broke are not in the log; under your rule they are wins, so the true rate is >= this"}


def main():
    breaks = json.load(open(LOG))
    h = hp.fetch()
    rows = analyse(breaks, h)
    summary = summarize(rows)
    print(f"\n{'week':9s} {'broke (IL)':17s} {'dir':5s} {'stop':>7s} {'target':>7s} {'system':7s} your rule")
    for r in rows:
        print(f"{r['week']:9s} {r['broke_at']:17s} {r['direction']:5s} {r['stop']:7.2f} {r['target']:7.2f} {r['system_says']:7s} {r['your_rule']}")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "plans": rows}, open("setup_winrate_report.json", "w"), indent=2, default=str)
    print("\nWrote setup_winrate_report.json (nothing else was changed).")


if __name__ == "__main__":
    main()
