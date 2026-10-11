#!/usr/bin/env python3
"""
setup_rescore.py  --  READ-ONLY (created 2026-10-11)

Re-scores the weekly setups ("plans") on the USER'S OWN rules:
  * she follows every setup blindly
  * TARGET reached                                              = WIN
  * stop TOUCHED (within 0.5% beyond the stop) and price later reverses to the target = WIN
  * price goes MORE than 0.5% beyond the stop                     = LOSS (even if the target comes later)
  * stop touched and the target is never reached                  = LOSS
  * neither stop nor target reached (still in range, any number of days) = NEUTRAL
Each plan is followed from its start until its target/breach, or until the end of its ISO week
(Friday 21:30 Israel) - the deadline she did not object to.

Plans: every row of weekly_break_log.json (levels old_stop / old_target are exact). A plan's start is
the previous break in the same ISO week, otherwise Monday 03:05 Israel time of its week. Side:
DOWN when stop > target, else UP. Plans that never broke (neutral weeks) are not in that log, so the
NEUTRAL count is a lower bound; win rate excludes neutrals anyway.

Path: hourly ZW=F bars (cleaned like the other research scripts). Hourly bars hide the order of events
inside one bar: a bar that reaches both the target and the 0.5% breach is reported as AMBIGUOUS and the
win rate is shown as a range (ambiguous counted as loss .. counted as win). Plans touching glitch-cleaned
weeks are skipped and listed.

Also shown for comparison: the STRICT rule the log used (stop touched first = LOSS, no tolerance).
Never writes repo state, never sends Telegram, never commits. Needs hunter_pattern_research.py.
"""
import datetime as dt
import json

import numpy as np
import pandas as pd

import hunter_pattern_research as hp

TOL = 0.005                       # 0.5% beyond the stop
IL = "Asia/Jerusalem"


def week_bounds(iso_key):
    y, w = iso_key.split("-W")
    mon = dt.date.fromisocalendar(int(y), int(w), 1)
    fri = dt.date.fromisocalendar(int(y), int(w), 5)
    start = pd.Timestamp(f"{mon} 03:05", tz=IL)
    end = pd.Timestamp(f"{fri} 21:30", tz=IL)
    return start, end


def build_plans(breaks):
    plans, last_break = [], {}
    for b in sorted(breaks, key=lambda x: x["broken_at"]):
        ws, we = week_bounds(b["iso_key"])
        start = last_break.get(b["iso_key"], ws)
        stop, tgt = float(b["old_stop"]), float(b["old_target"])
        t = pd.Timestamp(b["broken_at"])
        txt = b["reason"].upper()
        plans.append({"iso_key": b["iso_key"], "side": "DOWN" if stop > tgt else "UP", "stop": stop, "target": tgt,
                      "start": start, "end": we, "system_result": "WIN" if "WIN" in txt else "LOSS" if "STOP" in txt or "LOSS" in txt else "?",
                      "broken_at": t})
        last_break[b["iso_key"]] = t
    return plans


def score(plan, bars, tol=TOL, strict=False):
    """returns (result, first_event_time). results: WIN, LOSS, NEUTRAL, AMBIGUOUS"""
    s, g, side = plan["stop"], plan["target"], plan["side"]
    seg = bars[(bars.index >= plan["start"].floor("h")) & (bars.index <= plan["end"])]
    touched = False
    for ts, b in seg.iterrows():
        if side == "DOWN":
            touch, breach, tgt = b["High"] >= s, b["High"] >= s * (1 + tol), b["Low"] <= g
        else:
            touch, breach, tgt = b["Low"] <= s, b["Low"] <= s * (1 - tol), b["High"] >= g
        if strict:
            if touch and tgt:
                return "AMBIGUOUS", ts
            if touch:
                return "LOSS", ts
            if tgt:
                return "WIN", ts
            continue
        if breach and tgt:
            return "AMBIGUOUS", ts
        if breach:
            return "LOSS", ts
        if tgt:
            return "WIN", ts
        if touch:
            touched = True
    return ("LOSS" if touched else "NEUTRAL"), None


def rates(res):
    w, l, a, n = (res.count(x) for x in ("WIN", "LOSS", "AMBIGUOUS", "NEUTRAL"))
    dec = w + l
    out = {"WIN": w, "LOSS": l, "AMBIGUOUS": a, "NEUTRAL": n}
    if dec + a:
        out["win_rate_ambiguous_as_loss"] = round(w / (dec + a), 3)
        out["win_rate_ambiguous_as_win"] = round((w + a) / (dec + a), 3)
    return out


def analyse(plans, bars):
    bad = bars[hp.bad_array(bars)].index if "Bad" in bars.columns else pd.DatetimeIndex([])
    rows, skipped = [], []
    for p in plans:
        seg_bad = ((bad >= p["start"].floor("h")) & (bad <= p["end"])).any() if len(bad) else False
        if seg_bad:
            skipped.append({"iso_key": p["iso_key"], "start": str(p["start"])[:16], "why": "glitch-cleaned bars inside the plan's span"})
            continue
        if bars.index.max() < p["start"] or bars.index.min() > p["start"]:
            skipped.append({"iso_key": p["iso_key"], "start": str(p["start"])[:16], "why": "no price data"})
            continue
        res, ts = score(p, bars)
        st, _ = score(p, bars, strict=True)
        rows.append({**{k: (str(v)[:16] if isinstance(v, pd.Timestamp) else v) for k, v in p.items()},
                     "her_rules": res, "strict_rule": st, "first_event": None if ts is None else str(ts)[:16]})
    her = [r["her_rules"] for r in rows]
    strict = [r["strict_rule"] for r in rows]
    sysres = [r["system_result"] for r in rows]
    conf = {}
    for r in rows:
        conf[f'{r["system_result"]} -> {r["her_rules"]}'] = conf.get(f'{r["system_result"]} -> {r["her_rules"]}', 0) + 1
    return {"plans_scored": len(rows), "plans_skipped": skipped,
            "HER_RULES": rates(her), "STRICT_RULE_stop_touch_is_loss": rates(strict),
            "system_log_as_recorded": {"WIN": sysres.count("WIN"), "LOSS": sysres.count("LOSS")},
            "how_system_label_maps_to_her_rules": dict(sorted(conf.items())), "per_plan": rows}


def main():
    bars = hp.fetch()
    breaks = json.load(open("weekly_break_log.json"))
    plans = build_plans(breaks)
    rep = analyse(plans, bars)
    rep["data"] = {"plans_in_log": len(breaks), "first_plan_week": plans[0]["iso_key"], "last_plan_week": plans[-1]["iso_key"],
                   "bars": len(bars), "to": str(bars.index.max())[:16]}
    txt = json.dumps(rep, indent=2, default=str)
    print(txt)
    open("setup_rescore_report.json", "w").write(txt)
    print("\nWrote setup_rescore_report.json (nothing else was changed).")


if __name__ == "__main__":
    main()
