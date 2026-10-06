#!/usr/bin/env python3
"""
stop_hunt_check.py  --  READ-ONLY research script (created 2026-10-06)

Question it answers: when a logged wheat setup (prediction_log.json, entries
that carry a real stop_price / target_price) failed, was the stop PIERCED
only briefly and the target reached afterwards ("hunt-like"), or did price
simply go against the setup and stay there?

It never writes to prediction_log.json, never sends Telegram, never commits.
Output: console tables + stop_hunt_report.json.

Method, per setup (one per calendar day / side / ~entry price, same dedup
idea as score_predictions.py):
  - side comes from the stop/target GEOMETRY (target > entry > stop = LONG,
    target < entry < stop = SHORT), NOT from the logged `direction` label,
    because since 2026-09-09 the label and the weekly setup can disagree.
    Those rows are flagged label_mismatch.
  - walk 15m bars after the log timestamp, up to LOOKFORWARD_DAYS days.
  - first_event: STOP_FIRST / TARGET_FIRST / SAME_BAR / NONE
  - pierce_cents: how far beyond the stop price reached within PIERCE_HOURS
    after the stop was first touched (the "hunt depth").
  - hunt_like: stop touched AND target reached later AND pierce_cents <=
    HUNT_MAX_PIERCE_CENTS.
  - sensitivity: the same bars replayed with the stop widened by N cents.

Resolution limit: Yahoo only serves ~60 days of 15m bars, so older setups
are reported as NO_DATA. Inside one 15m bar, order of high/low is unknown
(SAME_BAR is reported, never silently counted as a win).
"""
import json
import os
import sys
from datetime import timedelta

import pandas as pd

TICKER = "ZW=F"
LOG_FILE = os.getenv("PRED_LOG", "prediction_log.json")
CUTOFF_DATE = os.getenv("CUTOFF_DATE", "2026-08-23")          # stats_cutoff_date in wheat_monitor_state.json
LOOKFORWARD_DAYS = int(os.getenv("LOOKFORWARD_DAYS", "10"))   # same as score_predictions.py
PIERCE_HOURS = float(os.getenv("PIERCE_HOURS", "24"))
HUNT_MAX_PIERCE_CENTS = float(os.getenv("HUNT_MAX_PIERCE_CENTS", "5"))
WIDEN_LEVELS = [0, 3, 5, 8, 12, 16, 20]                       # cents added to the stop distance


def load_setups(path, cutoff):
    rows = json.load(open(path))
    seen, out = set(), []
    for r in rows:
        e, s, t = r.get("entry_price"), r.get("stop_price"), r.get("target_price")
        if None in (e, s, t):
            continue                      # old synthetic rows have no real levels
        if r["timestamp"][:10] < cutoff:
            continue
        key = (r["timestamp"][:10], round(e), round(s), round(t))
        if key in seen:
            continue
        seen.add(key)
        if t > e > s:
            side = "LONG"
        elif t < e < s:
            side = "SHORT"
        else:
            side = "ODD"
        out.append({
            "ts": pd.Timestamp(r["timestamp"]).tz_convert("UTC"),
            "label": r["direction"], "side": side,
            "entry": float(e), "stop": float(s), "target": float(t),
            "label_mismatch": (side == "LONG" and r["direction"] != "UP")
                              or (side == "SHORT" and r["direction"] != "DOWN"),
        })
    return out


def _hits(side, bar, stop, target):
    if side == "LONG":
        return bar["Low"] <= stop, bar["High"] >= target
    return bar["High"] >= stop, bar["Low"] <= target


def replay(side, entry, stop, target, bars):
    """First-event replay. Returns 'STOP_FIRST' | 'TARGET_FIRST' | 'SAME_BAR' | 'NONE'."""
    for _, bar in bars.iterrows():
        hs, ht = _hits(side, bar, stop, target)
        if hs and ht:
            return "SAME_BAR"
        if hs:
            return "STOP_FIRST"
        if ht:
            return "TARGET_FIRST"
    return "NONE"


def analyze_one(s, bars_all):
    res = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in s.items()}
    if s["side"] == "ODD":
        res["status"] = "ODD_GEOMETRY"
        return res
    end = s["ts"] + timedelta(days=LOOKFORWARD_DAYS)
    bars = bars_all[(bars_all.index > s["ts"]) & (bars_all.index <= end)]
    if bars.empty or bars_all.index.min() > s["ts"]:
        res["status"] = "NO_DATA"
        return res
    side, e, st, tg = s["side"], s["entry"], s["stop"], s["target"]
    res["status"] = "OK"
    res["first_event"] = replay(side, e, st, tg, bars)

    # Hunt analysis only applies when the STOP was hit before the target.
    # (TARGET_FIRST = the trade won; SAME_BAR = order unknown, reported as is.)
    res["stop_touched"] = res["first_event"] == "STOP_FIRST"
    res["target_reached_after_stop"] = False
    res["pierce_cents"] = None
    res["hunt_like"] = False
    if res["stop_touched"]:
        touched_idx = None
        for ts, bar in bars.iterrows():
            if _hits(side, bar, st, tg)[0]:
                touched_idx = ts
                break
        # first bar AFTER the touch that reaches the target (if any)
        target_ts = None
        for ts, bar in bars[bars.index > touched_idx].iterrows():
            if _hits(side, bar, st, tg)[1]:
                target_ts = ts
                break
        res["target_reached_after_stop"] = target_ts is not None
        # hunt depth = deepest excursion beyond the stop between the touch and
        # (target reached | PIERCE_HOURS later), whichever comes first
        stop_end = touched_idx + timedelta(hours=PIERCE_HOURS)
        if target_ts is not None:
            stop_end = min(stop_end, target_ts)
        win = bars[(bars.index >= touched_idx) & (bars.index <= stop_end)]
        if side == "LONG":
            res["pierce_cents"] = round(max(0.0, st - float(win["Low"].min())), 2)
        else:
            res["pierce_cents"] = round(max(0.0, float(win["High"].max()) - st), 2)
        res["hunt_like"] = bool(res["target_reached_after_stop"]
                                and res["pierce_cents"] <= HUNT_MAX_PIERCE_CENTS)

    # sensitivity: widen the stop, keep entry and target
    sens = {}
    for w in WIDEN_LEVELS:
        wst = st - w if side == "LONG" else st + w
        sens[str(w)] = replay(side, e, wst, tg, bars)
    res["sensitivity"] = sens
    return res


def summarize(results):
    ok = [r for r in results if r["status"] == "OK"]
    out = {"setups_total": len(results), "analyzed": len(ok),
           "no_data": sum(r["status"] == "NO_DATA" for r in results),
           "odd_geometry": sum(r["status"] == "ODD_GEOMETRY" for r in results),
           "label_mismatch_rows": sum(bool(r.get("label_mismatch")) for r in results)}
    if not ok:
        return out
    fe = pd.Series([r["first_event"] for r in ok]).value_counts().to_dict()
    touched = [r for r in ok if r["stop_touched"]]   # stop hit BEFORE target
    out.update({
        "first_event_counts": fe,
        "stop_touched": len(touched),
        "target_reached_after_stop": sum(r["target_reached_after_stop"] for r in touched),
        "hunt_like": sum(r["hunt_like"] for r in ok),
        "median_pierce_cents": (round(float(pd.Series([r["pierce_cents"] for r in touched]).median()), 2)
                                if touched else None),
        "hunt_rule": f"stop touched AND target reached later AND pierce <= {HUNT_MAX_PIERCE_CENTS}c within {PIERCE_HOURS}h",
    })
    sens = {}
    for w in WIDEN_LEVELS:
        c = pd.Series([r["sensitivity"][str(w)] for r in ok]).value_counts().to_dict()
        sens[f"stop +{w}c"] = {k: int(c.get(k, 0)) for k in ("TARGET_FIRST", "STOP_FIRST", "SAME_BAR", "NONE")}
    out["sensitivity"] = sens
    return out


def fetch_bars():
    import yfinance as yf
    df = yf.download(TICKER, period="60d", interval="15m", progress=False, auto_adjust=False)
    if df is None or df.empty:
        sys.exit("No 15m data returned from Yahoo - try again later.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df[["High", "Low", "Close"]].dropna()


def main():
    setups = load_setups(LOG_FILE, CUTOFF_DATE)
    print(f"Setups with real stop/target since {CUTOFF_DATE}: {len(setups)}")
    bars = fetch_bars()
    print(f"15m bars: {len(bars)}  {bars.index.min()} -> {bars.index.max()}\n")
    results = [analyze_one(s, bars) for s in setups]
    summary = summarize(results)

    print(f"{'logged (UTC)':17s} {'label':5s} {'side':5s} {'entry':>7s} {'stop':>7s} {'target':>7s} "
          f"{'first_event':13s} {'pierce':>6s} hunt  note")
    for r in results:
        note = "label!=geometry" if r.get("label_mismatch") else ""
        print(f"{r['ts'][:16]:17s} {r['label']:5s} {r['side']:5s} {r['entry']:7.2f} {r['stop']:7.2f} {r['target']:7.2f} "
              f"{r.get('first_event', r['status']):13s} {str(r.get('pierce_cents', '')):>6s} "
              f"{'YES' if r.get('hunt_like') else '':4s}  {note}")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "setups": results}, open("stop_hunt_report.json", "w"), indent=2, default=str)
    print("\nWrote stop_hunt_report.json (nothing else was changed).")


if __name__ == "__main__":
    main()
