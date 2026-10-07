#!/usr/bin/env python3
"""
fade_robustness.py  --  READ-ONLY research (created 2026-10-07)

Stress-tests the one cell that won in level_fade_backtest.py
(stop 1.5% / target 2.0% / US session) and its neighbours:

  1. thirds of the period      - is the edge steady or only in one stretch?
  2. per level / per side      - which levels carry it (small n, read with care)
  3. roll-gap exclusion        - drop trades near suspected contract-roll jumps
  4. per-DAY view              - several touches on one day are one bet, not many
                                 (day-level mean, t-stat, bootstrap 95% interval)
  5. costs                     - Plus500 spread (default 0.071% = 0.50c on ~704c)
                                 plus wider-spread cases

Needs hunter_pattern_research.py and level_fade_backtest.py in the same folder.
Never writes repo state, never sends Telegram, never commits.
"""
import json
import os

import numpy as np
import pandas as pd

import hunter_pattern_research as hp
import level_fade_backtest as lf

CELLS = [(1.5, 2.0), (2.0, 2.0), (1.5, 1.5), (1.0, 2.0), (2.0, 1.5)]   # first = the winner
COST_PCT = float(os.getenv("COST_PCT", "0.071"))     # round-trip spread, % of price
GAP_PCT = float(os.getenv("GAP_PCT", "1.2"))         # bar-to-bar jump that flags a possible roll
ROLL_EXCLUDE_DAYS = int(os.getenv("ROLL_EXCLUDE_DAYS", "7"))   # calendar days after a flagged jump


def find_roll_jumps(df, gap_pct=GAP_PCT):
    prev_c = df["Close"].shift(1)
    gap = (df["Open"] - prev_c).abs() / prev_c * 100
    hrs = df.index.to_series().diff().dt.total_seconds() / 3600
    # only jumps between consecutive-ish bars (<= 4h apart) - weekend/overnight
    # gaps in real trading are not roll artefacts
    flagged = gap[(gap >= gap_pct) & (hrs <= 4)]
    return [(ts, round(float(v), 2)) for ts, v in flagged.items()]


def trades_for(df, touches, stop, target, session="US"):
    sel = [x for x in touches if session == "ALL" or (session == "US") == x["us"]]
    rows = []
    for x in sel:
        rows.append({"ts": x["ts"], "level": x["level"], "side": x["side"],
                     "pnl": lf.simulate(df, x, stop, target)})
    return pd.DataFrame(rows)


def stats(d, cost=COST_PCT):
    if len(d) == 0:
        return {"n": 0}
    pn = d["pnl"].values
    return {"n": int(len(d)), "win_rate": round(float((pn > 0).mean()), 3),
            "gross_exp_pct": round(float(pn.mean()), 3), "net_exp_pct": round(float(pn.mean() - cost), 3)}


def per_day(d, cost=COST_PCT, reps=4000, seed=1):
    if len(d) < 10:
        return {"days": int(len(d))}
    day = d.assign(day=d["ts"].dt.tz_convert(hp.TZ_CT).dt.date).groupby("day")["pnl"].mean() - cost
    v = day.values
    t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 1 and v.std() > 0 else None
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(reps)])
    return {"trading_days_with_trades": int(len(v)), "mean_net_per_day_pct": round(float(v.mean()), 3),
            "t_stat": None if t is None else round(t, 2),
            "bootstrap95_net_pct": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
            "share_of_days_positive": round(float((v > 0).mean()), 3)}


def analyse(h):
    levels = hp.levels_for(h)
    touches = lf.list_touches(h, levels)
    jumps = find_roll_jumps(h)
    bad_days = set()
    for ts, _ in jumps:
        for k in range(ROLL_EXCLUDE_DAYS + 1):
            bad_days.add((ts + pd.Timedelta(days=k)).tz_convert(hp.TZ_CT).date())
    out = {"cost_pct_used": COST_PCT, "suspected_roll_jumps": [{"at": str(t), "gap_pct": g} for t, g in jumps],
           "cells": {}}
    for (s, g) in CELLS:
        d = trades_for(h, touches, s, g)
        c = {"all_trades": stats(d)}
        if len(d) == 0:
            out["cells"][f"stop{s}_target{g}"] = c
            continue
        edges = np.linspace(0, len(d), 4).astype(int)
        d = d.sort_values("ts").reset_index(drop=True)
        c["thirds"] = [stats(d.iloc[edges[i]:edges[i + 1]]) | {"from": str(d["ts"].iloc[edges[i]])[:10]} for i in range(3)]
        c["by_level"] = {k: stats(x) for k, x in d.groupby("level")}
        c["by_side"] = {"short_fades(R)": stats(d[d.side == "R"]), "long_fades(S)": stats(d[d.side == "S"])}
        keep = d[~d["ts"].dt.tz_convert(hp.TZ_CT).dt.date.isin(bad_days)]
        c["excluding_roll_windows"] = stats(keep)
        c["per_day_view"] = per_day(d)
        c["cost_cases_net_exp_pct"] = {f"spread_{x:.3f}pct": round(float(d["pnl"].mean() - x), 3)
                                       for x in (0.0, COST_PCT, 0.10, 0.15, 0.20)}
        out["cells"][f"stop{s}_target{g}"] = c
    return out


def main():
    h = hp.fetch()
    print(f"hourly bars: {len(h)}  {h.index.min()} -> {h.index.max()}")
    rep = analyse(h)
    txt = json.dumps(rep, indent=2, default=str)
    print(txt)
    open("fade_robustness_report.json", "w").write(txt)
    print("\nWrote fade_robustness_report.json (nothing else was changed).")


if __name__ == "__main__":
    main()
