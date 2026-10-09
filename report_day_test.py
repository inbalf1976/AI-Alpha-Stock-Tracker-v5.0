#!/usr/bin/env python3
"""
report_day_test.py  --  READ-ONLY research (created 2026-10-08)

Question: are USDA WASDE days really bigger movers than normal days, at the release hour
(12:00 ET = 19:00 Israel while Israel is on summer time) and for the rest of the day?

Method (cleaned hourly ZW=F bars, same cleaning as the other research scripts):
  * release bar    = the hourly bar that starts at 12:00 ET on a WASDE date
  * release window = 12:00-14:20 ET (bars 12, 13, 14) - the CBOT day session ends 14:20 ET, there is no 15:00 bar
  * day range      = high-low from 00:00 to 14:59 ET of that day (overnight + day session; the evening session that
                     starts at 20:00 ET belongs to the next trade date and is left out)
  * each is compared with the SAME measure on all other weekdays (same hour / same window),
    p-value = share of 10,000 random draws of that many normal days that average at least as large
  * dates whose day contains contaminated (glitch-cleaned) bars are skipped and listed
  * continuation check: does the move after the first release hour (13:00-14:20 ET)
    go the same way as the release hour?

WASDE dates used (12:00 ET). 2024: FX Blue list + CME article. 2026: USDA site.
2025: Jan 10 (DTN), Sep 12 and Dec 9 (calendar sites), Nov 14 (USDA reset after the shutdown; no October
report); Feb-Aug 2025 come from a third-party calendar and were not checked against USDA - treat as
'probably right'. Oct 9 2026 has not happened yet and is excluded.

Never writes repo state, never sends Telegram, never commits.
Needs hunter_pattern_research.py in the same folder.
"""
import json
import math

import numpy as np
import pandas as pd

import hunter_pattern_research as hp

WASDE = [
    # 2024
    "2024-06-12", "2024-07-12", "2024-08-12", "2024-09-12", "2024-10-11", "2024-11-08", "2024-12-10",
    # 2025 (no October report - shutdown; Nov moved to the 14th)
    "2025-01-10", "2025-02-11", "2025-03-11", "2025-04-10", "2025-05-12", "2025-06-12", "2025-07-11",
    "2025-08-12", "2025-09-12", "2025-11-14", "2025-12-09",
    # 2026
    "2026-01-12", "2026-02-10", "2026-03-10", "2026-04-09", "2026-05-12", "2026-06-11", "2026-07-10",
    "2026-08-12", "2026-09-11",
]
UNVERIFIED = {"2025-02-11", "2025-03-11", "2025-04-10", "2025-05-12", "2025-06-12", "2025-07-11", "2025-08-12"}
DRAWS = 10000
RNG = np.random.default_rng(11)


def build(h):
    et = h.index.tz_convert("America/New_York")
    d = pd.DataFrame({"date": et.date, "hour": et.hour, "o": h["Open"].values, "h": h["High"].values,
                      "l": h["Low"].values, "c": h["Close"].values, "bad": hp.bad_array(h)})
    d["ret"] = (d.c / d.o - 1) * 100
    d["rng"] = (d.h - d.l) / d.o * 100
    return d


def per_day(d):
    """one row per ET date with release-hour, 12:00-14:20 window and day-session measures."""
    rows = []
    for dt, g in d.groupby("date"):
        if g["bad"].any() or pd.Timestamp(dt).weekday() > 4:
            continue
        gs = g[g["hour"] <= 14]                       # overnight + day session of this trade date
        gi = gs.set_index("hour")
        if gi.index.duplicated().any() or not all(x in gi.index for x in (12, 13, 14)):
            continue
        o12, c14, o13 = gi.loc[12, "o"], gi.loc[14, "c"], gi.loc[13, "o"]
        rows.append({"date": dt, "rel_abs": abs(gi.loc[12, "ret"]), "rel_ret": gi.loc[12, "ret"],
                     "rel_rng": gi.loc[12, "rng"], "win_abs": abs((c14 / o12 - 1) * 100),
                     "after_ret": (c14 / o13 - 1) * 100,
                     "day_rng": (gs["h"].max() - gs["l"].min()) / gs["o"].iloc[0] * 100})
    return pd.DataFrame(rows)


def compare(rep, base, col):
    obs = float(rep[col].mean())
    nrep = len(rep)
    means = np.array([RNG.choice(base[col].values, size=nrep, replace=False).mean() for _ in range(DRAWS)])
    return {"report_days_mean": round(obs, 3), "normal_days_mean": round(float(base[col].mean()), 3),
            "ratio": round(obs / float(base[col].mean()), 2),
            "p_value": round((1 + int((means >= obs).sum())) / (1 + DRAWS), 4)}


def main():
    h = hp.fetch()
    days = per_day(build(h))
    if days.empty:
        raise SystemExit('No usable days: no ET hours 12, 13 and 14 found together. Check the bar timestamps.')
    wd = [pd.Timestamp(x).date() for x in WASDE]
    rep = days[days["date"].isin(wd)].copy()
    base = days[~days["date"].isin(wd)].copy()
    skipped = [str(x) for x in wd if x not in set(days["date"])]
    out = {"report_days_used": int(len(rep)), "report_days_skipped_(glitch_or_missing_bars)": skipped,
           "unverified_dates_used": sorted(str(x) for x in rep["date"] if str(x) in UNVERIFIED),
           "normal_days": int(len(base))}
    if len(rep) >= 5:
        out["release_hour_abs_move_pct"] = compare(rep, base, "rel_abs")
        out["release_hour_range_pct"] = compare(rep, base, "rel_rng")
        out["window_12_to_1420ET_abs_move_pct"] = compare(rep, base, "win_abs")
        out["whole_day_range_pct"] = compare(rep, base, "day_rng")
        r = rep[(rep.rel_ret.abs() > 0.05) & (rep.after_ret.abs() > 0.05)]
        same = int((np.sign(r.rel_ret) == np.sign(r.after_ret)).sum())
        n = len(r)
        pk = [math.comb(n, i) for i in range(n + 1)]
        p = min(1.0, 2 * min(sum(pk[:same + 1]), sum(pk[same:])) / 2.0 ** n) if n else None
        out["after_release_continues_first_hour"] = {"same_direction": same, "n": n,
                                                    "p_value": None if p is None else round(p, 4)}
        out["per_date"] = [{"date": str(r_["date"]), "release_hour_ret_pct": round(r_["rel_ret"], 2),
                            "move_next_3h_pct": round(r_["after_ret"], 2), "day_range_pct": round(r_["day_rng"], 2)}
                           for _, r_ in rep.iterrows()]
        out["verdict_rule"] = "report days differ only if p < 0.01 AND ratio >= 1.3"
        out["verdicts"] = {k: ("MORE VOLATILE on report days" if (out[k]["p_value"] < 0.01 and out[k]["ratio"] >= 1.3)
                               else "no clear difference")
                           for k in ("release_hour_abs_move_pct", "window_12_to_1420ET_abs_move_pct", "whole_day_range_pct")}
    txt = json.dumps(out, indent=2, default=str)
    print(txt)
    open("report_day_report.json", "w").write(txt)
    print("\nWrote report_day_report.json (nothing else was changed).")


if __name__ == "__main__":
    main()
