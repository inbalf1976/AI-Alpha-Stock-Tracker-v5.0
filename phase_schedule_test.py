#!/usr/bin/env python3
"""
phase_schedule_test.py  --  READ-ONLY research (created 2026-10-07)

Tests the daily schedule described by the user (all times Israel):
  P0  03:00-09:00  "AI follows the natural movement"
  P1  09:00-11:00  slow push toward the human stop clusters
  P2  11:00-15:00  keep pushing or reverse depending on how people reacted (to ~14:30; hourly bars end at 15:00)
  P4  19:00-21:00  second, high-value move (18:30 start; hourly bars begin at 19:00)
Questions, one per claim:
  T1  does P0 direction predict P1 direction?
  T2  does P1 direction predict P2 direction (continue or reverse)?
  T3  does P2 direction predict P4 direction?
  T4  P1 -> P2 conditional on what P1 did at the nearest prior day/week level:
      passed it / touched and was rejected / no touch.
      Claim: pass -> keep pushing; rejection -> reverse.
  T5  intraday profile: is P2 busier (volume) than P1? is P4 on Fridays bigger?

Method: one row per Israel calendar day (Mon-Fri), using only days where every bar involved
exists and none is flagged 'Bad' by the glitch cleaning. A direction test counts as supported
only if exact two-sided binomial p < 0.01 AND the rate is at least 5 points from 50%.
(About 8 tests are run, so p < 0.05 alone would be expected by chance.)

Never writes repo state, never sends Telegram, never commits.
Needs hunter_pattern_research.py in the same folder.
"""
import json
import math
import os

import numpy as np
import pandas as pd

import hunter_pattern_research as hp

LEVEL_NEAR_PCT = float(os.getenv("LEVEL_NEAR_PCT", "2.0"))     # only levels within this % of the P1 start
TOUCH_PCT = float(os.getenv("TOUCH_PCT", "0.10"))
PASS_PCT = float(os.getenv("PASS_PCT_T4", "0.20"))
PHASES = {"P0": (3, 9), "P1": (9, 11), "P2": (11, 15), "P4": (19, 21)}


def binom_p(k, n):
    """exact two-sided binomial p against 0.5"""
    if n == 0:
        return None
    pk = [math.comb(n, i) for i in range(n + 1)]
    tot = 2.0 ** n
    lo = sum(pk[:k + 1]) / tot
    hi = sum(pk[k:]) / tot
    return min(1.0, 2 * min(lo, hi))


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": round(k / n, 3) if n else None,
            "ci95": hp.wilson(k, n), "p": None if n == 0 else round(binom_p(k, n), 4)}


def supported(r):
    return bool(r["n"] >= 30 and r["p"] is not None and r["p"] < 0.01 and abs(r["rate"] - 0.5) >= 0.05)


def build_days(h):
    il = h.index.tz_convert(hp.TZ_IL)
    bad = hp.bad_array(h)
    key = pd.Series(range(len(h)), index=pd.MultiIndex.from_arrays([il.date, il.hour]))
    key = key[~key.index.duplicated(keep="first")]
    tdates = np.array(hp.trade_date(h.index), dtype=object)
    levels = hp.levels_for(h)
    o, hi, lo = h["Open"].values, h["High"].values, h["Low"].values
    vol = h["Volume"].values.astype(float)
    rows = []
    for d in sorted({x for x in il.date}):
        if pd.Timestamp(d).weekday() > 4:
            continue
        try:
            pos = {nm: (key[(d, a)], key[(d, b)]) for nm, (a, b) in PHASES.items()}
        except KeyError:
            continue
        lo_i = min(p[0] for p in pos.values())
        hi_i = max(p[1] for p in pos.values())
        if bad[lo_i:hi_i + 1].any():
            continue
        r = {"date": d, "weekday": pd.Timestamp(d).strftime("%a")}
        for nm, (i0, i1) in pos.items():
            r[nm] = (o[i1] / o[i0] - 1) * 100
            r[nm + "_hi"] = hi[i0:i1].max()
            r[nm + "_lo"] = lo[i0:i1].min()
            r[nm + "_vol"] = vol[i0:i1].mean()
        # P1 vs nearest level in the direction of P1
        i0, i1 = pos["P1"]
        start, end = o[i0], o[i1]
        lv = levels.get(tdates[i0], {})
        up = end >= start
        cand = [L for n, L in lv.items() if (n in ("PDH", "PWH") if up else n in ("PDL", "PWL"))
                and ((L >= start and (L - start) / start * 100 <= LEVEL_NEAR_PCT) if up else (L <= start and (start - L) / start * 100 <= LEVEL_NEAR_PCT))]
        grp = "no_level_nearby"
        if cand:
            L = min(cand) if up else max(cand)
            touched = (r["P1_hi"] >= L * (1 - TOUCH_PCT / 100)) if up else (r["P1_lo"] <= L * (1 + TOUCH_PCT / 100))
            if not touched:
                grp = "no_touch"
            else:
                beyond = (end >= L * (1 + PASS_PCT / 100)) if up else (end <= L * (1 - PASS_PCT / 100))
                back = (end <= L * (1 - PASS_PCT / 100)) if up else (end >= L * (1 + PASS_PCT / 100))
                grp = "passed" if beyond else "rejected" if back else "touched_inside"
        r["p1_group"] = grp
        rows.append(r)
    return pd.DataFrame(rows)


def pair_test(df, a, b):
    d = df[(df[a].abs() > 0.05) & (df[b].abs() > 0.05)]
    same = int((np.sign(d[a]) == np.sign(d[b])).sum())
    out = {"days": int(len(d)), "same_direction": rate(same, len(d))}
    rng = np.random.default_rng(3)
    if len(d) > 10:
        ra, rb = d[a].rank().values, d[b].rank().values
        corr = float(np.corrcoef(ra, rb)[0, 1])
        perm = [abs(np.corrcoef(ra, rng.permutation(rb))[0, 1]) for _ in range(2000)]
        out["spearman"] = round(corr, 3)
        out["spearman_p"] = round((1 + sum(x >= abs(corr) for x in perm)) / 2001, 4)
    out["verdict"] = ("SUPPORTED (%s)" % ("they continue" if out["same_direction"]["rate"] > 0.5 else "they reverse")
                      ) if supported(out["same_direction"]) else "not supported"
    return out


def main():
    h = hp.fetch()
    df = build_days(h)
    print(f"usable days: {len(df)}")
    rep = {"usable_days": int(len(df)), "from": str(df["date"].min()), "to": str(df["date"].max())}
    rep["T1_P0_to_P1"] = pair_test(df, "P0", "P1")
    rep["T2_P1_to_P2"] = pair_test(df, "P1", "P2")
    rep["T3_P2_to_P4"] = pair_test(df, "P2", "P4")

    t4 = {}
    for g, sub in df.groupby("p1_group"):
        d = sub[(sub["P1"].abs() > 0.05) & (sub["P2"].abs() > 0.05)]
        same = int((np.sign(d["P1"]) == np.sign(d["P2"])).sum())
        t4[g] = {"days": int(len(d)), "P2_continues_P1": rate(same, len(d))}
        t4[g]["verdict"] = ("SUPPORTED (%s)" % ("continues" if t4[g]["P2_continues_P1"]["rate"] > 0.5 else "reverses")
                            ) if supported(t4[g]["P2_continues_P1"]) else "not supported"
    rep["T4_P1_to_P2_by_what_P1_did_at_level"] = t4
    rep["T4_claim"] = "passed -> P2 continues ; rejected -> P2 reverses"

    # T5: intraday profile and Friday check
    il = h.index.tz_convert(hp.TZ_IL)
    prof = pd.DataFrame({"hour": il.hour, "absret": ((h["Close"] / h["Open"] - 1).abs() * 100).values,
                         "vol": h["Volume"].values.astype(float), "day": il.date})[~hp.bad_array(h)]
    prof["vol_rel"] = prof["vol"] / prof.groupby("day")["vol"].transform("mean").replace(0, np.nan)
    ph = prof.groupby("hour").agg(mean_abs_ret_pct=("absret", "mean"), mean_volume_vs_day=("vol_rel", "mean"), bars=("absret", "size"))
    rep["T5_intraday_profile_IL_hour"] = {int(k): {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in ph.to_dict("index").items()}
    v1 = prof[prof.hour.isin([9, 10])]["vol_rel"].mean()
    v2 = prof[prof.hour.isin([11, 12, 13, 14])]["vol_rel"].mean()
    rep["T5_volume_P2_vs_P1"] = {"P1_hours_9_10": round(float(v1), 3), "P2_hours_11_14": round(float(v2), 3),
                                 "claim": "P2 has higher volume than P1"}
    fri = df[df.weekday == "Fri"]["P4"].abs()
    oth = df[df.weekday != "Fri"]["P4"].abs()
    rep["T5_P4_abs_move_pct"] = {"fridays": round(float(fri.mean()), 3), "other_days": round(float(oth.mean()), 3),
                                 "n_fri": int(len(fri)), "n_other": int(len(oth))}
    rep["T5_mean_abs_move_pct_by_phase"] = {nm: round(float(df[nm].abs().mean()), 3) for nm in PHASES}
    txt = json.dumps(rep, indent=2, default=str)
    print(txt)
    open("phase_schedule_report.json", "w").write(txt)
    print("\nWrote phase_schedule_report.json (nothing else was changed).")


if __name__ == "__main__":
    main()
