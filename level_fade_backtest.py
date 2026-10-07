#!/usr/bin/env python3
"""
level_fade_backtest.py  --  READ-ONLY research (created 2026-10-07)

Question: the hunter research showed prior-day/week highs/lows and round-50
levels act as barriers (price passes them less often than random). Does
FADING the first touch of such a level actually make money after costs?

Rule tested (all parameters in a small grid, nothing hand-tuned):
  - level = PDH, PDL, PWH, PWL, R50 (same definitions as hunter_pattern_research.py)
  - first touch per trade date and level; approached from below -> SHORT at the
    level, approached from above -> LONG at the level (limit fill at the level)
  - stop STOP_PCT beyond the level, target TARGET_PCT back, time exit after
    HOLD_BARS hourly bars at the close
  - conservative bar handling (hourly bars hide the real order of events):
      * entry bar: stop counts as hit if the bar reached it; target only if the
        bar CLOSED at/over it
      * later bars: if stop and target are both inside one bar -> STOP
      * gaps through the stop fill at the open (worse price)
  - grid: stop {0.5,1,1.5,2}% x target {0.5,1,1.5,2}% x session {ALL, US, OVERNIGHT}
    (US = CT 08:00-13:59, the CBOT day session)

Honesty guards:
  1. Split-sample: best cell is picked on the FIRST half of the data and judged
     on the SECOND half only.
  2. Selection-aware null: the same 48-cell search is run on bootstrap price
     paths with no structure. The real best result must beat what the best of
     48 cells achieves on pure noise (p-value reported).
  3. Costs: break-even cost is printed; compare it with your REAL Plus500 spread.

Needs hunter_pattern_research.py in the same folder (imports its helpers).
Never writes to repo state, never sends Telegram, never commits.
"""
import itertools
import json
import os

import numpy as np
import pandas as pd

import hunter_pattern_research as hp

STOPS = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0]
SESSIONS = ["ALL", "US", "OVERNIGHT"]
HOLD_BARS = int(os.getenv("HOLD_BARS", "24"))
MIN_TRADES = int(os.getenv("MIN_TRADES", "40"))
NULL_REPS = int(os.getenv("NULL_REPS", "30"))
COSTS_PCT = [0.0, 0.05, 0.10, 0.20]       # round-trip cost as % of price (700 -> 0.35c per 0.05%)


def list_touches(df, levels):
    """first touch per trade date and level; side R = approached from below."""
    tdates = hp.trade_date(df.index)
    ct_hour = df.index.tz_convert(hp.TZ_CT).hour
    o, h, l = df["Open"].values, df["High"].values, df["Low"].values
    seen, out = set(), []
    for t in range(1, len(df) - 1):
        td = tdates[t]
        lv = dict(levels.get(td, {}))
        lv.update(hp.round_levels(o[t]))
        for name, L in lv.items():
            if (td, name) in seen:
                continue
            side = "R" if o[t] < L <= h[t] else "S" if o[t] > L >= l[t] else None
            if side is None:
                continue
            seen.add((td, name))
            out.append({"t": t, "L": float(L), "side": side, "level": name,
                        "us": 8 <= ct_hour[t] < 14, "ts": df.index[t]})
    return out


def simulate(df, tc, stop_pct, target_pct, hold=HOLD_BARS):
    """returns pnl in percent of the level price for one fade trade."""
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    t, L = tc["t"], tc["L"]
    short = tc["side"] == "R"
    sp, gp = stop_pct / 100, target_pct / 100
    stop = L * (1 + sp) if short else L * (1 - sp)
    tgt = L * (1 - gp) if short else L * (1 + gp)
    # entry bar
    if (short and h[t] >= stop) or (not short and l[t] <= stop):
        return -stop_pct
    if (short and c[t] <= tgt) or (not short and c[t] >= tgt):
        return target_pct
    last = min(t + hold, len(df) - 1)
    for j in range(t + 1, last + 1):
        if short:
            hit_stop, hit_tgt = h[j] >= stop, l[j] <= tgt
            if hit_stop:                                  # stop wins ties
                fill = max(stop, o[j])
                return -(fill - L) / L * 100
        else:
            hit_stop, hit_tgt = l[j] <= stop, h[j] >= tgt
            if hit_stop:
                fill = min(stop, o[j])
                return -(L - fill) / L * 100
        if hit_tgt:
            return target_pct
    return ((L - c[last]) / L * 100) if short else ((c[last] - L) / L * 100)


def grid(df, touches):
    """expectancy per cell. returns {(stop,target,session): {n, win, exp, pnls-by-time}}"""
    res = {}
    for sess in SESSIONS:
        sel = [x for x in touches if sess == "ALL" or (sess == "US") == x["us"]]
        for s, g in itertools.product(STOPS, TARGETS):
            pn = np.array([simulate(df, x, s, g) for x in sel])
            res[(s, g, sess)] = {"n": len(sel), "pnl": pn, "ts": [x["ts"] for x in sel]}
    return res


def cell_stats(pn):
    if len(pn) == 0:
        return {"n": 0, "win_rate": None, "exp_pct": None}
    return {"n": int(len(pn)), "win_rate": round(float((pn > 0).mean()), 3), "exp_pct": round(float(pn.mean()), 4)}


def best_cell(res, subset=None):
    best, bk = None, None
    for k, v in res.items():
        pn = v["pnl"] if subset is None else v["pnl"][subset(v["ts"])]
        if len(pn) < MIN_TRADES:
            continue
        e = pn.mean()
        if best is None or e > best:
            best, bk = e, k
    return bk, best


def split_mask(ts_list, cut):
    ts = pd.DatetimeIndex(ts_list)
    return np.asarray(ts < cut), np.asarray(ts >= cut)


def run(df):
    levels = hp.build_levels(hp.daily_from_hourly(df))
    touches = list_touches(df, levels)
    return grid(df, touches), touches


def main():
    h = hp.fetch()
    print(f"hourly bars: {len(h)}  {h.index.min()} -> {h.index.max()}")
    real, touches = run(h)
    cut = h.index[len(h) // 2]

    # ---- full-sample table ----
    table = []
    for (s, g, sess), v in real.items():
        st = cell_stats(v["pnl"])
        table.append({"stop_pct": s, "target_pct": g, "session": sess, **st})
    table.sort(key=lambda r: -(r["exp_pct"] if r["exp_pct"] is not None and r["n"] >= MIN_TRADES else -9))
    print("\nTOP 8 cells, FULL sample (expectancy per trade in % of price, gross of costs):")
    for r in table[:8]:
        print(r)

    # ---- split sample: pick on first half, judge on second ----
    k1, e1 = best_cell(real, subset=lambda ts: split_mask(ts, cut)[0])
    out = {"data": {"hourly_bars": len(h), "from": str(h.index.min()), "to": str(h.index.max()), "split_at": str(cut)},
           "touches_total": len(touches)}
    if k1:
        v = real[k1]
        first, second = split_mask(v["ts"], cut)
        out["split_sample"] = {"picked_on_first_half": {"stop_pct": k1[0], "target_pct": k1[1], "session": k1[2],
                                                         **cell_stats(v["pnl"][first])},
                               "judged_on_second_half": cell_stats(v["pnl"][second])}
        print("\nSPLIT SAMPLE  picked on 1st half:", out["split_sample"]["picked_on_first_half"])
        print("              judged on 2nd half:", out["split_sample"]["judged_on_second_half"])

    # ---- selection-aware null ----
    kb, eb = best_cell(real)
    nulls = []
    for seed in range(NULL_REPS):
        nd = hp.bootstrap_null(h, seed)
        nres, _ = run(nd)
        _, ne = best_cell(nres)
        nulls.append(ne if ne is not None else np.nan)
    nulls = np.array([x for x in nulls if not np.isnan(x)])
    if kb and len(nulls):
        p = (1 + int((nulls >= eb).sum())) / (1 + len(nulls))
        out["null_test"] = {
            "real_best_cell": {"stop_pct": kb[0], "target_pct": kb[1], "session": kb[2], **cell_stats(real[kb]["pnl"])},
            "null_best_exp_pct": {"mean": round(float(nulls.mean()), 4), "max": round(float(nulls.max()), 4), "reps": int(len(nulls))},
            "p_value_best_of_48": round(p, 3),
            "note": "p = share of structure-free price paths whose BEST of 48 cells did at least as well as the real best",
        }
        pn = real[kb]["pnl"]
        out["cost_sensitivity_best_cell"] = {f"cost_{c:.2f}pct": round(float(pn.mean() - c), 4) for c in COSTS_PCT}
        out["break_even_cost_pct"] = round(float(pn.mean()), 4)
        print("\nNULL TEST:", json.dumps(out["null_test"], indent=1))
        print("COST SENSITIVITY (best cell, expectancy % per trade after cost):", out["cost_sensitivity_best_cell"])
    out["top_cells_full_sample"] = table[:12]
    txt = json.dumps(out, indent=2, default=str)
    open("level_fade_report.json", "w").write(txt)
    print("\nWrote level_fade_report.json (nothing else was changed).")


if __name__ == "__main__":
    main()
