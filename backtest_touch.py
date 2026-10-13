#!/usr/bin/env python3
"""
backtest_touch.py -- touch-only re-score of backtest.py (READ-ONLY w.r.t. the live system).

WHY: backtest.py's compute_outcomes() scores a day as a WIN when neither the stop nor the target
is touched but the last hourly close is above the entry. With a 1.5% stop / 2.5% target (~10c / ~17c)
that happens on most days, so the reported "accuracy" (and the 67.6% baseline) is mostly an
"up day" rate, not a target-before-stop rate.

HERE: a win = target touched first, a loss = stop touched first (or both inside the same hourly bar,
conservative), and a day where NEITHER is touched inside the window is EXCLUDED (NaN), not judged by
the close. UP and DOWN are scored independently (backtest.py's same-bar `break` also cut the DOWN
scoring short on those days). Accuracy is reported only over decided days, with Wilson 95% intervals,
the break-even rate for this stop/target, and expectancy in R (stop units).

SAFETY: imports backtest.py's data/feature code but never calls its main(), so it does NOT overwrite
backtest_results.json or validated_conditions.json (the file ConvictionGate loads). Outputs go to
backtest_touch_results.json and backtest_touch_conditions.csv only.

Env: HORIZON_DAYS (default 1 = next trading day, same as backtest.py; use 5 for a weekly-style window)
"""
import os, json, math
import numpy as np
import pandas as pd
import backtest as bt

HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "1"))
S, T = bt.STOP_PCT, bt.TARGET_PCT
BREAKEVEN = S / (S + T)                      # win rate needed at this stop/target (37.5%)
CONDITIONS = ['above_sma20', 'above_sma50', 'trend_aligned', 'rsi_oversold', 'rsi_overbought', 'rsi_neutral',
              'macd_bullish', 'vol_good', 'vol_high', 'vol_low', 'near_bb_lower', 'near_bb_upper', 'inside_bb',
              'wc_bullish', 'wc_bearish', 'in_lower_half', 'in_upper_half', 'momentum_up', 'momentum_down',
              'bullish_month', 'bearish_month']
VOLUME_SUSPECT = {'vol_low', 'vol_good', 'vol_high'}   # Yahoo ZW=F volume lag, see ConvictionGate docstring


def first_touch(hi, lo, entry, side):
    """1.0 target first, 0.0 stop first / same bar, NaN neither."""
    if side == "UP":
        tgt, stp = np.flatnonzero(hi >= entry * (1 + T)), np.flatnonzero(lo <= entry * (1 - S))
    else:
        tgt, stp = np.flatnonzero(lo <= entry * (1 - T)), np.flatnonzero(hi >= entry * (1 + S))
    ti = tgt[0] if tgt.size else None
    si = stp[0] if stp.size else None
    if ti is None and si is None:
        return np.nan
    if ti is None:
        return 0.0
    if si is None:
        return 1.0
    return 1.0 if ti < si else 0.0


def compute_outcomes_touch(daily, hourly):
    dd = daily.index.normalize().unique()
    hd = hourly.index.normalize().values
    hi, lo = hourly["High"].values, hourly["Low"].values
    up, dn = {}, {}
    for i in range(len(dd) - 1 - HORIZON_DAYS):
        d = dd[i]
        entry = float(daily.loc[daily.index.normalize() == d, "Close"].iloc[-1])
        a = np.searchsorted(hd, np.datetime64(dd[i + 1]), side="left")
        b = np.searchsorted(hd, np.datetime64(dd[i + HORIZON_DAYS]), side="right")
        if b <= a:
            continue
        up[d] = first_touch(hi[a:b], lo[a:b], entry, "UP")
        dn[d] = first_touch(hi[a:b], lo[a:b], entry, "DOWN")
    return pd.Series(up, dtype=float), pd.Series(dn, dtype=float)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def summarize(res):
    dec = res.dropna()
    n = len(dec)
    if n == 0:
        return dict(n_days=int(len(res)), n_decided=0, win=None, ci_lo=None, ci_hi=None, exp_R=None)
    k = int(dec.sum())
    p = k / n
    lo, hi = wilson(k, n)
    return dict(n_days=int(len(res)), n_decided=n, win=round(p, 3), ci_lo=lo, ci_hi=hi,
                exp_R=round(p * (T / S) - (1 - p), 3))


def main():
    print("=" * 66)
    print(f"TOUCH-ONLY BACKTEST | stop {S:.1%} target {T:.1%} | horizon {HORIZON_DAYS} trading day(s)")
    print(f"break-even win rate at this stop/target: {BREAKEVEN:.1%}")
    print("=" * 66)
    daily, hourly, corn, soy = bt.fetch_data()
    df = bt.build_features(daily, corn, soy)
    up, dn = compute_outcomes_touch(daily, hourly)
    o_up, o_dn = bt.compute_outcomes(daily, hourly)          # original, for comparison only
    base = {"UP": summarize(up), "DOWN": summarize(dn)}
    orig = {"UP": round(float(o_up.mean()), 3), "DOWN": round(float(o_dn.mean()), 3)}
    und = {"UP": round(float(up.isna().mean()), 3), "DOWN": round(float(dn.isna().mean()), 3)}
    print(f"\nORIGINAL backtest.py baseline : UP {orig['UP']:.1%}  DOWN {orig['DOWN']:.1%}")
    print(f"TOUCH-ONLY baseline (decided) : UP {base['UP']['win']:.1%} (n={base['UP']['n_decided']})  "
          f"DOWN {base['DOWN']['win']:.1%} (n={base['DOWN']['n_decided']})")
    print(f"days where neither level was touched (original scored these by the close): "
          f"UP {und['UP']:.1%}  DOWN {und['DOWN']:.1%}")

    dates = df.index.sort_values()
    split = dates[int(len(dates) * bt.TRAIN_FRACTION)]

    # same-side baselines ("always UP" / "always DOWN") for train, holdout and each calendar quarter
    side_out = {"UP": up, "DOWN": dn}
    base_tr, base_ho = {}, {}
    for side, out in side_out.items():
        al = df[df.index.isin(out.index)]
        r = out.reindex(al.index)
        base_tr[side] = summarize(r[al.index < split])
        base_ho[side] = summarize(r[al.index >= split])
    print("\nSAME-SIDE BASELINES (no condition, decided days only):")
    for side in ("UP", "DOWN"):
        print(f"  always {side:<4}  train win {base_tr[side]['win']:.1%} (n={base_tr[side]['n_decided']})   "
              f"holdout win {base_ho[side]['win']:.1%} (n={base_ho[side]['n_decided']})   exp_R holdout {base_ho[side]['exp_R']}")

    # by calendar quarter: how did plain "always UP" / "always DOWN" behave when the market changed?
    quarters = []
    for side, out in side_out.items():
        o = out.copy()
        o.index = pd.DatetimeIndex(o.index)
        for q, grp in o.groupby(o.index.to_period("Q")):
            sm = summarize(grp)
            quarters.append(dict(quarter=str(q), side=side, **sm))
    qtab = pd.DataFrame(quarters)
    print("\nBY QUARTER (always-UP / always-DOWN, decided days):")
    if not qtab.empty:
        print(qtab[["quarter", "side", "n_decided", "win", "exp_R"]].to_string(index=False))

    rows = []
    for cond in CONDITIONS:
        if cond not in df.columns:
            continue
        for side, out in side_out.items():
            al = df[df.index.isin(out.index)]
            r = out.reindex(al.index)
            m = al[cond] == 1
            tr = summarize(r[m & (al.index < split)])
            ho = summarize(r[m & (al.index >= split)])
            if tr["n_days"] < 10:
                continue
            b_tr, b_ho = base_tr[side]["win"], base_ho[side]["win"]
            tr_lift = None if (tr["win"] is None or b_tr is None) else round(tr["win"] - b_tr, 3)
            ho_lift = None if (ho["win"] is None or b_ho is None) else round(ho["win"] - b_ho, 3)
            # a real edge must beat the SAME-SIDE baseline in train, and the holdout 95% CI-low must clear the holdout baseline
            beats = bool(tr_lift is not None and tr_lift > 0 and ho["ci_lo"] is not None and b_ho is not None
                         and ho["ci_lo"] > b_ho)
            rows.append(dict(condition=cond, side=side, volume_suspect=cond in VOLUME_SUSPECT,
                             **{f"train_{k}": v for k, v in tr.items()}, train_base=b_tr, train_lift=tr_lift,
                             **{f"hold_{k}": v for k, v in ho.items()}, hold_base=b_ho, hold_lift=ho_lift,
                             beats_same_side_baseline=beats))
    tab = pd.DataFrame(rows)
    tab.to_csv("backtest_touch_conditions.csv", index=False)
    qtab.to_csv("backtest_touch_quarters.csv", index=False)
    pd.set_option("display.width", 240)
    cols = ["condition", "side", "train_n_decided", "train_win", "train_lift", "hold_n_decided", "hold_win",
            "hold_base", "hold_lift", "hold_ci_lo", "hold_ci_hi", "beats_same_side_baseline"]
    if not tab.empty:
        print("\nCONDITIONS vs SAME-SIDE BASELINE (sorted by holdout lift; lift = condition win rate minus plain always-same-side):")
        print(tab.sort_values("hold_lift", ascending=False)[cols].head(20).to_string(index=False))
        n_ok = int(tab.beats_same_side_baseline.sum())
        print(f"\n{n_ok} of {len(tab)} condition/side tests beat their same-side baseline in train AND have a holdout "
              f"95% CI-low above the holdout baseline. With {len(tab)} tests, ~{0.025 * len(tab):.1f} would pass by chance alone.")
        print("Volume conditions (vol_*) are flagged volume_suspect: Yahoo ZW=F volume lags, so treat them as unreliable.")
    json.dump({"stop_pct": S, "target_pct": T, "horizon_days": HORIZON_DAYS, "breakeven": round(BREAKEVEN, 3),
               "split_date": str(split.date()), "original_baseline": orig, "touch_baseline": base,
               "undecided_share": und, "baseline_train": base_tr, "baseline_holdout": base_ho,
               "n_tests": int(len(tab)),
               "n_beating_same_side_baseline": int(tab.beats_same_side_baseline.sum()) if not tab.empty else 0},
              open("backtest_touch_results.json", "w"), indent=2, default=str)
    print("\nSaved backtest_touch_results.json, backtest_touch_conditions.csv and backtest_touch_quarters.csv "
          "(backtest_results.json / validated_conditions.json were NOT touched).")


if __name__ == "__main__":
    main()
