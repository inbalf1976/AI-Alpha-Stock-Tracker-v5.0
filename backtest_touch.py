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
    rows = []
    for cond in CONDITIONS:
        if cond not in df.columns:
            continue
        for side, out in (("UP", up), ("DOWN", dn)):
            al = df[df.index.isin(out.index)]
            r = out.reindex(al.index)
            m = al[cond] == 1
            tr = summarize(r[m & (al.index < split)])
            ho = summarize(r[m & (al.index >= split)])
            if tr["n_days"] < 10:
                continue
            ok = (ho["ci_lo"] is not None and ho["ci_lo"] > BREAKEVEN and tr["win"] is not None and tr["win"] > BREAKEVEN)
            rows.append(dict(condition=cond, side=side, volume_suspect=cond in VOLUME_SUSPECT,
                             **{f"train_{k}": v for k, v in tr.items()}, **{f"hold_{k}": v for k, v in ho.items()},
                             beats_breakeven_both=ok))
    tab = pd.DataFrame(rows)
    tab.to_csv("backtest_touch_conditions.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["condition", "side", "train_n_decided", "train_win", "hold_n_decided", "hold_win", "hold_ci_lo", "hold_ci_hi",
            "hold_exp_R", "beats_breakeven_both"]
    if not tab.empty:
        print("\nCONDITIONS (decided days only; holdout = most recent %.0f%%):" % (100 * (1 - bt.TRAIN_FRACTION)))
        print(tab.sort_values("hold_exp_R", ascending=False)[cols].head(20).to_string(index=False))
        n_ok = int(tab.beats_breakeven_both.sum())
        print(f"\n{n_ok} of {len(tab)} condition/side tests have train win AND holdout 95% CI-low above break-even "
              f"({BREAKEVEN:.1%}). With {len(tab)} tests, ~{0.025 * len(tab):.1f} would pass by chance alone.")
    json.dump({"stop_pct": S, "target_pct": T, "horizon_days": HORIZON_DAYS, "breakeven": round(BREAKEVEN, 3),
               "split_date": str(split.date()), "original_baseline": orig, "touch_baseline": base,
               "undecided_share": und, "n_tests": int(len(tab)),
               "n_beating_breakeven": int(tab.beats_breakeven_both.sum()) if not tab.empty else 0},
              open("backtest_touch_results.json", "w"), indent=2, default=str)
    print("\nSaved backtest_touch_results.json and backtest_touch_conditions.csv "
          "(backtest_results.json / validated_conditions.json were NOT touched).")


if __name__ == "__main__":
    main()
