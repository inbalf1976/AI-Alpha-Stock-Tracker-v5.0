#!/usr/bin/env python3
"""
backtest_touch.py -- touch-only re-score of backtest.py (READ-ONLY w.r.t. the live system).

v3 additions (vs v2):
  * NON-OVERLAPPING samples: with a H-day window, entries every day overlap, so n is inflated ~H-fold.
    Headline "always UP / always DOWN" numbers are now computed on H disjoint entry sets (one per offset)
    and reported as mean and range; condition CIs use n/H as the effective sample size.
  * REGIME SPLIT: always-UP / always-DOWN results when the entry-day close was above vs below its SMA50.
  * GLITCH/ROLL CLEANING: reuses hunter_pattern_research.mark_bad() (the repo's own cleaner). Entries whose
    window touches a flagged hourly bar are dropped. Falls back to no cleaning (with a warning) if the
    module cannot be imported.
  * COSTS: COST_PCT (default 0.1% of price round-trip, an ASSUMPTION, not your broker's real spread) is
    charged on every trade. Overnight financing is NOT modelled.

Scoring (unchanged): win = target touched first, loss = stop touched first or both inside one hourly bar,
neither touched = EXCLUDED. UP and DOWN scored independently.

SAFETY: never calls backtest.main(); does NOT write backtest_results.json or validated_conditions.json.
Outputs: backtest_touch_results.json, backtest_touch_conditions.csv, backtest_touch_quarters.csv,
         backtest_touch_regimes.csv

Env: HORIZON_DAYS (default 1; use 5 for a weekly-style window), COST_PCT (default 0.001)
"""
import os, json, math
import numpy as np
import pandas as pd
import backtest as bt

H = int(os.getenv("HORIZON_DAYS", "1"))
HORIZON_DAYS = H
COST = float(os.getenv("COST_PCT", "0.001"))
S, T = bt.STOP_PCT, bt.TARGET_PCT
BREAKEVEN = S / (S + T)
CONDITIONS = ['above_sma20', 'above_sma50', 'trend_aligned', 'rsi_oversold', 'rsi_overbought', 'rsi_neutral',
              'macd_bullish', 'vol_good', 'vol_high', 'vol_low', 'near_bb_lower', 'near_bb_upper', 'inside_bb',
              'wc_bullish', 'wc_bearish', 'in_lower_half', 'in_upper_half', 'momentum_up', 'momentum_down',
              'bullish_month', 'bearish_month']
VOLUME_SUSPECT = {'vol_low', 'vol_good', 'vol_high'}


def exp_r(p, cost=0.0):
    """expectancy in R (stop units); a win pays (T-cost)/S, a loss costs (S+cost)/S."""
    return p * (T - cost) / S - (1 - p) * (S + cost) / S


def first_touch(hi, lo, entry, side):
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


def bad_hourly_dates(hourly):
    """dates (normalized, naive) containing hourly bars flagged by the repo's glitch cleaner."""
    try:
        import hunter_pattern_research as hp
        h = hourly[["Open", "High", "Low", "Close"]].copy()
        h.index = h.index.tz_localize("UTC") if h.index.tz is None else h.index
        h2, wins = hp.mark_bad(h)
        bad = h2["Bad"].values.astype(bool)
        print(f"Glitch cleaning: {len(wins)} window(s), {int(bad.sum())} of {len(h)} hourly bars flagged")
        return set(hourly.index.normalize()[bad])
    except Exception as e:
        print(f"WARNING: glitch cleaning skipped ({e}) -- results use raw bars")
        return set()


def compute_outcomes_touch(daily, hourly, bad_dates=frozenset(), entry_mode="close"):
    """entry_mode 'close' = daily Close (as before); 'next_open' = Open of the first hourly bar of the next
    session (CONTROL: removes any close-to-open gap / data-alignment effect)."""
    dd = daily.index.normalize().unique()
    hd = hourly.index.normalize().values
    hi, lo = hourly["High"].values, hourly["Low"].values
    op = hourly["Open"].values
    up, dn, dropped = {}, {}, 0
    for i in range(len(dd) - 1 - H):
        d = dd[i]
        if bad_dates and any(x in bad_dates for x in dd[i:i + H + 1]):
            dropped += 1
            continue
        entry = float(daily.loc[daily.index.normalize() == d, "Close"].iloc[-1])
        a = np.searchsorted(hd, np.datetime64(dd[i + 1]), side="left")
        b = np.searchsorted(hd, np.datetime64(dd[i + H]), side="right")
        if b <= a:
            continue
        if entry_mode == "next_open":
            entry = float(op[a])
        up[d] = first_touch(hi[a:b], lo[a:b], entry, "UP")
        dn[d] = first_touch(hi[a:b], lo[a:b], entry, "DOWN")
    return pd.Series(up, dtype=float), pd.Series(dn, dtype=float), dropped


def wilson(k, n, z=1.96):
    if n <= 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def summarize(res):
    """stats over decided days; CI uses effective n = n/H because H-day windows overlap."""
    dec = res.dropna()
    n = len(dec)
    if n == 0:
        return dict(n_days=int(len(res)), n_decided=0, n_eff=0, win=None, ci_lo=None, ci_hi=None, exp_R=None, exp_R_cost=None)
    p = float(dec.mean())
    n_eff = max(1, round(n / H))
    lo, hi = wilson(p * n_eff, n_eff)
    return dict(n_days=int(len(res)), n_decided=n, n_eff=n_eff, win=round(p, 3), ci_lo=lo, ci_hi=hi,
                exp_R=round(exp_r(p), 3), exp_R_cost=round(exp_r(p, COST), 3))


def disjoint(res, mask=None):
    """H disjoint entry sets (every H-th entry, one per offset). Returns summary across offsets."""
    outs = []
    for off in range(H):
        sub = res.iloc[off::H]
        if mask is not None:
            sub = sub[mask.reindex(sub.index).fillna(False).values]
        dec = sub.dropna()
        if len(dec):
            outs.append((len(dec), float(dec.mean())))
    if not outs:
        return dict(n_per_set=0, win=None, win_min=None, win_max=None, exp_R=None, exp_R_cost=None)
    ws = [w for _, w in outs]
    w = float(np.mean(ws))
    return dict(n_per_set=int(round(np.mean([n for n, _ in outs]))), win=round(w, 3), win_min=round(min(ws), 3),
                win_max=round(max(ws), 3), exp_R=round(exp_r(w), 3), exp_R_cost=round(exp_r(w, COST), 3))


def main():
    print("=" * 70)
    print(f"TOUCH-ONLY BACKTEST v3 | stop {S:.1%} target {T:.1%} | horizon {H} trading day(s)")
    print(f"break-even win rate: {BREAKEVEN:.1%} | assumed round-trip cost: {COST:.2%} of price (financing not modelled)")
    print("=" * 70)
    daily, hourly, corn, soy = bt.fetch_data()
    df = bt.build_features(daily, corn, soy)
    bad = bad_hourly_dates(hourly)
    up, dn, dropped = compute_outcomes_touch(daily, hourly, bad)
    print(f"entries dropped for glitch/roll windows: {dropped}")
    # CONTROL: same test, but enter at the next session's first hourly OPEN instead of the daily Close
    c_up, c_dn, _ = compute_outcomes_touch(daily, hourly, bad, entry_mode="next_open")
    ctrl = {"UP": disjoint(c_up), "DOWN": disjoint(c_dn)}
    o_up, o_dn = bt.compute_outcomes(daily, hourly)
    orig = {"UP": round(float(o_up.mean()), 3), "DOWN": round(float(o_dn.mean()), 3)}
    side_out = {"UP": up, "DOWN": dn}
    und = {k: round(float(v.isna().mean()), 3) for k, v in side_out.items()}
    print(f"\nORIGINAL backtest.py baseline : UP {orig['UP']:.1%}  DOWN {orig['DOWN']:.1%}")
    print(f"undecided share (neither level touched): UP {und['UP']:.1%}  DOWN {und['DOWN']:.1%}")

    # ---- headline: non-overlapping always-UP / always-DOWN
    print(f"\nALWAYS-SAME-SIDE, NON-OVERLAPPING ({H} disjoint set(s); win = mean, range across sets):")
    head = {}
    for side, out in side_out.items():
        d = disjoint(out)
        head[side] = d
        if d["win"] is None:
            continue
        print(f"  always {side:<4} n/set={d['n_per_set']:<4} win {d['win']:.1%} ({d['win_min']:.1%}-{d['win_max']:.1%})  "
              f"exp {d['exp_R']:+.2f}R  after {COST:.2%} cost {d['exp_R_cost']:+.2f}R")

    print("\nCONTROL (enter at next session's first hourly open; non-overlapping):")
    for side in ("UP", "DOWN"):
        d = ctrl[side]
        if d["win"] is not None:
            print(f"  always {side:<4} n/set={d['n_per_set']:<4} win {d['win']:.1%} ({d['win_min']:.1%}-{d['win_max']:.1%})  "
                  f"exp {d['exp_R']:+.2f}R  after cost {d['exp_R_cost']:+.2f}R")
    if head["UP"]["win"] is not None and head["DOWN"]["win"] is not None:
        gap = head["UP"]["win"] - head["DOWN"]["win"]
        cgap = ctrl["UP"]["win"] - ctrl["DOWN"]["win"] if ctrl["UP"]["win"] is not None and ctrl["DOWN"]["win"] is not None else None
        print(f"  asymmetry check: UP minus DOWN win rate = {gap:+.1%} (entry at daily close)"
              + (f" vs {cgap:+.1%} (entry at next open)." if cgap is not None else "."))
        if abs(gap) > 0.15:
            print("  A gap this large is more than wheat's daily drift should produce. If the control gap is much smaller, "
                  "the close-entry result was partly a gap/alignment artifact; if similar, the asymmetry is in the price path itself.")

    # ---- regime split (entry-day close vs SMA50)
    reg_rows = []
    if "above_sma50" in df.columns:
        for side, out in side_out.items():
            m_all = df["above_sma50"].reindex(out.index) == 1
            for lab, mask in (("above SMA50", m_all), ("below SMA50", ~m_all)):
                d = disjoint(out, mask)
                reg_rows.append(dict(side=side, regime=lab, **d))
    rtab = pd.DataFrame(reg_rows)
    rtab.to_csv("backtest_touch_regimes.csv", index=False)
    print("\nREGIME SPLIT (non-overlapping):")
    if not rtab.empty:
        print(rtab[["side", "regime", "n_per_set", "win", "win_min", "win_max", "exp_R", "exp_R_cost"]].to_string(index=False))

    # ---- baselines, quarters (overlapping, decided)
    dates = df.index.sort_values()
    split = dates[int(len(dates) * bt.TRAIN_FRACTION)]
    base_tr, base_ho = {}, {}
    for side, out in side_out.items():
        al = df[df.index.isin(out.index)]
        r = out.reindex(al.index)
        base_tr[side], base_ho[side] = summarize(r[al.index < split]), summarize(r[al.index >= split])
    quarters = []
    for side, out in side_out.items():
        o = out.copy()
        o.index = pd.DatetimeIndex(o.index)
        for q, grp in o.groupby(o.index.to_period("Q")):
            quarters.append(dict(quarter=str(q), side=side, **summarize(grp)))
    qtab = pd.DataFrame(quarters)
    qtab.to_csv("backtest_touch_quarters.csv", index=False)
    print("\nBY QUARTER (overlapping entries; n_eff = n/H):")
    if not qtab.empty:
        print(qtab[["quarter", "side", "n_decided", "n_eff", "win", "exp_R", "exp_R_cost"]].to_string(index=False))

    # ---- conditions vs same-side baseline
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
            beats = bool(tr_lift is not None and tr_lift > 0 and ho["ci_lo"] is not None and b_ho is not None
                         and ho["ci_lo"] > b_ho)
            rows.append(dict(condition=cond, side=side, volume_suspect=cond in VOLUME_SUSPECT,
                             **{f"train_{k}": v for k, v in tr.items()}, train_base=b_tr, train_lift=tr_lift,
                             **{f"hold_{k}": v for k, v in ho.items()}, hold_base=b_ho, hold_lift=ho_lift,
                             beats_same_side_baseline=beats))
    tab = pd.DataFrame(rows)
    tab.to_csv("backtest_touch_conditions.csv", index=False)
    pd.set_option("display.width", 240)
    cols = ["condition", "side", "train_n_eff", "train_win", "train_lift", "hold_n_eff", "hold_win", "hold_base",
            "hold_lift", "hold_ci_lo", "hold_ci_hi", "beats_same_side_baseline"]
    if not tab.empty:
        print("\nCONDITIONS vs SAME-SIDE BASELINE (CIs use effective n = n/H; sorted by holdout lift):")
        print(tab.sort_values("hold_lift", ascending=False)[cols].head(15).to_string(index=False))
        n_ok = int(tab.beats_same_side_baseline.sum())
        print(f"\n{n_ok} of {len(tab)} tests beat their same-side baseline in train AND have a holdout 95% CI-low above "
              f"the holdout baseline. ~{0.025 * len(tab):.1f} would pass by chance alone. vol_* rows: volume data is unreliable.")
    json.dump({"stop_pct": S, "target_pct": T, "horizon_days": H, "cost_pct": COST, "breakeven": round(BREAKEVEN, 3),
               "split_date": str(split.date()), "original_baseline": orig, "undecided_share": und,
               "entries_dropped_glitch": dropped, "always_side_nonoverlapping": head, "control_next_open": ctrl,
               "baseline_train": base_tr, "baseline_holdout": base_ho, "n_tests": int(len(tab)),
               "n_beating_same_side_baseline": int(tab.beats_same_side_baseline.sum()) if not tab.empty else 0},
              open("backtest_touch_results.json", "w"), indent=2, default=str)
    print("\nSaved backtest_touch_results.json + _conditions.csv + _quarters.csv + _regimes.csv "
          "(backtest_results.json / validated_conditions.json NOT touched).")


if __name__ == "__main__":
    main()
