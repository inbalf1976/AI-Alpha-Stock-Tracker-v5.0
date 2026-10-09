#!/usr/bin/env python3
"""
seasonality_test.py  --  READ-ONLY research (created 2026-10-08)

Question: do the seasonal rules in the pasted text hold in about 25 years of daily wheat prices?
  Rule A  "harvest low": long from late Q2 into early Q3  (tested: Jun 20 -> Jul 31 and Jun 20 -> Aug 31)
  Rule B  "Q4 rally":    long through Q4                   (tested: Oct 1 -> Dec 31)
plus every calendar month on its own, with a multiple-testing guard.

BIG CAVEAT, handled explicitly: Yahoo's ZW=F is a stitched front-month series. Each roll adds a
one-day jump (the spread between two contracts) that happens in the same calendar windows every
year - exactly what could fake a "seasonal". Two versions are reported:
  RAW        = as downloaded
  ROLL-TRIMMED = in every roll window (first 20 days of Feb/Apr/Jun/Aug/Nov and Aug 20-31) the single
                 biggest absolute daily return of each year is removed. That also removes real shocks, so it is
                 conservative: an effect that survives it is more credible; one that vanishes is unproven,
                 not disproven.

For each rule: per-year log returns, mean / median, years positive, exact sign-test p, t-stat of the
yearly returns, and the percentile of the rule's mean among ALL same-length windows starting on
any day of the year (is this window special, or is every window like it?).
Monthly table: p-value from 2,000 circular shifts of the return series (max |monthly mean| across
all 12 months), so picking the best month is not rewarded.

Never writes repo state, never sends Telegram, never commits.
"""
import json
import math

import numpy as np
import pandas as pd

TICKER = "ZW=F"
ROLL_MONTHS = (2, 4, 6, 8, 11)
RULES = {
    "A1 long Jun20->Jul31": ((6, 20), (7, 31)),
    "A2 long Jun20->Aug31": ((6, 20), (8, 31)),
    "B  long Oct1->Dec31": ((10, 1), (12, 31)),
}
RNG = np.random.default_rng(5)


def sign_p(k, n):
    if n == 0:
        return None
    pk = [math.comb(n, i) for i in range(n + 1)]
    return min(1.0, 2 * min(sum(pk[:k + 1]), sum(pk[k:])) / 2.0 ** n)


def load():
    import yfinance as yf
    df = yf.download(TICKER, period="max", interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    c = df["Close"].dropna()
    c = c[c > 0]
    c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
    return c


def roll_trim(ret):
    """drop the biggest |return| day in each roll window of each year."""
    r = ret.copy()
    for y in sorted(set(r.index.year)):
        wins = [(pd.Timestamp(y, m, 1), pd.Timestamp(y, m, 20)) for m in ROLL_MONTHS]
        wins.append((pd.Timestamp(y, 8, 20), pd.Timestamp(y, 8, 31)))
        for a, b in wins:
            seg = r[(r.index >= a) & (r.index <= b)].dropna()
            if len(seg):
                r.loc[seg.abs().idxmax()] = np.nan
    return r


def window_returns(cs, start_md, end_md, years):
    out = {}
    for y in years:
        a, b = pd.Timestamp(y, *start_md), pd.Timestamp(y, *end_md)
        if a < cs.index[0] or b > cs.index[-1]:
            continue
        out[y] = float(cs.asof(b) - cs.asof(a))
    return pd.Series(out)


def rule_stats(ret, cs, years, name, start_md, end_md):
    w = window_returns(cs, start_md, end_md, years) * 100
    n = len(w)
    k = int((w > 0).sum())
    t = float(w.mean() / (w.std(ddof=1) / math.sqrt(n))) if n > 2 and w.std() > 0 else None
    length = (pd.Timestamp(2001, *end_md) - pd.Timestamp(2001, *start_md)).days
    # every possible start day of the year, same length
    means = []
    for off in range(0, 365, 3):
        s = pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=off)
        vals = []
        for y in years:
            a = pd.Timestamp(y, s.month, s.day)
            b = a + pd.Timedelta(days=length)
            if a >= cs.index[0] and b <= cs.index[-1]:
                vals.append(float(cs.asof(b) - cs.asof(a)) * 100)
        if vals:
            means.append(np.mean(vals))
    pct = float((np.array(means) < w.mean()).mean() * 100) if means else None
    return {"years": n, "mean_pct": round(float(w.mean()), 2), "median_pct": round(float(w.median()), 2),
            "years_positive": f"{k}/{n}", "sign_test_p": None if sign_p(k, n) is None else round(sign_p(k, n), 4),
            "t_stat": None if t is None else round(t, 2),
            "percentile_vs_all_same_length_windows": None if pct is None else round(pct, 1)}


def monthly(ret, reps=2000):
    r = ret.dropna()
    ym = r.groupby([r.index.year, r.index.month]).sum() * 100
    tab = ym.groupby(level=1).agg(["mean", "median", "count"])
    pos = ym.groupby(level=1).apply(lambda s: float((s > 0).mean()))
    obs = tab["mean"].abs().max()
    vals = r.values
    months = r.index.month.values
    years = r.index.year.values
    cnt = 0
    for _ in range(reps):
        sh = np.roll(vals, int(RNG.integers(250, len(vals) - 250)))
        ym2 = pd.Series(sh).groupby([years, months]).sum() * 100
        if ym2.groupby(level=1).mean().abs().max() >= obs:
            cnt += 1
    rows = {int(m): {"mean_pct": round(float(tab.loc[m, "mean"]), 2), "median_pct": round(float(tab.loc[m, "median"]), 2),
                     "share_of_years_up": round(float(pos.loc[m]), 2), "years": int(tab.loc[m, "count"])} for m in tab.index}
    return {"by_month": rows, "p_best_month_by_chance": round((1 + cnt) / (1 + reps), 4)}


def analyse(close):
    ret = np.log(close).diff().dropna()
    years = sorted(set(ret.index.year))[1:-1]          # full years only
    out = {"data": {"from": str(close.index[0].date()), "to": str(close.index[-1].date()), "full_years": len(years),
                    "avg_daily_drift_pct": round(float(ret.mean() * 100), 4)}}
    for label, r in (("RAW", ret), ("ROLL_TRIMMED", roll_trim(ret))):
        cs = r.fillna(0).cumsum()
        block = {"rules": {nm: rule_stats(r, cs, years, nm, s, e) for nm, (s, e) in RULES.items()}, "months": monthly(r)}
        out[label] = block
    return out


def verdict(rep):
    res = {}
    for nm in RULES:
        r, t = rep["RAW"]["rules"][nm], rep["ROLL_TRIMMED"]["rules"][nm]
        ok = lambda x: x["sign_test_p"] is not None and x["sign_test_p"] < 0.0167 and x["mean_pct"] > 0   # 0.05 / 3 rules
        res[nm] = ("HOLDS in both versions" if ok(r) and ok(t) else
                   "only in RAW (could be roll artifact)" if ok(r) else "not supported")
    return res


def main():
    close = load()
    print(f"daily bars: {len(close)}  {close.index[0].date()} -> {close.index[-1].date()}")
    rep = analyse(close)
    rep["verdicts"] = verdict(rep)
    rep["rule_for_support"] = "yearly sign-test p < 0.0167 (0.05 split over 3 rules) with positive mean, in BOTH raw and roll-trimmed versions"
    rep["power_note"] = "with ~25 yearly observations only effects of roughly 5%+ over the window can show up; a smaller real seasonal tendency cannot be confirmed OR ruled out"
    txt = json.dumps(rep, indent=2, default=str)
    print(txt)
    open("seasonality_report.json", "w").write(txt)
    print("\nWrote seasonality_report.json (nothing else was changed).")


if __name__ == "__main__":
    main()
