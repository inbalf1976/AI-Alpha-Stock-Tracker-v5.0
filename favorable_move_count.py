#!/usr/bin/env python3
"""
favorable_move_count.py -- PRINT-ONLY one-off (writes no files). Wheat ZW=F, one source (yfinance), no price mixing.
Q1 (daily bars, 10y): on how many trading days did price move +0.25% / +0.50% / ... in your favor from the OPEN?
    Long = High/Open-1, Short = 1-Low/Open. Also shows days where BOTH sides were hit (order unknowable on daily bars).
Q2 (hourly bars, ~2y): entry = first hourly open of the CBOT trade day. Which comes first, TARGET or STOP, same day?
    Same-bar both touched -> counted as STOP (conservative). Neither -> closed at last bar's close.
ENV: COST_PCT (default 0.06 = ~0.4 pt spread on ~690c)
"""
import os, sys, numpy as np, pandas as pd
TICKER = "ZW=F"; COST = float(os.getenv("COST_PCT", "0.06"))
TH = [0.25, 0.50, 0.75, 1.00]
GEOM = [(0.25, 0.25), (0.50, 0.50), (0.25, 0.50), (0.50, 0.25), (1.09, 0.72)]  # (target %, stop %); last = ~7.5pt/5pt at 690c

def load_daily():
    import yfinance as yf
    return yf.Ticker(TICKER).history(period="10y", interval="1d", auto_adjust=False)

def load_hourly():
    import yfinance as yf
    return yf.Ticker(TICKER).history(period="730d", interval="1h", auto_adjust=False)

def clean_daily(d):
    d = d[["Open", "High", "Low", "Close"]].dropna().copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d[(d.High >= d.Low) & (d.Open > 0) & (d.Close > 0)]
    d = d[(d.High / d.Low - 1) < 0.08]          # drop glitch days
    return d

def q1(d):
    up = (d.High / d.Open - 1) * 100; dn = (1 - d.Low / d.Open) * 100
    print(f"\n=== Q1 DAILY: {len(d)} days {d.index[0].date()} -> {d.index[-1].date()} | avg open-to-high {up.mean():.2f}% | avg open-to-low {dn.mean():.2f}% ===")
    print("% of days price reached the level from the OPEN (in your favor):")
    print(f"{'level':>7} | {'LONG days':>14} | {'SHORT days':>14} | {'EITHER':>7} | {'BOTH sides (order unknown)':>26}")
    for t in TH:
        a, b = up >= t, dn >= t
        print(f"{t:>6.2f}% | {a.sum():>5} ({a.mean():>5.1%}) | {b.sum():>5} ({b.mean():>5.1%}) | {(a|b).mean():>6.1%} | {(a&b).sum():>5} ({(a&b).mean():>5.1%})")
    print("\nBy year, LONG / SHORT % of days reaching:")
    yrs = sorted(set(d.index.year))
    print("year  days  " + "  ".join(f"{t:.2f}%(L/S)" for t in TH))
    for y in yrs:
        m = d.index.year == y
        print(f"{y}  {m.sum():>4}  " + "  ".join(f"{(up[m] >= t).mean():>4.0%}/{(dn[m] >= t).mean():<4.0%}" for t in TH))

def trade_days(h):
    h = h[["Open", "High", "Low", "Close"]].dropna().copy()
    idx = pd.DatetimeIndex(h.index)
    idx = idx.tz_convert("America/Chicago") if idx.tz is not None else idx
    h.index = idx
    h["day"] = (idx + pd.Timedelta(hours=5)).date      # 19:00 CT bar belongs to next CBOT trade date
    return h

def run_day(g, side, tgt, stp):
    e = float(g.Open.iloc[0])
    if (g.High.max() / e - 1) > 0.05 or (1 - g.Low.min() / e) > 0.05:
        return None                                    # glitch day
    T, S = tgt / 100, stp / 100
    for hi, lo in zip(g.High.values, g.Low.values):
        if side > 0: th, sh = hi >= e * (1 + T), lo <= e * (1 - S)
        else:        th, sh = lo <= e * (1 - T), hi >= e * (1 + S)
        if sh: return -stp, 0                           # stop first (also if both same bar)
        if th: return tgt, 1
    last = float(g.Close.iloc[-1]); r = side * (last / e - 1) * 100
    return float(np.clip(r, -stp, tgt)), 2

def q2(h):
    h = trade_days(h); days = [g for _, g in h.groupby("day") if len(g) >= 4]
    print(f"\n=== Q2 HOURLY: {len(days)} trade days {days[0].index[0].date()} -> {days[-1].index[-1].date()} | cost {COST}% per trade ===")
    print("Entry = first hourly open of the day. TARGET vs STOP, same day. 'neither' = marked to last close.")
    print(f"{'side':<5} {'target':>6} {'stop':>5} | {'n':>4} {'target1st':>9} {'stop1st':>8} {'neither':>8} | {'breakeven':>9} | {'avg% gross':>10} {'avg% net':>9} | {'net H1':>7} {'net H2':>7}")
    for (t, s) in GEOM:
        for side, nm in ((1, "LONG"), (-1, "SHORT")):
            res = [run_day(g, side, t, s) for g in days]; res = [r for r in res if r is not None]
            r = np.array([x[0] for x in res]); k = np.array([x[1] for x in res]); n = len(r)
            half = n // 2
            print(f"{nm:<5} {t:>5.2f}% {s:>4.2f}% | {n:>4} {(k==1).mean():>9.1%} {(k==0).mean():>8.1%} {(k==2).mean():>8.1%} | {s/(s+t):>9.1%} | {r.mean():>+10.3f} {r.mean()-COST:>+9.3f} | {r[:half].mean()-COST:>+7.3f} {r[half:].mean()-COST:>+7.3f}")
    print("\nREAD: a setup is only interesting if avg% net > 0 in BOTH halves (H1/H2) AND for the same side. One-sided or one-half gains = noise.")

def main():
    d = clean_daily(load_daily()); q1(d)
    try: q2(load_hourly())
    except Exception as ex: print("hourly failed:", ex)

if __name__ == "__main__":
    main()
