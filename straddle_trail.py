#!/usr/bin/env python3
"""
straddle_trail.py -- PRINT-ONLY. ZW=F intraday bars, one series: 1h (~2y) and 5m (last ~60 days, much tighter bounds).
Each CBOT trade day: at the first hourly open open BOTH a BUY and a SELL, each with a TRAILING stop of T% (trail starts at entry).
Day ends: any open leg is closed at the last bar's close. Cost = COST_PCT per leg (round trip).
Result per day = BUY leg % + SELL leg %  (theory: M - 2T - costs). One pair per day.
A bar cannot show which came first INSIDE it -> two honest bounds: LOWER (worst order for each leg) and UPPER (best order). Truth lies between.
Also prints average number of >=0.25% reversals ('swings') per day = how choppy wheat is.
ENV: COST_PCT (default 0.06)
"""
import os, numpy as np, pandas as pd
COST = float(os.getenv("COST_PCT", "0.06")); TRAILS = [0.25, 0.50, 1.00]

def load(interval="1h", period="730d"):
    import yfinance as yf
    return yf.Ticker("ZW=F").history(period=period, interval=interval, auto_adjust=False)

def days(h):
    h = h[["Open", "High", "Low", "Close"]].dropna().copy()
    idx = pd.DatetimeIndex(h.index); idx = idx.tz_convert("America/Chicago") if idx.tz is not None else idx
    h.index = idx; h["day"] = (idx + pd.Timedelta(hours=5)).date
    return [g for _, g in h.groupby("day") if len(g) >= 4]

def seq(bar, mode, side):
    o, hi, lo, cl = bar
    if mode == "LOWER": order = (hi, lo) if side > 0 else (lo, hi)      # worst order for that leg
    elif mode == "UPPER": order = (lo, hi) if side > 0 else (hi, lo)    # best order for that leg
    else: order = (lo, hi) if cl >= o else (hi, lo)                     # MID: bullish bar dips first, bearish bar spikes first
    return (o, order[0], order[1], cl)

def leg(side, bars, T, mode):
    """path-consistent trailing stop. returns (pct result, bar index closed or None)"""
    e = bars[0][0]; t = T / 100; peak = e
    for i, bar in enumerate(bars):
        for x in seq(bar, mode, side):
            stop = peak * (1 - t) if side > 0 else peak * (1 + t)
            if (side > 0 and x <= stop) or (side < 0 and x >= stop):
                px = stop if i > 0 or x != bar[0] else x
                return side * (px / e - 1) * 100, i
            peak = max(peak, x) if side > 0 else min(peak, x)
    return side * (bars[-1][3] / e - 1) * 100, None

def day_result(g, T, mode):
    bars = g[["Open", "High", "Low", "Close"]].values
    rb, _ = leg(1, bars, T, mode); rs, _ = leg(-1, bars, T, mode)
    return rb + rs - 2 * COST

def swings(g, thr=0.25):
    c = g.Close.values; n = 0; piv = c[0]; d = 0
    for x in c[1:]:
        ch = (x / piv - 1) * 100
        if d == 0:
            if abs(ch) >= thr: d, piv = (1 if ch > 0 else -1), x
        elif d == 1:
            if x > piv: piv = x
            elif (x / piv - 1) * 100 <= -thr: n += 1; d, piv = -1, x
        else:
            if x < piv: piv = x
            elif (x / piv - 1) * 100 >= thr: n += 1; d, piv = 1, x
    return n

def run(interval, period):
    D = days(load(interval, period)); h = len(D) // 2
    print(f"\n=== STRADDLE (BUY+SELL, trailing stop both legs) | {interval} bars | {len(D)} trade days | cost {COST}% per leg | % of price per day ===")
    print(f"avg >=0.25% reversals ('swings') per day: {np.mean([swings(g) for g in D]):.1f}")
    print(f"{'trail':>6} {'bound':<6} | {'mean%/day':>9} {'win days':>8} {'avg win':>8} {'avg loss':>9} | {'H1':>7} {'H2':>7}")
    for T in TRAILS:
        for mode in ("LOWER", "UPPER"):
            r = np.array([day_result(g, T, mode) for g in D]); w, l = r[r > 0], r[r <= 0]
            print(f"{T:>5.2f}% {mode:<6} | {r.mean():>+9.3f} {(r>0).mean():>8.0%} {w.mean() if len(w) else 0:>+8.2f} {l.mean() if len(l) else 0:>+9.2f} | {r[:h].mean():>+7.3f} {r[h:].mean():>+7.3f}")

def main():
    for iv, pe in (("1h", "730d"), ("5m", "60d")):
        try: run(iv, pe)
        except Exception as ex: print(f"{iv} failed: {ex}")
    print("\nREAD: LOWER > 0 in both halves = it works for sure. UPPER < 0 = it fails for sure. In between = bar-order uncertainty; trust the 5m table (tighter).")

if __name__ == "__main__":
    main()
