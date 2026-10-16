#!/usr/bin/env python3
"""
stop_grid_tp.py -- PRINT-ONLY. ZW=F intraday bars, one series (1h ~2y, 5m ~60d).
Single-direction trade, entry at a bar's open, TAKE PROFIT fixed (TP, default 0.25%), TRAILING STOP varied over a grid.
For each stop: TP-hit rate and mean result after cost, compared with a SHUFFLED CONTROL (same bars, random order = pure noise, same engine),
worst single trade, mean result per 1% of risk, first/second half. LOWER/UPPER = worst/best order of high/low inside a bar.
Pooled = BUY and SELL entries together (direction-agnostic). ENV: COST_PCT (0.06), TP (0.25)
"""
import os, numpy as np, pandas as pd
COST = float(os.getenv("COST_PCT", "0.06")); TP = float(os.getenv("TP", "0.25")); STOPS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

def load(interval, period):
    import yfinance as yf
    return yf.Ticker("ZW=F").history(period=period, interval=interval, auto_adjust=False)

def prep(h):
    h = h[["Open", "High", "Low", "Close"]].dropna().copy()
    idx = pd.DatetimeIndex(h.index); idx = idx.tz_convert("America/Chicago") if idx.tz is not None else idx
    codes = pd.factorize(pd.Series((idx + pd.Timedelta(hours=5)).date))[0]
    ok = pd.Series(codes).map(pd.Series(codes).value_counts()).values >= 4
    return h.values[ok], codes[ok]

def leg(side, bars, tr, mode):
    e = bars[0][0]; tp = TP / 100; tr = tr / 100; peak = e
    for i, (o, hi, lo, cl) in enumerate(bars):
        adv, fav = (lo, hi) if side > 0 else (hi, lo)
        for x in ((o, adv, fav, cl) if mode == "LOWER" else (o, fav, adv, cl)):
            stop = peak * (1 - tr) if side > 0 else peak * (1 + tr); tpl = e * (1 + tp) if side > 0 else e * (1 - tp)
            if (side > 0 and x <= stop) or (side < 0 and x >= stop): return side * (stop / e - 1) * 100, i, "SL"
            if (side > 0 and x >= tpl) or (side < 0 and x <= tpl): return side * (tpl / e - 1) * 100, i, "TP"
            peak = max(peak, x) if side > 0 else min(peak, x)
    return side * (bars[-1][3] / e - 1) * 100, None, "TO"

def trades(B, D, side, tr, mode, hold):
    out = []; k = 0; n = len(B)
    while k < n - 3:
        end = k
        while end + 1 < n and D[end + 1] <= D[k] + hold - 1: end += 1
        r, i, c = leg(side, B[k:end + 1], tr, mode); out.append((r - COST, c))
        k = end + 1 if (hold == 1 or i is None) else k + i + 1
    return out

def shuffled(B, seed=11):
    """control series: same bar shapes (high/low/close vs open), random ORDER, chained open=prev close -> keeps volatility, destroys any real pattern"""
    rng = np.random.default_rng(seed); sh = B[:, 1:] / B[:, [0]]; sh = sh[rng.permutation(len(B))]
    out = np.empty_like(B); o = B[0, 0]
    for i in range(len(B)):
        out[i, 0] = o; out[i, 1:] = o * sh[i]; o = out[i, 3]
    return out

def row(label, hold, tr, lo, up, base):
    r = np.array([x[0] for x in lo]); ru = np.array([x[0] for x in up]); bs = np.array([x[0] for x in base]); h = len(r) // 2
    tp = np.mean([x[1] == "TP" for x in lo]); tpb = np.mean([x[1] == "TP" for x in base])
    print(f"{label:<6} {tr:>5.2f}% {hold:>3}d | {len(r):>5} {tp:>5.0%} {tpb:>6.0%} | {r.mean():>+7.3f} /{ru.mean():>+7.3f} {bs.mean():>+8.3f} {r.mean()-bs.mean():>+8.3f} | {r.min():>+7.2f} | {r.mean()/tr:>+8.3f} | {r[:h].mean():>+7.3f} {r[h:].mean():>+7.3f}")

def run(interval, period):
    B, D = prep(load(interval, period)); S = shuffled(B)
    print(f"\n=== {interval} bars | {len(set(D))} trade days | TP {TP}% | cost {COST}% per trade | % of price ===")
    print(f"{'trade':<6} {'stop':>6} {'hold':>4} | {'n':>5} {'TP%':>5} {'shuf':>6} | {'mean% LOWER/UPPER':>17} {'shuffled':>8} {'real-shuf':>9} | {'worst':>7} | {'mean/risk':>9} | {'H1':>7} {'H2':>7}")
    for hold in (1, 5):
        for tr in STOPS:
            res = {}
            for side, nm in ((1, "BUY"), (-1, "SELL")):
                res[nm] = (trades(B, D, side, tr, "LOWER", hold), trades(B, D, side, tr, "UPPER", hold), trades(S, D, side, tr, "LOWER", hold))
                if hold == 1: row(nm, hold, tr, *res[nm])
            row("POOLED", hold, tr, res["BUY"][0] + res["SELL"][0], res["BUY"][1] + res["SELL"][1], res["BUY"][2] + res["SELL"][2])

def main():
    for iv, pe in (("1h", "730d"), ("5m", "60d")):
        try: run(iv, pe)
        except Exception as ex: print(f"{iv} failed: {ex}")
    print("\nREAD: 'real-shuf' = real result minus the shuffled-noise control (same bars, random order). A stop is usable only if mean% > 0 for LOWER and UPPER,")
    print("in BOTH halves, with real-shuf positive for neighbouring stops too and for BOTH BUY and SELL rows. At hold 1d the result can wander +/-0.06 by luck alone")
    print("(tested on pure noise) - trust the 5d rows (many more trades) and the 5m table before the 1h 1d rows. 'mean/risk' = result per 1% of price risked.")

if __name__ == "__main__":
    main()
