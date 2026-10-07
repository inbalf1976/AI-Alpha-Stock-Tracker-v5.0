#!/usr/bin/env python3
"""
hunter_pattern_research.py  --  READ-ONLY research (created 2026-10-06)

Turns the "stop-loss hunter" hypotheses into tests on real ZW=F data.
It never writes to the repo state, never sends Telegram, never commits.
Output: console report + hunter_research_report.json.

Hypotheses tested (letters match the user's description):
  A  rapid moves cluster in a time window (default 09:30-14:30 Israel, plus Fri 18:30)
  B  rapid moves are large (>= MOVE_PCT within MOVE_HOURS), either direction
  C  the extreme of a rapid move lands next to a "logical stop cluster" level
     (prior day/week high/low, round 50) more often than random windows do
  G/H at a first touch of such a level: does price PASS (close beyond by PASS_PCT)
     or REJECT (close back by PASS_PCT) first - and does it depend on time of
     day, weekday, MFI, level type?  After a PASS, how far does it extend (the
     "next cluster 0.5-1% away" claim)?

What it can NOT show: who is trading, or where the orders actually sit.
It tests whether the price behaviour the hypothesis predicts is really
there, with sample sizes and 95% intervals so a small sample is not mistaken
for an edge.
"""
import json
import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd

TICKER = "ZW=F"
MOVE_PCT = float(os.getenv("MOVE_PCT", "1.5"))          # rapid-move threshold, percent
MOVE_HOURS = int(os.getenv("MOVE_HOURS", "6"))          # "a few hours"
EVENT_GAP_BARS = 12                                     # no double counting
NEAR_LEVEL_PCT = float(os.getenv("NEAR_LEVEL_PCT", "0.3"))
PASS_PCT = float(os.getenv("PASS_PCT", "0.5"))
TOUCH_HORIZON_BARS = int(os.getenv("TOUCH_HORIZON_BARS", "8"))
EXT_HORIZON_BARS = 24
IL_WINDOW = (9, 15)        # bar start hours 09:00-14:59 Israel  (~09:30-14:30)
TZ_IL, TZ_CT = "Asia/Jerusalem", "America/Chicago"


# ---------- helpers ----------
def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - a) / d, 3), round((c + a) / d, 3))


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def add_mfi(df, n=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    up = mf.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    dn = mf.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    df["MFI"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return df


def trade_date(idx):
    """CME ags trade date: bars from 19:00 CT belong to the next date."""
    return (idx.tz_convert(TZ_CT) + pd.Timedelta(hours=5)).date


def build_levels(daily):
    """per trade date -> {name: price}; prior-day and prior-week highs/lows."""
    d = daily.copy()
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    iso = d.index.isocalendar()
    d["wk"] = list(zip(iso.year, iso.week))
    g = d.groupby("wk").agg(WH=("High", "max"), WL=("Low", "min"))
    wk = {k: (float(r.WH), float(r.WL)) for k, r in zip(g.index, g.itertuples())}
    levels = {}
    dates = list(d.index)
    for i, dt in enumerate(dates):
        lv = {}
        if i > 0:
            lv["PDH"], lv["PDL"] = float(d["High"].iloc[i - 1]), float(d["Low"].iloc[i - 1])
        pw = (dt - timedelta(days=7)).isocalendar()
        key = (pw.year, pw.week)
        if key in wk:
            lv["PWH"], lv["PWL"] = wk[key]
        levels[dt.date()] = lv
    return levels


def round_levels(price, step=50.0):
    base = math.floor(price / step) * step
    return {"R50_dn": base, "R50_up": base + step}


# ---------- test 1: rapid moves (A, B, C) ----------
def find_events(df, levels):
    """A rapid move = >= MOVE_PCT from a swing point to the extreme within MOVE_HOURS.
    The start is refined to the true turning bar (lowest low before an UP extreme,
    highest high before a DOWN extreme), so time-of-day stats are not shifted early."""
    o, h, l = df["Open"].values, df["High"].values, df["Low"].values
    n, ev, i = len(df), [], 0
    while i < n - MOVE_HOURS:
        w = slice(i + 1, i + 1 + MOVE_HOURS)
        up = (h[w].max() - o[i]) / o[i] * 100
        dn = (o[i] - l[w].min()) / o[i] * 100
        if max(up, dn) >= MOVE_PCT:
            direction = "UP" if up >= dn else "DOWN"
            if direction == "UP":
                e = i + 1 + int(np.argmax(h[w])); seg = l[i:e + 1]; s0 = i + len(seg) - 1 - int(np.argmin(seg[::-1]))
                ext, move = float(h[e]), (h[e] - l[s0]) / l[s0] * 100
            else:
                e = i + 1 + int(np.argmin(l[w])); seg = h[i:e + 1]; s0 = i + len(seg) - 1 - int(np.argmax(seg[::-1]))
                ext, move = float(l[e]), (h[s0] - l[e]) / h[s0] * 100
            ev.append({"i": s0, "ts": df.index[s0], "dir": direction, "move_pct": round(float(move), 2), "extreme": ext, "ext_i": e})
            i = e + EVENT_GAP_BARS
        else:
            i += 1
    return ev


def near_any_level(price, td, levels):
    lv = dict(levels.get(td, {}))
    lv.update(round_levels(price))
    for name, L in lv.items():
        if abs(price - L) / L * 100 <= NEAR_LEVEL_PCT:
            return name
    return None


def test_events(df, levels):
    ev = find_events(df, levels)
    il = df.index.tz_convert(TZ_IL)
    out = {"threshold_pct": MOVE_PCT, "window_hours": MOVE_HOURS, "n_events": len(ev)}
    if not ev:
        return out
    starts = [e["ts"].tz_convert(TZ_IL) for e in ev]
    in_win = [(IL_WINDOW[0] <= s.hour < IL_WINDOW[1]) or (s.weekday() == 4 and s.hour == 18) for s in starts]
    all_in_win = np.mean([(IL_WINDOW[0] <= t.hour < IL_WINDOW[1]) or (t.weekday() == 4 and t.hour == 18) for t in il])
    out["direction"] = pd.Series([e["dir"] for e in ev]).value_counts().to_dict()
    out["start_hour_IL"] = pd.Series([s.hour for s in starts]).value_counts().sort_index().to_dict()
    out["start_weekday"] = pd.Series([s.strftime("%a") for s in starts]).value_counts().to_dict()
    out["A_share_in_window"] = rate(sum(in_win), len(ev))
    out["A_share_of_all_bars_in_window"] = round(float(all_in_win), 3)
    out["B_median_move_pct"] = round(float(np.median([e["move_pct"] for e in ev])), 2)
    # C: extreme near a level vs random windows
    tdates = pd.Series(trade_date(df.index), index=df.index)
    hit = sum(near_any_level(e["extreme"], tdates.loc[e["ts"]], levels) is not None for e in ev)
    rng = np.random.default_rng(7)
    used = {e["i"] for e in ev}
    hh, ll, oo = df["High"].values, df["Low"].values, df["Open"].values
    n = len(df) - MOVE_HOURS
    fwd_up = np.array([(hh[i + 1:i + 1 + MOVE_HOURS].max() - oo[i]) / oo[i] * 100 for i in range(n)])
    fwd_dn = np.array([(oo[i] - ll[i + 1:i + 1 + MOVE_HOURS].min()) / oo[i] * 100 for i in range(n)])
    big = np.maximum(fwd_up, fwd_dn)

    def baseline(mask, label):
        pool = [i for i in np.where(mask)[0] if i not in used]
        if not pool:
            return None
        samp = rng.choice(pool, size=min(len(pool), 2000), replace=False)
        r = 0
        for i in samp:
            w = slice(i + 1, i + 1 + MOVE_HOURS)
            px = hh[w].max() if fwd_up[i] >= fwd_dn[i] else ll[w].min()
            r += near_any_level(float(px), tdates.iloc[i], levels) is not None
        return rate(r, len(samp))

    out["C_extreme_near_level"] = rate(hit, len(ev))
    out["C_baseline_any_window"] = baseline(big >= 0, "any")
    out["C_baseline_almost_moves"] = baseline((big >= MOVE_PCT * 0.5) & (big < MOVE_PCT), "almost")
    return out


# ---------- test 2: level touches (G, H) ----------
def classify_touch(df, t, L, side, horizon=TOUCH_HORIZON_BARS):
    """side='R' resistance (approached from below) or 'S' support (from above).
    Close-based: PASS = a close beyond L by PASS_PCT, REJECT = a close back
    PASS_PCT on the origin side, whichever comes first. Returns (outcome, bars)."""
    c = df["Close"].values
    p = PASS_PCT / 100
    for j in range(t, min(t + horizon + 1, len(df))):
        if side == "R":
            if c[j] >= L * (1 + p):
                return "PASS", j
            if c[j] <= L * (1 - p):
                return "REJECT", j
        else:
            if c[j] <= L * (1 - p):
                return "PASS", j
            if c[j] >= L * (1 + p):
                return "REJECT", j
    return "NONE", None


def find_touches(df, levels):
    tdates = trade_date(df.index)
    o, h, l = df["Open"].values, df["High"].values, df["Low"].values
    seen, touches = set(), []
    for t in range(1, len(df)):
        td = tdates[t]
        lv = dict(levels.get(td, {}))
        lv.update(round_levels(o[t]))
        for name, L in lv.items():
            if (td, name) in seen:
                continue
            side = None
            if o[t] < L <= h[t]:
                side = "R"
            elif o[t] > L >= l[t]:
                side = "S"
            if side is None:
                continue
            seen.add((td, name))
            outcome, j = classify_touch(df, t, L, side)
            ext = None
            if outcome == "PASS":
                w = slice(j, min(j + EXT_HORIZON_BARS + 1, len(df)))
                ext = ((h[w].max() - L) if side == "R" else (L - l[w].min())) / L * 100
            sweep = False   # wicked through, closed back inside within 3 bars
            for k in range(t, min(t + 4, len(df))):
                if (side == "R" and df["Close"].values[k] < L and h[t:k + 1].max() > L) or \
                   (side == "S" and df["Close"].values[k] > L and l[t:k + 1].min() < L):
                    sweep = True
                    break
            il = df.index[t].tz_convert(TZ_IL)
            touches.append({
                "level": name, "side": side, "outcome": outcome, "sweep": sweep,
                "hour_IL": il.hour, "weekday": il.strftime("%a"),
                "in_window": IL_WINDOW[0] <= il.hour < IL_WINDOW[1],
                "mfi": None if pd.isna(df["MFI"].iloc[t]) else float(df["MFI"].iloc[t]),
                "ext_pct": None if ext is None else float(ext),
            })
    return touches


def summarize_touches(tc):
    if not tc:
        return {"n_touches": 0}
    d = pd.DataFrame(tc)
    dec = d[d.outcome != "NONE"]

    def split(mask, label):
        s = dec[mask]
        return {label: rate((s.outcome == "PASS").sum(), len(s))}

    out = {"n_touches": len(d), "decided": len(dec), "none": int((d.outcome == "NONE").sum()),
           "overall_pass_rate": rate((dec.outcome == "PASS").sum(), len(dec))}
    out["by_level"] = {k: rate((g.outcome == "PASS").sum(), len(g)) for k, g in dec.groupby("level")}
    out["by_side"] = {k: rate((g.outcome == "PASS").sum(), len(g)) for k, g in dec.groupby("side")}
    out["by_weekday"] = {k: rate((g.outcome == "PASS").sum(), len(g)) for k, g in dec.groupby("weekday")}
    out["in_IL_window"] = {**split(dec.in_window, "in"), **split(~dec.in_window, "out")}
    m = dec.dropna(subset=["mfi"])
    out["by_MFI"] = {"mfi<30": rate((m[m.mfi < 30].outcome == "PASS").sum(), len(m[m.mfi < 30])),
                     "30-70": rate((m[(m.mfi >= 30) & (m.mfi <= 70)].outcome == "PASS").sum(), len(m[(m.mfi >= 30) & (m.mfi <= 70)])),
                     "mfi>70": rate((m[m.mfi > 70].outcome == "PASS").sum(), len(m[m.mfi > 70]))}
    sw, ns = dec[dec.sweep], dec[~dec.sweep]
    out["REJECT_rate_after_sweep"] = rate((sw.outcome == "REJECT").sum(), len(sw))
    out["REJECT_rate_without_sweep"] = rate((ns.outcome == "REJECT").sum(), len(ns))
    ex = d.dropna(subset=["ext_pct"])["ext_pct"]
    out["after_PASS_extension_pct"] = ({"n": int(len(ex)), "p25": round(float(ex.quantile(.25)), 2),
                                        "median": round(float(ex.median()), 2), "p75": round(float(ex.quantile(.75)), 2)}
                                       if len(ex) else None)
    return out


# ---------- null model: same volatility, no structure ----------
def daily_from_hourly(h):
    g = h.groupby(list(trade_date(h.index))).agg(High=("High", "max"), Low=("Low", "min"))
    g.index = pd.to_datetime(g.index)
    return g


def bootstrap_null(h, seed, block=24):
    """Rebuild a price path from 24-hour blocks of the real bars' returns/ranges/volume.
    Keeps the volatility level and clustering, destroys time-of-day, weekday and
    level-reaction structure. Any 'edge' must beat what this produces by chance."""
    rng = np.random.default_rng(seed)
    n = len(h)
    o, c = h["Open"].values, h["Close"].values
    ret = np.log(c / o)
    up = (h["High"].values - np.maximum(o, c)) / o
    dn = (np.minimum(o, c) - h["Low"].values) / o
    starts = rng.integers(0, n - block, size=n // block + 1)
    idx = np.concatenate([np.arange(s0, s0 + block) for s0 in starts])[:n]
    C = o[0] * np.exp(np.cumsum(ret[idx]))
    O = np.r_[o[0], C[:-1]]
    H = np.maximum(O, C) * (1 + up[idx])
    L = np.minimum(O, C) * (1 - dn[idx])
    df = pd.DataFrame({"Open": O, "High": H, "Low": L, "Close": C, "Volume": h["Volume"].values[idx]}, index=h.index)
    return add_mfi(df)


def analyze(h):
    levels = build_levels(daily_from_hourly(h))
    return {"events": test_events(h, levels), "touches": summarize_touches(find_touches(h, levels))}


def headline(r):
    """Rate metrics keep k/n and the Wilson interval so real vs null can be judged
    with the sample size in mind; plain numbers are kept for the others."""
    t, e = r["touches"], r["events"]
    return {
        "events_per_year": round(e["n_events"] / 2.0, 1) if e.get("n_events") is not None else None,
        "A_share_in_window": e.get("A_share_in_window"),
        "C_near_level": e.get("C_extreme_near_level"),
        "touch_pass_rate": t.get("overall_pass_rate"),
        "pass_rate_in_window": (t.get("in_IL_window") or {}).get("in"),
        "reject_after_sweep": t.get("REJECT_rate_after_sweep"),
        "reject_without_sweep": t.get("REJECT_rate_without_sweep"),
        "median_extension_after_pass_pct": (t.get("after_PASS_extension_pct") or {}).get("median"),
    }


def _val(x):
    return x["rate"] if isinstance(x, dict) else x


def compare(real_hl, null_hls):
    comp = {}
    for k, rv in real_hl.items():
        nv = [_val(x[k]) for x in null_hls if x.get(k) is not None and _val(x[k]) is not None]
        row = {"real": _val(rv), "null_mean": round(float(np.mean(nv)), 3) if nv else None,
               "null_sd": round(float(np.std(nv)), 3) if nv else None}
        if isinstance(rv, dict) and nv and rv.get("ci95") and rv["ci95"][0] is not None:
            row["real_n"] = rv["n"]
            row["real_ci95"] = list(rv["ci95"])
            row["EDGE_SIGNAL"] = not (rv["ci95"][0] <= np.mean(nv) <= rv["ci95"][1])
        comp[k] = row
    return comp


# ---------- data + main ----------
def fetch():
    import yfinance as yf
    h = yf.download(TICKER, period="730d", interval="1h", progress=False, auto_adjust=False)
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    h.index = h.index.tz_localize("UTC") if h.index.tz is None else h.index
    h = h[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    h["Volume"] = h["Volume"].fillna(0)
    return add_mfi(h)


def main():
    h = fetch()
    print(f"hourly bars: {len(h)}  {h.index.min()} -> {h.index.max()}")
    real = analyze(h)
    reps = int(os.getenv("NULL_REPS", "20"))
    nulls = [headline(analyze(bootstrap_null(h, seed))) for seed in range(reps)]
    comp = compare(headline(real), nulls)
    rep = {"data": {"hourly_bars": len(h), "from": str(h.index.min()), "to": str(h.index.max())},
           "REAL_vs_NULL (EDGE_SIGNAL=true only when the null mean is outside the real 95% interval; 8 rate metrics, so ~1 false alarm is expected by chance)": comp,
           "events_detail": real["events"], "touches_detail": real["touches"]}
    txt = json.dumps(rep, indent=2, default=str)
    print(txt)
    open("hunter_research_report.json", "w").write(txt)
    print("\nWrote hunter_research_report.json (nothing else was changed).")


if __name__ == "__main__":
    main()
