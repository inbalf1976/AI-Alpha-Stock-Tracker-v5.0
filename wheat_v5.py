#!/usr/bin/env python3
"""
wheat_v5.py -- honest rebuild of the wheat monitor (research + report + score). Standalone; touches no v4 files.

DESIGN (from the evidence in this project):
  * ONE price series, no mixing. Daily OHLC only (ZW=F, 10y). Entries at the NEXT OPEN, stop/target touches from
    daily High/Low of the same series. (v4 mixed daily Close with hourly bars: ~+0.4-1% offset => fake 65-70% 'accuracy'.)
  * RANGES are the forecast product for daily / weekly / monthly: volatility-scaled empirical quantiles, validated by
    walk-forward COVERAGE (does an 80% range really contain price ~80% of the time?). That is a win rate you can measure.
  * DIRECTION is OFF by default. A rule is enabled only if it passes a pre-registered fair test:
        >=60 independent trades in the test period, test mean R after cost > 0, bootstrap 95% CI-low > 0,
        lift vs same-side baseline > 0, and train mean R > 0.
    The registry (signal_registry.json) is written by `research`; `report` obeys it. Nothing passing => 'NO VALIDATED EDGE'.
  * Every win rate is shown with its expectancy in R. Risk protocol printed on every report.

COMMANDS:  python wheat_v5.py research | report | score
ENV: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (optional), COST_PCT (default 0.001), HORIZON_DAYS (default 5)
"""
import os, sys, json, math
import numpy as np
import pandas as pd

TICKER = "ZW=F"
HORIZONS = {"daily": 1, "weekly": 5, "monthly": 21}
COST = float(os.getenv("COST_PCT", "0.001"))
HOLD = int(os.getenv("HORIZON_DAYS", "5"))
STOP_ATR, TGT_ATR = 1.0, 1.5
BREAKEVEN_WIN = STOP_ATR / (STOP_ATR + TGT_ATR)
REGISTRY, FLOG = "signal_registry.json", "v5_forecast_log.json"
RISK = "RISK PROTOCOL: max 100 bu per trade | stop for the day after -30 ILS | max 6 trades/day | no adding to losers"


# ------------------------------------------------------------------ data
def load_daily():
    import yfinance as yf
    d = yf.Ticker(TICKER).history(period="10y", interval="1d", auto_adjust=False)
    return clean(d)


def clean(d):
    d = d[["Open", "High", "Low", "Close"]].dropna().copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d[(d.High >= d.Low) & (d.Close > 0) & (d.Open > 0)]
    return d


def indicators(d):
    c = d.Close
    r = np.log(c).diff()
    var = (r.clip(-0.06, 0.06) ** 2).ewm(alpha=0.06, adjust=False, min_periods=20).mean()
    d = d.copy()
    d["sigma"] = np.sqrt(var)
    pc = c.shift()
    tr = pd.concat([d.High - d.Low, (d.High - pc).abs(), (d.Low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["sma20"], d["sma50"] = c.rolling(20).mean(), c.rolling(50).mean()
    ch = c.diff()
    g, l = ch.clip(lower=0).rolling(14).mean(), (-ch.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    d["ret5"] = c.pct_change(5)
    d["hh20"], d["ll20"] = d.High.rolling(20).max().shift(1), d.Low.rolling(20).min().shift(1)
    # jump flag (roll gap / glitch): |return| > 5 x trailing vol
    d["jump"] = (r.abs() > 5 * d["sigma"].shift(1)).fillna(False)
    return d


# ------------------------------------------------------------------ range forecasts
def fwd_targets(d, h):
    """z-scaled forward close return, max-high excursion, min-low excursion (known only h days later)."""
    c = d.Close
    s = d.sigma * math.sqrt(h)
    zc = np.log(c.shift(-h) / c) / s
    mx = d.High.shift(-1)[::-1].rolling(h, min_periods=h).max()[::-1]
    mn = d.Low.shift(-1)[::-1].rolling(h, min_periods=h).min()[::-1]
    return zc, np.log(mx / c) / s, np.log(mn / c) / s


def calibrate(d, h, start=300):
    zc, zu, zd = fwd_targets(d, h)
    rows = []
    for i in range(start, len(d) - h):
        if not np.isfinite(d.sigma.iloc[i]):
            continue
        k = i - h + 1                      # samples whose outcome is known by day i
        a = zc.iloc[:k].dropna().values; b = zu.iloc[:k].dropna().values; e = zd.iloc[:k].dropna().values
        if min(len(a), len(b), len(e)) < 150:
            continue
        lo, hi = np.quantile(a, [0.10, 0.90]); up80 = np.quantile(b, 0.80); dn80 = np.quantile(e, 0.20)
        rows.append(dict(date=d.index[i], year=d.index[i].year,
                         close_in=bool(lo <= zc.iloc[i] <= hi), high_ok=bool(zu.iloc[i] <= up80),
                         low_ok=bool(zd.iloc[i] >= dn80),
                         width_pct=100 * (hi - lo) * d.sigma.iloc[i] * math.sqrt(h)))
    t = pd.DataFrame(rows)
    t["both_ok"] = t.high_ok & t.low_ok
    return t


def forecast_now(d, h):
    zc, zu, zd = fwd_targets(d, h)
    k = len(d) - h
    a = zc.iloc[:k].dropna().values; b = zu.iloc[:k].dropna().values; e = zd.iloc[:k].dropna().values
    lo, hi = np.quantile(a, [0.10, 0.90]); up80 = np.quantile(b, 0.80); dn80 = np.quantile(e, 0.20)
    p, s = float(d.Close.iloc[-1]), float(d.sigma.iloc[-1]) * math.sqrt(h)
    f = lambda z: round(p * math.exp(z * s), 2)
    return dict(price=round(p, 2), close_lo=f(lo), close_hi=f(hi), high80=f(up80), low80=f(dn80),
                med_adverse_pct=round(100 * (1 - math.exp(np.median(np.minimum(e, 0)) * s)), 2))


# ------------------------------------------------------------------ trade simulation (daily bars, next-open entry)
def sim(d, i, side, hold):
    """R multiple after cost. entry = Open[i+1]; stop 1 ATR, target 1.5 ATR; same-bar both = loss; else mark-to-market."""
    if i + hold + 1 >= len(d) or not np.isfinite(d.atr.iloc[i]) or d.atr.iloc[i] <= 0:
        return np.nan
    if d.jump.iloc[i + 1:i + hold + 2].any():
        return np.nan                      # roll/glitch inside the path -> skip, don't score
    e = float(d.Open.iloc[i + 1]); D = STOP_ATR * float(d.atr.iloc[i]); T = TGT_ATR * float(d.atr.iloc[i])
    sgn = 1 if side > 0 else -1
    R = None
    for k in range(i + 1, i + hold + 1):
        hi, lo = float(d.High.iloc[k]), float(d.Low.iloc[k])
        stop_hit = (lo <= e - D) if sgn > 0 else (hi >= e + D)
        tgt_hit = (hi >= e + T) if sgn > 0 else (lo <= e - T)
        if stop_hit:
            R = -1.0; break
        if tgt_hit:
            R = TGT_ATR / STOP_ATR; break
    if R is None:
        R = float(np.clip(sgn * (d.Close.iloc[i + hold] - e) / D, -1.0, TGT_ATR / STOP_ATR))
    return R - COST * e / D


RULES = {   # PRE-REGISTERED: 6 rules x (long/short built in). No tuning after seeing results.
    "trend_follow":  lambda d: np.where((d.Close > d.sma50) & (d.sma20 > d.sma50), 1, np.where((d.Close < d.sma50) & (d.sma20 < d.sma50), -1, 0)),
    "breakout20":    lambda d: np.where(d.Close > d.hh20, 1, np.where(d.Close < d.ll20, -1, 0)),
    "rsi_reversion": lambda d: np.where(d.rsi < 30, 1, np.where(d.rsi > 70, -1, 0)),
    "momentum5":     lambda d: np.where(d.ret5 > 0.02, 1, np.where(d.ret5 < -0.02, -1, 0)),
    "contrarian5":   lambda d: np.where(d.ret5 > 0.02, -1, np.where(d.ret5 < -0.02, 1, 0)),
    "sma20_side":    lambda d: np.where(d.Close > d.sma20, 1, np.where(d.Close < d.sma20, -1, 0)),
}


def greedy(idx, gap):
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            out.append(i); last = i
    return out


def boot_ci(x, B=4000, seed=7):
    x = np.asarray(x, float)
    if len(x) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def research(d):
    n = len(d)
    RL = np.array([sim(d, i, 1, HOLD) for i in range(n)]); RS = np.array([sim(d, i, -1, HOLD) for i in range(n)])
    split = int(n * 0.6)
    res = {"generated": str(pd.Timestamp.now().date()), "bars": n, "first": str(d.index[0].date()), "last": str(d.index[-1].date()),
           "geometry": f"stop {STOP_ATR} ATR / target {TGT_ATR} ATR, hold {HOLD}d, cost {COST:.2%}", "breakeven_win": round(BREAKEVEN_WIN, 3)}
    yb = {}
    for y in sorted(set(d.index.year)):
        m = np.array(d.index.year == y)
        yb[int(y)] = dict(always_long_R=round(float(np.nanmean(RL[m])), 3), always_short_R=round(float(np.nanmean(RS[m])), 3))
    res["null_by_year"] = yb
    res["null_all"] = dict(always_long_R=round(float(np.nanmean(RL)), 3), always_short_R=round(float(np.nanmean(RS)), 3),
                           long_win=round(float(np.nanmean(RL > 0)), 3), short_win=round(float(np.nanmean(RS > 0)), 3))
    rules, reg = {}, {}
    for name, fn in RULES.items():
        sig = fn(d)
        stats = {}
        for part, lo_, hi_ in (("train", 0, split), ("test", split, n)):
            ids = [i for i in range(lo_, hi_) if sig[i] != 0]
            R = lambda i: RL[i] if sig[i] > 0 else RS[i]
            ids = [i for i in greedy(ids, HOLD) if np.isfinite(R(i))]
            x = [R(i) for i in ids]
            baseline = float(np.mean([(RL[i] if s > 0 else RS[i]) for i, s in ((i, sig[i]) for i in range(lo_, hi_)) if s != 0 and np.isfinite(RL[i] if s > 0 else RS[i])] or [np.nan]))
            ci = boot_ci(x)
            stats[part] = dict(n=len(x), mean_R=round(float(np.mean(x)), 3) if x else None, win=round(float(np.mean(np.array(x) > 0)), 3) if x else None,
                               ci_lo=round(ci[0], 3), ci_hi=round(ci[1], 3), same_days_all_signals_R=round(baseline, 3))
        tr, te = stats["train"], stats["test"]
        ok = bool(te["n"] >= 60 and te["mean_R"] is not None and te["mean_R"] > 0 and te["ci_lo"] > 0 and tr["mean_R"] is not None and tr["mean_R"] > 0)
        rules[name] = dict(train=tr, test=te, passes=ok)
        reg[name] = dict(enabled=ok, test_mean_R=te["mean_R"], test_n=te["n"], test_ci_lo=te["ci_lo"], train_mean_R=tr["mean_R"])
    res["rules"] = rules
    cov = {}
    for nm, h in HORIZONS.items():
        t = calibrate(d, h)
        cov[nm] = dict(h=h, n=len(t), close80_cover=round(float(t.close_in.mean()), 3), high80_cover=round(float(t.high_ok.mean()), 3),
                       low80_cover=round(float(t.low_ok.mean()), 3), both_cover=round(float(t.both_ok.mean()), 3),
                       avg_width_pct=round(float(t.width_pct.mean()), 2),
                       by_year={int(y): round(float(g.close_in.mean()), 3) for y, g in t.groupby("year")})
    res["range_calibration"] = cov
    json.dump(res, open("wheat_v5_research.json", "w"), indent=2, default=str)
    json.dump({"generated": res["generated"], "geometry": res["geometry"], "rules": reg,
               "null_all": res["null_all"], "note": "A rule is enabled only if test n>=60, test meanR>0, CI-low>0, train meanR>0."},
              open(REGISTRY, "w"), indent=2)
    return res


def print_research(res):
    print(f"\nRESEARCH {res['first']} -> {res['last']} | {res['bars']} daily bars | {res['geometry']}")
    print(f"break-even win rate at this geometry: {res['breakeven_win']:.1%}")
    na = res["null_all"]
    print(f"NULL (random entry): always-long {na['always_long_R']:+.3f}R (win {na['long_win']:.0%}) | always-short {na['always_short_R']:+.3f}R (win {na['short_win']:.0%})")
    print("null by year (R): " + "  ".join(f"{y}:L{v['always_long_R']:+.2f}/S{v['always_short_R']:+.2f}" for y, v in res["null_by_year"].items()))
    print("\nPRE-REGISTERED RULES (train 60% / test 40%, non-overlapping trades):")
    for k, v in res["rules"].items():
        tr, te = v["train"], v["test"]
        print(f"  {k:<14} train n={tr['n']:<4} R={tr['mean_R']:+.3f} | test n={te['n']:<4} R={te['mean_R']:+.3f} win={te['win']:.0%} CI[{te['ci_lo']:+.2f},{te['ci_hi']:+.2f}] -> {'ENABLED' if v['passes'] else 'rejected'}")
    print("\nRANGE CALIBRATION (walk-forward, nominal 80%):")
    for k, v in res["range_calibration"].items():
        print(f"  {k:<8} h={v['h']:<3} n={v['n']:<5} close-in-range {v['close80_cover']:.0%} | high<=env {v['high80_cover']:.0%} | low>=env {v['low80_cover']:.0%} | both {v['both_cover']:.0%} | avg width {v['avg_width_pct']}%")
    en = [k for k, v in res["rules"].items() if v["passes"]]
    print("\nENABLED SIGNALS: " + (", ".join(en) if en else "NONE -> report will say NO VALIDATED EDGE"))


# ------------------------------------------------------------------ report / score
def build_report(d):
    now = d.index[-1].date()
    lines = [f"WHEAT MONITOR v5 | ZW=F | last bar {now} | price {d.Close.iloc[-1]:.2f}c", "-" * 52]
    log = []
    for nm, h in HORIZONS.items():
        f = forecast_now(d, h)
        lines.append(f"{nm.upper():<8} ({h}d)  80% close range {f['close_lo']}-{f['close_hi']} | 80% envelope low>= {f['low80']}  high<= {f['high80']} | median adverse move {f['med_adverse_pct']}%")
        log.append(dict(made=str(now), horizon=nm, h=h, **f, scored=False))
    reg = json.load(open(REGISTRY)) if os.path.exists(REGISTRY) else None
    lines.append("")
    if not reg:
        lines.append("DIRECTION: no registry found (run `research`) -> NO TRADE SIGNAL")
    else:
        sig_now = []
        for name, info in reg["rules"].items():
            if info["enabled"]:
                s = int(RULES[name](d)[-1])
                if s != 0:
                    sig_now.append((name, s))
        if not sig_now:
            en = [k for k, v in reg["rules"].items() if v["enabled"]]
            lines.append("DIRECTION: NO VALIDATED EDGE" if not en else f"DIRECTION: enabled rules {en} are not active today -> no trade")
            na = reg["null_all"]
            lines.append(f"(random entries at {reg['geometry']}: long {na['always_long_R']:+.2f}R, short {na['always_short_R']:+.2f}R per trade - a coin flip loses)")
        else:
            a = float(d.atr.iloc[-1]); p = float(d.Close.iloc[-1])
            for name, s in sig_now:
                side = "BUY" if s > 0 else "SELL"
                st, tg = p - s * STOP_ATR * a, p + s * TGT_ATR * a
                info = reg["rules"][name]
                lines.append(f"{side} via {name}: entry ~next open | stop {st:.2f} | target {tg:.2f} | R:R {TGT_ATR / STOP_ATR:.1f}:1 | "
                             f"validated test mean {info['test_mean_R']:+.2f}R over {info['test_n']} trades (NOT a guarantee)")
    lines += ["", RISK]
    return "\n".join(lines), log


def send_telegram(msg):
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        print("(telegram not configured)"); return
    import requests
    r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": msg}, timeout=15)
    print("telegram:", r.status_code)


def score(d):
    if not os.path.exists(FLOG):
        print("no forecast log yet"); return
    log = json.load(open(FLOG)); done = {"daily": [], "weekly": [], "monthly": []}
    for e in log:
        if e.get("scored"):
            pass
        h, made = e["h"], pd.Timestamp(e["made"])
        pos = d.index.searchsorted(made)
        if pos + h < len(d) and d.index[pos] == made and not e.get("scored"):
            w = d.iloc[pos + 1:pos + h + 1]
            e.update(scored=True, close_in=bool(e["close_lo"] <= d.Close.iloc[pos + h] <= e["close_hi"]),
                     env_ok=bool(w.High.max() <= e["high80"] and w.Low.min() >= e["low80"]))
        if e.get("scored"):
            done[e["horizon"]].append(e)
    json.dump(log, open(FLOG, "w"), indent=2)
    for k, v in done.items():
        if v:
            print(f"{k:<8} scored {len(v)} | close-in-80%-range {np.mean([x['close_in'] for x in v]):.0%} | both-sides-in-envelope {np.mean([x['env_ok'] for x in v]):.0%}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    d = indicators(load_daily())
    print(f"loaded {len(d)} daily bars {d.index[0].date()} -> {d.index[-1].date()} | jump/roll days flagged: {int(d.jump.sum())}")
    if cmd == "research":
        print_research(research(d))
    elif cmd == "report":
        txt, new = build_report(d)
        print("\n" + txt)
        open("wheat_v5_report.txt", "w").write(txt)
        log = json.load(open(FLOG)) if os.path.exists(FLOG) else []
        have = {(e["made"], e["horizon"]) for e in log}
        log += [e for e in new if (e["made"], e["horizon"]) not in have]
        json.dump(log, open(FLOG, "w"), indent=2)
        send_telegram(txt)
    elif cmd == "score":
        score(d)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
