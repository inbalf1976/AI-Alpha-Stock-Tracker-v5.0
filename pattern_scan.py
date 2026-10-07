#!/usr/bin/env python3
"""
pattern_scan.py  --  READ-ONLY research (created 2026-10-07)

Analyst-style search for ANY predictable structure in WHEN and WHICH WAY wheat
makes rapid moves, using everything available at each hour (time of day,
weekday, recent returns, volatility compression, volume, MFI, distance to
previous-day/week highs-lows and round-50 levels, session).

Two questions, answered with strict walk-forward testing (train on the past,
score on the future, 3 folds, embargo so forward windows cannot leak):
  1. TIMING    - will a rapid move (>= MOVE_PCT within HORIZON hours) start now?
  2. DIRECTION - given the next HORIZON hours move at least DIR_MIN_PCT, up or down?

Guards against fooling ourselves:
  * a time-of-day-only model is reported next to the full model, so volatility
    seasonality is not mistaken for a hidden pattern
  * p-values come from circularly shifting the TEST labels (keeps their
    autocorrelation) 300 times
  * a pattern counts only if pooled out-of-sample AUC beats 0.5 with p < 0.01
    AND holds in most folds
  * hour-of-day drift table uses day-clustered errors and Benjamini-Hochberg FDR

What this cannot see: order-book depth, order sizes, account identities. If a
controlled hunt leaves its footprint only there, hourly bars cannot show it.

Needs hunter_pattern_research.py in the same folder (cleaned data helpers).
Never writes repo state, never sends Telegram, never commits.
"""
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

import hunter_pattern_research as hp

MOVE_PCT = float(os.getenv("MOVE_PCT", "2.5"))
HORIZON = int(os.getenv("HORIZON", "6"))
DIR_MIN_PCT = float(os.getenv("DIR_MIN_PCT", "1.0"))
PERMS = int(os.getenv("PERMS", "300"))
SEED = 7


# ---------------- dataset ----------------
def build_dataset(h):
    n = len(h)
    o, hi, lo, c = (h[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    v = h["Volume"].values.astype(float)
    mfi = h["MFI"].values.astype(float)
    bad = hp.bad_array(h)
    il = h.index.tz_convert(hp.TZ_IL)
    tdates = np.array(hp.trade_date(h.index), dtype=object)
    levels = hp.levels_for(h)

    S = pd.Series
    cs = S(c)
    rng_pct = S((hi - lo) / c * 100)
    f = pd.DataFrame(index=h.index)
    # every feature uses bars strictly BEFORE bar t (shift(1)) -> known at the open of t
    f["hour_il"] = il.hour
    f["weekday"] = il.weekday
    f["us_session"] = ((h.index.tz_convert(hp.TZ_CT).hour >= 8) & (h.index.tz_convert(hp.TZ_CT).hour < 14)).astype(int)
    f["r1"] = ((cs / S(o) - 1) * 100).shift(1).values
    f["r3"] = ((cs / cs.shift(3) - 1) * 100).shift(1).values
    f["r6"] = ((cs / cs.shift(6) - 1) * 100).shift(1).values
    f["r24"] = ((cs / cs.shift(24) - 1) * 100).shift(1).values
    atr6, atr72 = rng_pct.rolling(6).mean(), rng_pct.rolling(72).mean()
    f["atr_ratio"] = (atr6 / atr72).shift(1).values
    hh6, ll6 = S(hi).rolling(6).max(), S(lo).rolling(6).min()
    hh24, ll24 = S(hi).rolling(24).max(), S(lo).rolling(24).min()
    f["compression"] = (((hh6 - ll6) / (hh24 - ll24)).shift(1)).values
    f["vol_ratio"] = (S(v).rolling(3).mean() / S(v).rolling(72).mean().replace(0, np.nan)).shift(1).values
    f["mfi"] = S(mfi).shift(1).values
    for nm in ("PDH", "PDL", "PWH", "PWL"):
        col = np.full(n, np.nan)
        for i in range(n):
            L = levels.get(tdates[i], {}).get(nm)
            if L:
                col[i] = (o[i] - L) / L * 100
        f["d_" + nm] = col
    f["d_r50"] = (o - np.round(o / 50) * 50) / o * 100

    # targets: window = bars t .. t+HORIZON-1, measured from the open of t
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    for i in range(n - HORIZON):
        w = slice(i, i + HORIZON)
        up[i] = (hi[w].max() - o[i]) / o[i] * 100
        dn[i] = (o[i] - lo[w].min()) / o[i] * 100
        ret[i] = (c[i + HORIZON - 1] - o[i]) / o[i] * 100
    f["fwd_up"], f["fwd_dn"], f["fwd_ret"] = up, dn, ret
    f["event"] = (np.maximum(up, dn) >= MOVE_PCT).astype(float)
    f.loc[np.isnan(up), "event"] = np.nan

    # a row is usable only if nothing contaminated lies in its look-back or look-forward
    ok = np.ones(n, bool)
    for i in range(n):
        if bad[max(0, i - 72):i + HORIZON + 1].any():
            ok[i] = False
    f["ok"] = ok & ~np.isnan(up) & (np.arange(n) >= 72)
    f["day"] = tdates
    return f[f["ok"]].drop(columns="ok")


FEATS = ["hour_il", "weekday", "us_session", "r1", "r3", "r6", "r24", "atr_ratio", "compression",
         "vol_ratio", "mfi", "d_PDH", "d_PDL", "d_PWH", "d_PWL", "d_r50"]
TIME_ONLY = ["hour_il", "weekday"]
CAT = {"hour_il", "weekday"}


# ---------------- walk-forward ----------------
def folds(n, k=3, start=0.55):
    cuts = np.linspace(start, 1.0, k + 1)
    return [(int(cuts[i] * n), int(cuts[i + 1] * n)) for i in range(k)]


def fit_predict(Xtr, ytr, Xte, cols):
    cat = [cols.index(c) for c in cols if c in CAT]
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0,
                                       categorical_features=cat or None, random_state=SEED)
    m.fit(Xtr, ytr)
    return m, m.predict_proba(Xte)[:, 1]


def shift_perm_p(y, p, reps=PERMS, seed=SEED):
    """p-value for AUC by circularly shifting labels (keeps their autocorrelation)."""
    rng = np.random.default_rng(seed)
    real = roc_auc_score(y, p)
    cnt, n = 0, len(y)
    for _ in range(reps):
        s = rng.integers(n // 10, n - n // 10)
        if roc_auc_score(np.roll(y, s), p) >= real:
            cnt += 1
    return real, (1 + cnt) / (1 + reps)


def evaluate(df, target, cols, label):
    y_all = df[target].values
    X_all = df[cols].values.astype(float)
    n = len(df)
    ys, ps, per_fold = [], [], []
    last_model, last_Xte, last_yte = None, None, None
    for a, b in folds(n):
        tr_end = a - HORIZON * 2                       # embargo: forward windows overlap
        ytr, yte = y_all[:tr_end], y_all[a:b]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        m, p = fit_predict(X_all[:tr_end], ytr, X_all[a:b], cols)
        per_fold.append(round(float(roc_auc_score(yte, p)), 3))
        ys.append(yte)
        ps.append(p)
        last_model, last_Xte, last_yte = m, X_all[a:b], yte
    if not ys:
        return {"label": label, "error": "not enough data/classes"}, None
    y, p = np.concatenate(ys), np.concatenate(ps)
    auc, pv = shift_perm_p(y.astype(int), p)
    out = {"label": label, "n_test": int(len(y)), "base_rate": round(float(y.mean()), 3),
           "pooled_auc": round(float(auc), 3), "p_value": round(float(pv), 4), "auc_per_fold": per_fold}
    return out, (last_model, last_Xte, last_yte)


def hour_drift(df):
    """mean forward 6h return by Israel hour with day-clustered SE and BH-FDR."""
    rows = []
    for hr, g in df.groupby("hour_il"):
        d = g.groupby("day")["fwd_ret"].mean()
        if len(d) < 30:
            continue
        mu, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        z = mu / se if se > 0 else 0.0
        p = float(math.erfc(abs(z) / math.sqrt(2)))
        rows.append({"hour_il": int(hr), "days": int(len(d)), "mean_fwd6h_pct": round(mu, 3), "t": round(z, 2), "p": p})
    if not rows:
        return []
    ps = np.array([r["p"] for r in rows])
    order = np.argsort(ps)
    m = len(ps)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        prev = min(prev, ps[idx] * m / rank)
        q[idx] = prev
    for r, qq in zip(rows, q):
        r["q_fdr"] = round(float(qq), 3)
        r["p"] = round(r["p"], 4)
    return rows


def main():
    h = hp.fetch()
    df = build_dataset(h)
    print(f"usable rows: {len(df)}  events (>= {MOVE_PCT}% in {HORIZON}h): {int(df['event'].sum())}")
    rep = {"data": {"usable_rows": int(len(df)), "events": int(df["event"].sum()), "move_pct": MOVE_PCT,
                    "horizon_h": HORIZON, "dir_min_pct": DIR_MIN_PCT,
                    "rule": "a pattern needs pooled out-of-sample AUC > 0.5 with p < 0.01 AND most folds > 0.5"}}

    # 1. timing
    full, art = evaluate(df, "event", FEATS, "TIMING full model")
    tonly, _ = evaluate(df, "event", TIME_ONLY, "TIMING hour+weekday only")
    rep["timing"] = {"full": full, "time_of_day_only": tonly}
    if art and art[0] is not None and "pooled_auc" in full and full["p_value"] < 0.05:
        pi = permutation_importance(art[0], art[1], art[2], scoring="roc_auc", n_repeats=8, random_state=SEED)
        rep["timing"]["top_features_last_fold"] = [(FEATS[i], round(float(pi.importances_mean[i]), 4))
                                                   for i in np.argsort(-pi.importances_mean)[:6]]

    # 2. direction
    dd = df[np.abs(df["fwd_ret"]) >= DIR_MIN_PCT].copy()
    dd["up"] = (dd["fwd_ret"] > 0).astype(float)
    dfull, dart = evaluate(dd, "up", FEATS, "DIRECTION full model")
    dt, _ = evaluate(dd, "up", TIME_ONLY, "DIRECTION hour+weekday only")
    rep["direction"] = {"rows": int(len(dd)), "full": dfull, "time_of_day_only": dt}
    if dart and dart[0] is not None and "pooled_auc" in dfull and dfull["p_value"] < 0.05:
        pi = permutation_importance(dart[0], dart[1], dart[2], scoring="roc_auc", n_repeats=8, random_state=SEED)
        rep["direction"]["top_features_last_fold"] = [(FEATS[i], round(float(pi.importances_mean[i]), 4))
                                                      for i in np.argsort(-pi.importances_mean)[:6]]

    # 3. hour-of-day drift
    rep["hour_drift"] = hour_drift(df)

    def passes(r):
        if "pooled_auc" not in r:
            return False
        good = sum(a > 0.5 for a in r["auc_per_fold"])
        return r["p_value"] < 0.01 and r["pooled_auc"] > 0.5 and good >= len(r["auc_per_fold"])

    t_gain = (full.get("pooled_auc", 0) - tonly.get("pooled_auc", 0))
    rep["verdicts"] = {
        "timing_time_of_day_effect": "YES (normal intraday volatility pattern)" if passes(tonly) else "no",
        "timing_extra_beyond_time_of_day": ("PATTERN CANDIDATE (full model beats hour-only by %.3f AUC)" % t_gain)
                                           if passes(full) and t_gain >= 0.03 else "nothing beyond time-of-day",
        "direction_full_model": "PATTERN CANDIDATE" if passes(dfull) else "no reliable pattern",
        "direction_time_only": "PATTERN CANDIDATE" if passes(dt) else "no reliable pattern",
    }
    txt = json.dumps(rep, indent=2, default=str)
    print(txt)
    open("pattern_scan_report.json", "w").write(txt)
    print("\nWrote pattern_scan_report.json (nothing else was changed).")


if __name__ == "__main__":
    main()
