#!/usr/bin/env python3
"""
bot_sim.py  --  READ-ONLY (created 2026-10-12)

"Bot-mode" simulation: every time a weekly setup (entry / stop / target) is issued, it is pasted
into the platform as three orders and then NOTHING is touched.

Rules used (plain):
  * Plan list = plans_history.json (every setup that was ever frozen, including quiet weeks,
    rebuilt from the repo's git history). A plan lasts from its freeze time until the next plan's
    freeze time (a new plan = the old one was broken, or a new week started).
  * ENTRY order at the plan's entry price. Filled when an hourly bar trades through it. Not filled
    before the plan ends = no trade.
  * STOP and TARGET orders at the plan's levels. If a bar OPENS beyond the level (gap), the fill is
    at the open (slippage), not at the level.
  * If a bar touches both stop and target, the hourly bar cannot show which came first: results are
    shown pessimistic (stop first) and optimistic (target first).
  * If neither is hit before the next plan arrives, the position is closed at the next plan's entry
    price (= the market price at that moment).
  * Cost: 0.50 cent spread per trade (Plus500). No overnight financing, no slippage beyond gaps.
  * 1 cent per bushel = $1 per 100 bushels.
Plans with absurd width (> 5% stop distance, an old band bug) and plans overlapping glitch-cleaned
bars are skipped and listed. Never writes repo state, never sends Telegram, never commits.
Needs hunter_pattern_research.py and plans_history.json in the same folder.
"""
import json
import statistics as stt

import pandas as pd

import hunter_pattern_research as hp

SPREAD = 0.50
MAX_STOP_PCT = 5.0


def load_plans():
    raw = json.load(open("plans_history.json"))
    raw = [p for p in raw if p.get("frozen_at") and p.get("entry") and p.get("stop") and p.get("target")]
    raw.sort(key=lambda p: pd.Timestamp(p["frozen_at"]))
    plans = []
    for i, p in enumerate(raw):
        start = pd.Timestamp(p["frozen_at"])
        nxt = raw[i + 1] if i + 1 < len(raw) else None
        plans.append({
            "week": p["iso_week"], "entry": float(p["entry"]), "stop": float(p["stop"]), "target": float(p["target"]),
            "side": "DOWN" if p["stop"] > p["target"] else "UP",
            "start": start,
            "end": pd.Timestamp(nxt["frozen_at"]) if nxt else None,
            "exit_mkt": float(nxt["entry"]) if nxt else None,
        })
    return plans


def run_plan(p, bars, optimistic):
    """returns dict(result, pnl_cents) ; result in NOFILL, WIN, LOSS, FORCED, OPEN"""
    e, s, g, side = p["entry"], p["stop"], p["target"], p["side"]
    seg = bars[(bars.index >= p["start"].floor("h")) & (bars.index <= p["end"])]
    sign = 1 if side == "UP" else -1
    filled, fill_ts, amb = False, None, False
    for ts, b in seg.iterrows():
        if not filled:
            if b["Low"] <= e <= b["High"]:
                filled, fill_ts = True, ts
            continue                      # exits are checked from the bar AFTER the fill
        hit_stop = b["Low"] <= s if side == "UP" else b["High"] >= s
        hit_tgt = b["High"] >= g if side == "UP" else b["Low"] <= g
        if hit_stop and hit_tgt:
            amb = True
            hit_stop, hit_tgt = (False, True) if optimistic else (True, False)
        if hit_stop:
            px = b["Open"] if ((side == "UP" and b["Open"] < s) or (side == "DOWN" and b["Open"] > s)) else s
            return {"result": "LOSS", "pnl": sign * (px - e) - SPREAD, "amb": amb}
        if hit_tgt:
            px = b["Open"] if ((side == "UP" and b["Open"] > g) or (side == "DOWN" and b["Open"] < g)) else g
            return {"result": "WIN", "pnl": sign * (px - e) - SPREAD, "amb": amb}
    if not filled:
        return {"result": "NOFILL", "pnl": 0.0, "amb": False}
    return {"result": "FORCED", "pnl": sign * (p["exit_mkt"] - e) - SPREAD, "amb": False}


def stats(rows):
    tr = [r for r in rows if r["result"] in ("WIN", "LOSS", "FORCED")]
    w = [r["pnl"] for r in tr if r["result"] == "WIN"]
    l = [r["pnl"] for r in tr if r["result"] == "LOSS"]
    f = [r["pnl"] for r in tr if r["result"] == "FORCED"]
    cum, peak, dd, cur_l, worst_l = 0.0, 0.0, 0.0, 0, 0
    for r in tr:
        cum += r["pnl"]
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
        cur_l = cur_l + 1 if r["pnl"] < 0 else 0
        worst_l = max(worst_l, cur_l)
    tot = sum(r["pnl"] for r in tr)
    return {
        "trades": len(tr), "no_fill": sum(r["result"] == "NOFILL" for r in rows),
        "wins": len(w), "losses": len(l), "forced_exits": len(f),
        "avg_win_c": round(stt.mean(w), 2) if w else None, "avg_loss_c": round(stt.mean(l), 2) if l else None,
        "avg_forced_c": round(stt.mean(f), 2) if f else None, "forced_positive": sum(x > 0 for x in f),
        "win_rate_of_all_trades": round(sum(x > 0 for x in [r["pnl"] for r in tr]) / len(tr), 3) if tr else None,
        "total_net_cents": round(tot, 1), "total_net_usd_per_100bu": round(tot, 0),
        "avg_per_trade_c": round(tot / len(tr), 2) if tr else None,
        "max_drawdown_c": round(dd, 1), "worst_losing_streak": worst_l,
    }


def main():
    bars = hp.fetch()
    bad = bars[hp.bad_array(bars)].index if "Bad" in bars.columns else pd.DatetimeIndex([])
    plans = load_plans()
    use, skipped = [], []
    for p in plans:
        if p["end"] is None:
            skipped.append({"week": p["week"], "start": str(p["start"])[:16], "why": "still open (current plan)"}); continue
        if abs(p["stop"] - p["entry"]) / p["entry"] * 100 > MAX_STOP_PCT:
            skipped.append({"week": p["week"], "start": str(p["start"])[:16], "why": "absurd width (old band bug)"}); continue
        if bars.index.min() > p["start"] or bars.index.max() < p["end"]:
            skipped.append({"week": p["week"], "start": str(p["start"])[:16], "why": "no price data"}); continue
        if len(bad) and ((bad >= p["start"].floor("h")) & (bad <= p["end"])).any():
            skipped.append({"week": p["week"], "start": str(p["start"])[:16], "why": "glitch-cleaned bars inside span"}); continue
        use.append(p)
    pess = [dict(run_plan(p, bars, False), week=p["week"], start=str(p["start"])[:16], side=p["side"]) for p in use]
    opt = [run_plan(p, bars, True) for p in use]
    for r, o in zip(pess, opt):
        r["optimistic_result"], r["optimistic_pnl"] = o["result"], round(o["pnl"], 2)
        r["pnl"] = round(r["pnl"], 2)
    o_rows = [{"result": r["optimistic_result"], "pnl": r["optimistic_pnl"]} for r in pess]
    rep = {
        "assumptions": "entry/stop/target orders pasted at plan start, never touched; gap fills at open; spread 0.5c; unfinished plan closed at next plan's entry price",
        "plans_simulated": len(use), "plans_skipped": skipped,
        "PESSIMISTIC_(both-touched bar = stop first)": stats(pess),
        "OPTIMISTIC_(both-touched bar = target first)": stats(o_rows),
        "ambiguous_bars": sum(r["amb"] for r in pess), "per_plan": pess,
    }
    txt = json.dumps(rep, indent=2, default=str)
    print(txt)
    open("bot_sim_report.json", "w").write(txt)
    print("\nWrote bot_sim_report.json (nothing else was changed).")


if __name__ == "__main__":
    main()
