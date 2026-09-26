import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# Load the exact V5 spike detector from the repo so the backtest uses the
# same directional scoring logic as production.
import anti_hunt_filter as v5

CHICAGO_TZ = ZoneInfo("America/Chicago")
TICKER = os.environ.get("TICKER", "ZW=F")
PERIOD = os.environ.get("BACKTEST_PERIOD", "60d")
HORIZON_BARS = int(os.environ.get("SPIKE_HORIZON_BARS", "32"))
MIN_BARS = int(os.environ.get("SPIKE_MIN_BARS", "35"))

# Directional spike setup geometry requested for V5.
# Example: UP entry 683 -> stop 678 -> target 715.
SPIKE_STOP_DISTANCE = float(os.environ.get("SPIKE_STOP_DISTANCE", "5.0"))
SPIKE_TARGET_DISTANCE = float(os.environ.get("SPIKE_TARGET_DISTANCE", "32.0"))

OUTPUT_JSON = os.environ.get("BACKTEST_OUTPUT", "backtest_v5_spike_report.json")
OUTPUT_CSV = os.environ.get("BACKTEST_CSV", "backtest_v5_spike_trades.csv")


def clean_download(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[needed].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index = v5.chicago_index(df)
    return df.sort_index()


def directional_setup(direction, entry):
    entry = float(v5.round_tick(entry))
    if direction == "UP":
        stop = v5.round_tick(entry - SPIKE_STOP_DISTANCE)
        target = v5.round_tick(entry + SPIKE_TARGET_DISTANCE)
    elif direction == "DOWN":
        stop = v5.round_tick(entry + SPIKE_STOP_DISTANCE)
        target = v5.round_tick(entry - SPIKE_TARGET_DISTANCE)
    else:
        return None
    risk = abs(entry - stop)
    reward = abs(target - entry)
    return {
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": round(risk, 4),
        "reward": round(reward, 4),
        "rr": round(reward / risk, 2) if risk else 0.0,
    }


def resolve_trade(df, signal_idx, setup):
    pos = df.index.get_loc(signal_idx)
    future = df.iloc[pos + 1: pos + 1 + HORIZON_BARS]
    if future.empty:
        return "EXPIRED", None, None, 0

    direction = setup["direction"]
    entry = setup["entry"]
    stop = setup["stop"]
    target = setup["target"]

    mfe = 0.0
    mae = 0.0
    entry_time = signal_idx

    for bars_after, (ts, row) in enumerate(future.iterrows(), start=1):
        high = float(row["High"])
        low = float(row["Low"])

        if direction == "UP":
            mfe = max(mfe, high - entry)
            mae = max(mae, entry - low)
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            mfe = max(mfe, entry - low)
            mae = max(mae, high - entry)
            hit_stop = high >= stop
            hit_target = low <= target

        # Conservative rule when both levels occur in one OHLC bar:
        # mark AMBIGUOUS instead of assuming which level came first.
        if hit_stop and hit_target:
            return "AMBIGUOUS", ts, bars_after, mfe, mae
        if hit_target:
            return "WIN", ts, bars_after, mfe, mae
        if hit_stop:
            return "LOSS", ts, bars_after, mfe, mae

    return "EXPIRED", future.index[-1], len(future), mfe, mae


def daily_direction_at(daily, ts, current_price):
    try:
        available = daily.loc[daily.index <= ts]
    except Exception:
        available = daily
    if len(available) < 30:
        return "NEUTRAL", None
    macro = v5.daily_spike_context(available, current_price, ts)
    return macro.get("daily_direction", "NEUTRAL"), macro


def main():
    print(f"Downloading {TICKER} {PERIOD} 15m data...")
    intraday = clean_download(yf.download(TICKER, period=PERIOD, interval="15m", auto_adjust=False, progress=False))
    if intraday.empty or len(intraday) < MIN_BARS + 5:
        raise RuntimeError("Insufficient intraday data for backtest.")

    print("Downloading daily and hourly context...")
    daily = clean_download(yf.download(TICKER, period="2y", interval="1d", auto_adjust=False, progress=False))
    hourly = clean_download(yf.download(TICKER, period=PERIOD, interval="1h", auto_adjust=False, progress=False))
    if daily.empty or hourly.empty:
        raise RuntimeError("Missing daily/hourly context data.")

    trades = []
    seen_signals = set()

    # Walk forward one completed 15m bar at a time. The production detector
    # only sees bars through the signal bar, preventing look-ahead in the signal.
    for i in range(MIN_BARS, len(intraday)):
        signal_time = intraday.index[i]
        history = intraday.iloc[: i + 1].copy()
        current_price = float(history["Close"].iloc[-1])

        d = daily.loc[daily.index <= signal_time]
        h = hourly.loc[hourly.index <= signal_time]
        if len(d) < 30 or len(h) < 4:
            continue

        macro = v5.daily_spike_context(d, current_price, signal_time)
        spike = v5.spike_watch_context(history, d, h, signal_time, macro=macro)

        if spike.get("stage") != "SPIKE_CONFIRMED":
            continue
        direction = spike.get("direction", "NONE")
        if direction not in {"UP", "DOWN"}:
            continue

        # Match the production recommendation: counter-trend spikes are not
        # executable setups; they remain informational.
        daily_direction = macro.get("daily_direction", "NEUTRAL")
        if daily_direction in {"UP", "DOWN"} and daily_direction != direction:
            continue

        if signal_time in seen_signals:
            continue
        seen_signals.add(signal_time)

        setup = directional_setup(direction, current_price)
        result = resolve_trade(intraday, signal_time, setup)
        outcome, exit_time, bars_after, mfe, mae = result

        trades.append({
            "signal_time_ct": signal_time.isoformat(),
            "direction": direction,
            "daily_direction": daily_direction,
            "stage": spike.get("stage"),
            "score": spike.get("score"),
            "up_score": spike.get("up_score"),
            "down_score": spike.get("down_score"),
            "signal_price": round(current_price, 4),
            "entry": setup["entry"],
            "stop": setup["stop"],
            "target": setup["target"],
            "risk": setup["risk"],
            "reward": setup["reward"],
            "rr": setup["rr"],
            "outcome": outcome,
            "exit_time_ct": exit_time.isoformat() if exit_time is not None else None,
            "bars_after": bars_after,
            "mfe": round(float(mfe), 4),
            "mae": round(float(mae), 4),
            "volume_ratio": spike.get("volume_ratio"),
            "atr_expansion_ratio": spike.get("atr_expansion_ratio"),
            "breakout_pct": spike.get("breakout_pct"),
            "trend_16bar_pct": spike.get("trend_16bar_pct"),
            "trend_1h_pct": spike.get("trend_1h_pct"),
        })

    df = pd.DataFrame(trades)
    total = len(df)
    wins = int((df["outcome"] == "WIN").sum()) if total else 0
    losses = int((df["outcome"] == "LOSS").sum()) if total else 0
    ambiguous = int((df["outcome"] == "AMBIGUOUS").sum()) if total else 0
    expired = int((df["outcome"] == "EXPIRED").sum()) if total else 0
    resolved = wins + losses

    report = {
        "version": 1,
        "generated_at_ct": datetime.now(CHICAGO_TZ).isoformat(),
        "ticker": TICKER,
        "period": PERIOD,
        "signal_logic_source": "anti_hunt_filter.py / V5 spike_watch_context",
        "lookahead_control": "signal uses bars through current 15m bar only; outcome starts on next bar",
        "horizon_bars": HORIZON_BARS,
        "directional_setup": {
            "UP": "entry=current signal close; stop=entry-5.0; target=entry+32.0",
            "DOWN": "entry=current signal close; stop=entry+5.0; target=entry-32.0",
            "risk_points": SPIKE_STOP_DISTANCE,
            "reward_points": SPIKE_TARGET_DISTANCE,
            "rr": round(SPIKE_TARGET_DISTANCE / SPIKE_STOP_DISTANCE, 2),
        },
        "counter_trend_policy": "excluded from executable backtest",
        "signals_tested": total,
        "wins": wins,
        "losses": losses,
        "ambiguous": ambiguous,
        "expired": expired,
        "resolved": resolved,
        "win_rate_resolved": round(wins / resolved, 4) if resolved else None,
        "loss_rate_resolved": round(losses / resolved, 4) if resolved else None,
        "by_direction": {},
    }

    if total:
        for direction in ["UP", "DOWN"]:
            sub = df[df["direction"] == direction]
            rw = int((sub["outcome"] == "WIN").sum())
            rl = int((sub["outcome"] == "LOSS").sum())
            rr = rw + rl
            report["by_direction"][direction] = {
                "signals": len(sub),
                "wins": rw,
                "losses": rl,
                "ambiguous": int((sub["outcome"] == "AMBIGUOUS").sum()),
                "expired": int((sub["outcome"] == "EXPIRED").sum()),
                "win_rate_resolved": round(rw / rr, 4) if rr else None,
            }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    df.to_csv(OUTPUT_CSV, index=False)

    print(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
