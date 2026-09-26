import json
import os
import importlib.util
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


def load_v5_module():
    requested = os.environ.get("V5_MODULE_FILE", "").strip()
    candidates = [Path(requested)] if requested else []
    candidates.extend([
        Path("anti_hunt_filter.py"),
        Path("anti_hunt_filter_v5.py"),
        Path("anti_hunt_filter_v5_SPIKE_WATCH_SETUP_RECOMMENDATION_PRICES.py"),
        Path("anti_hunt_filter_v5_SPIKE_WATCH_SETUP_RECOMMENDATION_FIXED2.py"),
        Path("anti_hunt_filter_v5_SPIKE_WATCH_SETUP_RECOMMENDATION_FIXED.py"),
    ])
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("v5_research", path.resolve())
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                required = ["chicago_index", "round_tick", "daily_spike_context", "spike_watch_context"]
                missing = [x for x in required if not hasattr(module, x)]
                if not missing:
                    print(f"Using V5 spike detector: {path}")
                    return module
                print(f"Skipping {path}: missing {missing}")
    raise RuntimeError("No compatible V5 spike detector found.")


v5 = load_v5_module()
CHICAGO_TZ = ZoneInfo("America/Chicago")
TICKER = os.environ.get("TICKER", "ZW=F")
PERIOD = os.environ.get("BACKTEST_PERIOD", "60d")
HORIZON_BARS = int(os.environ.get("SPIKE_HORIZON_BARS", "32"))
MIN_BARS = int(os.environ.get("SPIKE_MIN_BARS", "35"))
EPISODE_GAP_BARS = int(os.environ.get("SPIKE_EPISODE_GAP_BARS", "4"))
TARGETS = [float(x) for x in os.environ.get("SPIKE_TARGETS", "5,8,10,12,16,20,24,28,32").split(",")]
STOPS = [float(x) for x in os.environ.get("SPIKE_STOPS", "3,4,5,6,8,10").split(",")]
DELAY_BARS = [int(x) for x in os.environ.get("SPIKE_DELAY_BARS", "0,1,2,4").split(",")]
OUT_JSON = os.environ.get("RESEARCH_JSON", "backtest_v5_spike_research_report.json")
OUT_TRADES = os.environ.get("RESEARCH_CSV", "backtest_v5_spike_research_trades.csv")


def clean_download(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df.index = v5.chicago_index(df)
    return df.sort_index()


def direction_levels(direction, entry, stop_dist, target_dist):
    entry = float(v5.round_tick(entry))
    if direction == "UP":
        stop = float(v5.round_tick(entry - stop_dist))
        target = float(v5.round_tick(entry + target_dist))
    else:
        stop = float(v5.round_tick(entry + stop_dist))
        target = float(v5.round_tick(entry - target_dist))
    return entry, stop, target


def resolve(df, start_pos, direction, entry, stop_dist, target_dist, horizon):
    _, stop, target = direction_levels(direction, entry, stop_dist, target_dist)
    future = df.iloc[start_pos + 1:start_pos + 1 + horizon]
    mfe = mae = 0.0
    for n, (ts, row) in enumerate(future.iterrows(), 1):
        hi, lo = float(row.High), float(row.Low)
        if direction == "UP":
            mfe = max(mfe, hi - entry); mae = max(mae, entry - lo)
            hit_s, hit_t = lo <= stop, hi >= target
        else:
            mfe = max(mfe, entry - lo); mae = max(mae, hi - entry)
            hit_s, hit_t = hi >= stop, lo <= target
        if hit_s and hit_t:
            return "AMBIGUOUS", ts, n, mfe, mae
        if hit_t:
            return "WIN", ts, n, mfe, mae
        if hit_s:
            return "LOSS", ts, n, mfe, mae
    if len(future):
        return "EXPIRED", future.index[-1], len(future), mfe, mae
    return "EXPIRED", None, None, mfe, mae


def feature_bucket(row):
    vol = float(row.volume_ratio or 0)
    br = abs(float(row.breakout_pct or 0))
    atr = float(row.atr_expansion_ratio or 1)
    if vol >= 4 and atr >= 1.15 and br >= 0.5:
        return "FRESH_IMPULSE"
    if vol < 1 and br >= 1.0:
        return "LATE_EXTENSION_LOW_VOLUME"
    if vol < 1:
        return "LOW_VOLUME"
    if vol >= 2:
        return "HIGH_VOLUME"
    return "MID_VOLUME"


def summarize(df, group_cols, target=16, stop=5):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        rec = dict(zip(group_cols, keys))
        w = int((g["outcome"] == "WIN").sum())
        l = int((g["outcome"] == "LOSS").sum())
        r = w + l
        rec.update({"signals": len(g), "wins": w, "losses": l, "expired": int((g.outcome == "EXPIRED").sum()),
                    "ambiguous": int((g.outcome == "AMBIGUOUS").sum()),
                    "resolved": r, "win_rate_resolved": round(w / r, 4) if r else None,
                    "avg_mfe": round(float(g.mfe.mean()), 3), "avg_mae": round(float(g.mae.mean()), 3)})
        rows.append(rec)
    return rows


def main():
    print(f"Downloading {TICKER} {PERIOD} 15m data...")
    intraday = clean_download(yf.download(TICKER, period=PERIOD, interval="15m", auto_adjust=False, progress=False))
    daily = clean_download(yf.download(TICKER, period="2y", interval="1d", auto_adjust=False, progress=False))
    hourly = clean_download(yf.download(TICKER, period=PERIOD, interval="1h", auto_adjust=False, progress=False))
    if len(intraday) < MIN_BARS + 5 or daily.empty or hourly.empty:
        raise RuntimeError("Insufficient data for spike research.")

    signals = []
    for i in range(MIN_BARS, len(intraday)):
        ts = intraday.index[i]
        d = daily.loc[daily.index <= ts]
        h = hourly.loc[hourly.index <= ts]
        if len(d) < 30 or len(h) < 4:
            continue
        price = float(intraday.Close.iloc[i])
        macro = v5.daily_spike_context(d, price, ts)
        spike = v5.spike_watch_context(intraday.iloc[:i + 1].copy(), d, h, ts, macro=macro)
        if spike.get("stage") != "SPIKE_CONFIRMED":
            continue
        direction = spike.get("direction")
        daily_direction = macro.get("daily_direction", "NEUTRAL")
        if direction not in {"UP", "DOWN"}:
            continue
        if daily_direction in {"UP", "DOWN"} and daily_direction != direction:
            continue
        signals.append({
            "signal_i": i, "signal_time_ct": ts.isoformat(), "direction": direction,
            "daily_direction": daily_direction, "score": spike.get("score"),
            "up_score": spike.get("up_score"), "down_score": spike.get("down_score"),
            "signal_price": price, "volume_ratio": spike.get("volume_ratio", 1),
            "atr_expansion_ratio": spike.get("atr_expansion_ratio", 1),
            "breakout_pct": spike.get("breakout_pct", 0),
            "trend_16bar_pct": spike.get("trend_16bar_pct", 0),
            "trend_1h_pct": spike.get("trend_1h_pct", 0),
        })

    base = pd.DataFrame(signals)
    if base.empty:
        raise RuntimeError("No confirmed directional spike signals found.")

    # Episode clustering: repeated 15m confirmations close together are one episode.
    base["episode_id"] = 0
    ep = 0
    for j in range(1, len(base)):
        prev_i = int(base.iloc[j - 1].signal_i)
        cur_i = int(base.iloc[j].signal_i)
        if cur_i - prev_i > EPISODE_GAP_BARS:
            ep += 1
        base.iloc[j, base.columns.get_loc("episode_id")] = ep
    base["episode_position"] = base.groupby("episode_id").cumcount() + 1
    base["episode_size"] = base.groupby("episode_id")["episode_id"].transform("size")
    base["feature_bucket"] = base.apply(feature_bucket, axis=1)

    # Full target ladder and stop ladder on the exact same confirmed signals.
    results = []
    for _, s in base.iterrows():
        for stop in STOPS:
            for target in TARGETS:
                out, exit_ts, bars, mfe, mae = resolve(intraday, int(s.signal_i), s.direction, s.signal_price, stop, target, HORIZON_BARS)
                results.append({**s.to_dict(), "experiment": "target_stop_matrix", "stop_dist": stop,
                                "target_dist": target, "rr": round(target / stop, 3), "outcome": out,
                                "exit_time_ct": exit_ts.isoformat() if exit_ts is not None else None,
                                "bars_after": bars, "mfe": mfe, "mae": mae})

    matrix = pd.DataFrame(results)

    # Delayed-entry experiment: entry at close of +1/+2/+4 bars, then same 5/16 geometry.
    delay_rows = []
    for _, s in base.iterrows():
        for delay in DELAY_BARS:
            entry_i = int(s.signal_i) + delay
            if entry_i >= len(intraday):
                continue
            entry = float(intraday.Close.iloc[entry_i])
            out, exit_ts, bars, mfe, mae = resolve(intraday, entry_i, s.direction, entry, 5.0, 16.0, HORIZON_BARS)
            delay_rows.append({**s.to_dict(), "experiment": "delayed_entry", "delay_bars": delay,
                               "entry_price": entry, "stop_dist": 5.0, "target_dist": 16.0,
                               "rr": 3.2, "outcome": out, "exit_time_ct": exit_ts.isoformat() if exit_ts is not None else None,
                               "bars_after": bars, "mfe": mfe, "mae": mae})
    delayed = pd.DataFrame(delay_rows)

    # First-confirmation-only episode test: one executable signal per episode.
    first = base[base.episode_position == 1].copy()
    first_rows = []
    for _, s in first.iterrows():
        out, exit_ts, bars, mfe, mae = resolve(intraday, int(s.signal_i), s.direction, s.signal_price, 5.0, 16.0, HORIZON_BARS)
        first_rows.append({**s.to_dict(), "experiment": "first_signal_per_episode", "stop_dist": 5.0,
                           "target_dist": 16.0, "rr": 3.2, "outcome": out, "exit_time_ct": exit_ts.isoformat() if exit_ts is not None else None,
                           "bars_after": bars, "mfe": mfe, "mae": mae})
    first_df = pd.DataFrame(first_rows)

    report = {
        "version": 2,
        "generated_at_ct": datetime.now(CHICAGO_TZ).isoformat(),
        "ticker": TICKER,
        "period": PERIOD,
        "signal_count": int(len(base)),
        "episode_count": int(base.episode_id.nunique()),
        "episode_gap_bars": EPISODE_GAP_BARS,
        "horizon_bars": HORIZON_BARS,
        "counter_trend_policy": "excluded",
        "purpose": "Research only. Does not modify the live bounded learner or production signal.",
        "target_stop_matrix": {
            "rows": int(len(matrix)),
            "summary": summarize(matrix, ["stop_dist", "target_dist", "rr"]),
        },
        "target_16_by_feature_bucket": summarize(matrix[matrix.target_dist == 16], ["feature_bucket"]),
        "target_16_by_episode_position": summarize(matrix[matrix.target_dist == 16], ["episode_position"]),
        "target_16_by_direction": summarize(matrix[matrix.target_dist == 16], ["direction"]),
        "delayed_entry_5_16": summarize(delayed, ["delay_bars"]),
        "first_signal_per_episode_5_16": summarize(first_df, ["direction"]),
        "episodes": {
            "mean_signals_per_episode": round(float(base.groupby("episode_id").size().mean()), 3),
            "multi_signal_episodes": int((base.groupby("episode_id").size() > 1).sum()),
            "single_signal_episodes": int((base.groupby("episode_id").size() == 1).sum()),
        },
        "feature_bucket_definition": {
            "FRESH_IMPULSE": "volume_ratio >= 4, ATR expansion >= 1.15, abs(breakout_pct) >= 0.5",
            "LATE_EXTENSION_LOW_VOLUME": "volume_ratio < 1 and abs(breakout_pct) >= 1.0",
            "LOW_VOLUME": "volume_ratio < 1",
            "HIGH_VOLUME": "volume_ratio >= 2 outside fresh-impulse rule",
            "MID_VOLUME": "remaining signals",
        },
        "interpretation_guardrails": [
            "Target/stop results use next-bar OHLC only and mark same-bar stop+target as AMBIGUOUS.",
            "Episode clustering is a research construct; it is not a production rule yet.",
            "Delayed entry is not a pullback/retest model; it only tests waiting N bars.",
            "Feature buckets are descriptive, not optimized thresholds for deployment.",
        ],
    }

    matrix.to_csv(OUT_TRADES, index=False)
    report["delayed_entry_rows"] = int(len(delayed))
    report["first_episode_rows"] = int(len(first_df))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    delayed.to_csv("backtest_v5_spike_delayed_entry.csv", index=False)
    first_df.to_csv("backtest_v5_spike_first_episode.csv", index=False)
    base.to_csv("backtest_v5_spike_signal_episodes.csv", index=False)

    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TRADES}")
    print("Wrote backtest_v5_spike_delayed_entry.csv")
    print("Wrote backtest_v5_spike_first_episode.csv")
    print("Wrote backtest_v5_spike_signal_episodes.csv")


if __name__ == "__main__":
    main()
