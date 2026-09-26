import json
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

import anti_hunt_filter as ahf

# ============================================================
# Backfills anti_hunt_filter.py's learning_state.json with
# historical setups so the shadow ML model doesn't have to wait
# weeks to accumulate ML_MIN_SAMPLES=60 live.
#
# Yahoo/yfinance caps 15m intraday data at ~60 days — that's a
# hard external limit, not a choice. This is NOT a 2-year
# backtest like backtest.py; every setup built here is tagged
# "source": "backtest" and is deliberately excluded from the
# bounded profile-switch decision in maybe_learn() (see there).
# It only accelerates the ML shadow model's bootstrap.
# ============================================================

RESULTS_FILE = "anti_hunt_backtest_results.json"


def fetch_backtest_data():
    intraday = ahf._flatten(yf.download(
        ahf.TICKER, period="60d", interval="15m", auto_adjust=False, progress=False
    ))
    daily = ahf._flatten(yf.download(
        ahf.TICKER, period="90d", interval="1d", auto_adjust=False, progress=False
    ))
    hourly = ahf._flatten(yf.download(
        ahf.TICKER, period="60d", interval="60m", auto_adjust=False, progress=False
    ))
    if intraday.empty or daily.empty or hourly.empty:
        raise ValueError("Yahoo returned empty backtest data.")
    return intraday, daily, hourly


def build_backtest_setups(intraday, daily, hourly):
    idx = ahf.chicago_index(intraday)
    df = intraday.copy()
    df.index = idx
    df = df.sort_index()

    last_bar_ts = df.index.max()
    trading_days = sorted({ts.date() for ts in df.index})

    setups = []
    for day in trading_days:
        if day in ahf.HOLIDAYS or day.weekday() >= 5:
            continue

        signal_dt = datetime.combine(day, ahf.WINDOW_OPEN, tzinfo=ahf.CHICAGO_TZ)
        valid_until = datetime.combine(day, ahf.WINDOW_CLOSE, tzinfo=ahf.CHICAGO_TZ)
        if valid_until > last_bar_ts:
            continue  # not enough future data in the 60d window to fully resolve this one

        past_intraday = df.loc[df.index <= signal_dt]
        if past_intraday.empty:
            continue
        today_bars = past_intraday.loc[[ts.date() == day for ts in past_intraday.index]]
        if today_bars.empty:
            continue

        # NOTE: intentionally not calling ahf.resolve_session_open() here — it
        # uses datetime.now(CHICAGO_TZ) internally to find "today's" bars, which
        # is correct for live runs but always misses when replaying a historical
        # day, silently falling back to the same single stale daily open for
        # every day. today_bars's own first Open is this day's real session open.
        try:
            daily_open = float(today_bars["Open"].iloc[0])
            current_price = float(today_bars["Close"].iloc[-1])
            atr = ahf.compute_atr(past_intraday)
        except (KeyError, IndexError, ValueError):
            continue
        if atr is None:
            continue

        setup = ahf.build_setup(daily_open, current_price, atr, ahf.DEFAULT_PROFILE)
        setup["timestamp_ct"] = signal_dt.isoformat()
        setup["valid_until_ct"] = valid_until.isoformat()
        setup["manual_mode"] = False
        setup["stale_data_min"] = 0.0
        setup["source"] = "backtest"

        try:
            context = ahf.market_context(past_intraday, daily, hourly, signal_dt)
        except (KeyError, IndexError, ValueError):
            continue
        setup["market_context"] = context
        setup.update(context)

        if setup["invalidated"]:
            continue  # matches live: invalidated setups are never appended to state["setups"]

        setups.append(setup)

    return setups, df


def resolve_backtest_setups(setups, df):
    resolved = []
    for setup in setups:
        evaluations = {
            profile_name: ahf.evaluate_geometry(setup, df, df.index, profile_name)
            for profile_name in ahf.PROFILES
        }
        setup["resolved_profiles"] = evaluations
        actual = evaluations.get(setup["profile"])
        if actual:
            setup["actual_outcome"] = actual["outcome"]
            setup["actual_r_multiple"] = actual.get("r_multiple")
            setup["actual_entry_time_ct"] = actual.get("entry_time_ct")
            setup["actual_mfe_r"] = actual.get("mfe_r")
            setup["actual_mae_r"] = actual.get("mae_r")
            setup["actual_bars_after_entry"] = actual.get("bars_after_entry")
            setup["outcome_version"] = ahf.OUTCOME_VERSION
        resolved.append(setup)
    return resolved


def main():
    print("Fetching ZW=F backtest data (Yahoo 15m cap: last ~60 days)...")
    intraday, daily, hourly = fetch_backtest_data()

    setups, df = build_backtest_setups(intraday, daily, hourly)
    print(f"Built {len(setups)} candidate backtest setups.")

    resolved = resolve_backtest_setups(setups, df)
    print(f"Resolved {len(resolved)} backtest setups.")

    state = ahf.load_state()
    existing_ts = {
        s.get("timestamp_ct") for s in state.get("setups", []) if s.get("source") == "backtest"
    }
    new_setups = [s for s in resolved if s["timestamp_ct"] not in existing_ts]
    state["setups"].extend(new_setups)
    ahf.save_json(ahf.STATE_FILE, state)
    print(
        f"Appended {len(new_setups)} new backtest setups to {ahf.STATE_FILE} "
        f"(skipped {len(resolved) - len(new_setups)} already present)."
    )

    summary = {
        "generated_at_ct": datetime.now(ahf.CHICAGO_TZ).isoformat(),
        "note": "Backtest limited to ~60 real trading days (Yahoo 15m data cap). "
                "Excluded from the live bounded profile-switch decision; only "
                "feeds the shadow ML model's training set.",
        "total_backtest_setups": len(resolved),
        "newly_added": len(new_setups),
        "profile_stats_backtest_only": ahf.profile_stats({"setups": resolved}),
    }
    ahf.save_json(RESULTS_FILE, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
