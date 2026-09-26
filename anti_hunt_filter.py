import json
import os
import sys
import time
from datetime import datetime, time as dt_time, date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf


TICKER = "ZW=F"
CHICAGO_TZ = ZoneInfo("America/Chicago")
TICK_SIZE = 0.25

ATR_PERIOD = 16
ATR_STOP_MIN_MULT = 2.0
MAX_DATA_AGE_MIN = 45
MAX_FETCH_ATTEMPTS = 3

WINDOW_OPEN = dt_time(8, 45)
WINDOW_CLOSE = dt_time(12, 30)
ENTRY_CUTOFF = dt_time(11, 30)

# Five bounded profiles. The learner can choose ONLY among these.
PROFILES = {
    "BASE": {"entry_mult": 0.994, "stop_mult": 1.012, "target_mult": 0.960},
    "ENTRY_993": {"entry_mult": 0.993, "stop_mult": 1.012, "target_mult": 0.960},
    "STOP_1010": {"entry_mult": 0.994, "stop_mult": 1.010, "target_mult": 0.960},
    "STOP_1014": {"entry_mult": 0.994, "stop_mult": 1.014, "target_mult": 0.960},
    "TARGET_958": {"entry_mult": 0.994, "stop_mult": 1.012, "target_mult": 0.958},
}

DEFAULT_PROFILE = "BASE"
MIN_LEARNING_SETUPS = 30
LEARNING_CHECKPOINT = 10
MIN_EXPECTED_R_IMPROVEMENT = 0.05
MIN_RECENT_EXPECTED_R_IMPROVEMENT = 0.02

STATE_FILE = "learning_state.json"
SETUP_FILE = "setup.json"
REPORT_FILE = "learning_report.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
HEALTHCHECK_URL = os.environ.get("HEALTHCHECK_URL", "").strip()

HOLIDAYS = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}


def is_manual() -> bool:
    return "--manual" in sys.argv


def ping_healthcheck() -> None:
    if not HEALTHCHECK_URL or is_manual():
        return
    try:
        requests.get(HEALTHCHECK_URL, timeout=10)
    except requests.RequestException as exc:
        print(f"Healthcheck ping failed (non-fatal): {exc}")


def round_tick(price: float) -> float:
    return round(round(price / TICK_SIZE) * TICK_SIZE, 4)


def check_time_window() -> bool:
    now = datetime.now(CHICAGO_TZ)
    return now.weekday() < 5 and WINDOW_OPEN <= now.time() <= WINDOW_CLOSE


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_market_data():
    intraday = daily = None

    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            intraday = _flatten(yf.download(
                TICKER,
                period="2d",
                interval="15m",
                auto_adjust=False,
                progress=False,
            ))
            daily = _flatten(yf.download(
                TICKER,
                period="5d",
                interval="1d",
                auto_adjust=False,
                progress=False,
            ))
        except Exception as exc:
            print(f"Fetch attempt {attempt} raised: {exc}")

        if (
            intraday is not None
            and daily is not None
            and not intraday.empty
            and not daily.empty
        ):
            return intraday, daily

        if attempt < MAX_FETCH_ATTEMPTS:
            time.sleep(10 * attempt)

    raise ValueError("yfinance returned no usable data after all attempts.")


def chicago_index(df: pd.DataFrame):
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(CHICAGO_TZ)


def resolve_session_open(intraday: pd.DataFrame, daily: pd.DataFrame) -> float:
    now = datetime.now(CHICAGO_TZ)
    idx = chicago_index(intraday)
    todays = intraday[idx.date == now.date()]

    if not todays.empty:
        return float(todays["Open"].iloc[0])

    return float(daily["Open"].iloc[-1])


def compute_atr(intraday: pd.DataFrame):
    if len(intraday) < ATR_PERIOD + 1:
        return None

    high = intraday["High"]
    low = intraday["Low"]
    close = intraday["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    return float(tr.rolling(ATR_PERIOD).mean().iloc[-1])


def check_staleness(intraday: pd.DataFrame) -> float:
    idx = chicago_index(intraday)
    return (datetime.now(CHICAGO_TZ) - idx[-1]).total_seconds() / 60.0


def load_state() -> dict:
    default = {
        "version": 3,
        "active_profile": DEFAULT_PROFILE,
        "setups": [],
        "last_learning_update": None,
        "learning_updates": [],
    }

    if not os.path.exists(STATE_FILE):
        return default

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            state = json.load(fh)

        if not isinstance(state, dict):
            return default

        state.setdefault("version", 3)
        state.setdefault("active_profile", DEFAULT_PROFILE)
        state.setdefault("setups", [])
        state.setdefault("last_learning_update", None)
        state.setdefault("learning_updates", [])

        if state["active_profile"] not in PROFILES:
            state["active_profile"] = DEFAULT_PROFILE

        return state

    except (OSError, json.JSONDecodeError) as exc:
        print(f"Learning state unreadable; safe defaults used: {exc}")
        return default


def save_json(path: str, data) -> None:
    tmp = path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    os.replace(tmp, path)


def build_setup(
    daily_open: float,
    current_price: float,
    atr,
    profile_name: str,
) -> dict:
    params = PROFILES[profile_name]

    entry = round_tick(daily_open * params["entry_mult"])
    stop = round_tick(daily_open * params["stop_mult"])
    target = round_tick(daily_open * params["target_mult"])

    risk = round(stop - entry, 4)
    reward = round(entry - target, 4)

    return {
        "ticker": TICKER,
        "profile": profile_name,
        "timestamp_ct": datetime.now(CHICAGO_TZ).isoformat(),
        "valid_until_ct": datetime.combine(
            datetime.now(CHICAGO_TZ).date(),
            WINDOW_CLOSE,
            tzinfo=CHICAGO_TZ,
        ).isoformat(),
        "daily_open": round(daily_open, 4),
        "current_price": round(current_price, 4),
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": risk,
        "reward": reward,
        "rr": round(reward / risk, 2) if risk > 0 else 0.0,
        "atr_15m": round(atr, 4) if atr else None,
        "stop_atr_multiple": round(risk / atr, 2) if atr and atr > 0 else None,
        "vol_warning": bool(atr and risk < ATR_STOP_MIN_MULT * atr),
        "invalidated": current_price > stop,
    }


def completed_bars(intraday: pd.DataFrame):
    idx = chicago_index(intraday)
    now = datetime.now(CHICAGO_TZ)

    mask = [(ts + timedelta(minutes=15)) <= now for ts in idx]

    return intraday.loc[mask].copy(), idx[mask]


def evaluate_geometry(setup: dict, bars: pd.DataFrame, idx) -> dict:
    # Evaluate one profile against future 15m bars.
    # If stop and target are both touched in one 15m bar, the order is
    # unknowable from 15m data, so classify it as AMBIGUOUS rather than guess.
    daily_open = float(setup["daily_open"])
    valid_until = datetime.fromisoformat(setup["valid_until_ct"])
    setup_time = datetime.fromisoformat(setup["timestamp_ct"])

    profile_name = setup.get("profile_for_evaluation", DEFAULT_PROFILE)
    params = PROFILES[profile_name]

    entry = round_tick(daily_open * params["entry_mult"])
    stop = round_tick(daily_open * params["stop_mult"])
    target = round_tick(daily_open * params["target_mult"])

    risk = stop - entry
    reward = entry - target

    entered = False
    entry_time = None

    for i, (_, bar) in enumerate(bars.iterrows()):
        bar_time = idx[i]

        if bar_time <= setup_time:
            continue

        if bar_time > valid_until:
            break

        high = float(bar["High"])
        low = float(bar["Low"])

        if not entered:
            if low <= entry <= high:
                entered = True
                entry_time = bar_time

                if high >= stop and low <= target:
                    return {
                        "outcome": "AMBIGUOUS",
                        "r_multiple": None,
                        "entry_time_ct": entry_time.isoformat(),
                    }

                if high >= stop:
                    return {
                        "outcome": "LOSS",
                        "r_multiple": -1.0,
                        "entry_time_ct": entry_time.isoformat(),
                    }

                if low <= target:
                    return {
                        "outcome": "WIN",
                        "r_multiple": round(reward / risk, 4) if risk > 0 else 0.0,
                        "entry_time_ct": entry_time.isoformat(),
                    }

        else:
            if high >= stop and low <= target:
                return {
                    "outcome": "AMBIGUOUS",
                    "r_multiple": None,
                    "entry_time_ct": entry_time.isoformat(),
                }

            if high >= stop:
                return {
                    "outcome": "LOSS",
                    "r_multiple": -1.0,
                    "entry_time_ct": entry_time.isoformat(),
                }

            if low <= target:
                return {
                    "outcome": "WIN",
                    "r_multiple": round(reward / risk, 4) if risk > 0 else 0.0,
                    "entry_time_ct": entry_time.isoformat(),
                }

    if not entered:
        return {
            "outcome": "NO_ENTRY",
            "r_multiple": 0.0,
            "entry_time_ct": None,
        }

    return {
        "outcome": "EXPIRED_AFTER_ENTRY",
        "r_multiple": 0.0,
        "entry_time_ct": entry_time.isoformat(),
    }


def resolve_previous_setups(state: dict, intraday: pd.DataFrame) -> int:
    """
    Resolve each historical setup against all five profiles.

    This is the learning mechanism: the system can compare profiles using
    the same historical price path without pretending that an untested
    counterfactual trade was actually executed.
    """
    bars, idx = completed_bars(intraday)

    if bars.empty:
        return 0

    changed = 0

    for setup in state["setups"]:
        if setup.get("resolved_profiles"):
            continue

        try:
            valid_until = datetime.fromisoformat(setup["valid_until_ct"])
        except (KeyError, ValueError):
            continue

        if datetime.now(CHICAGO_TZ) < valid_until:
            continue

        results = {}

        for profile_name in PROFILES:
            test_setup = dict(setup)
            test_setup["profile_for_evaluation"] = profile_name
            results[profile_name] = evaluate_geometry(
                test_setup,
                bars,
                idx,
            )

        setup["resolved_profiles"] = results
        setup["resolved_at_ct"] = datetime.now(CHICAGO_TZ).isoformat()

        changed += 1

    return changed


def profile_stats(state: dict) -> dict:
    stats = {}

    for profile_name in PROFILES:
        evaluations = []

        for setup in state["setups"]:
            result = setup.get("resolved_profiles", {}).get(profile_name)

            if not result:
                continue

            evaluations.append(result)

        trades = [
            x for x in evaluations
            if x["outcome"] in {"WIN", "LOSS"}
        ]

        entered = [
            x for x in evaluations
            if x["outcome"] in {
                "WIN",
                "LOSS",
                "AMBIGUOUS",
                "EXPIRED_AFTER_ENTRY",
            }
        ]

        r_values = [
            float(x["r_multiple"])
            for x in evaluations
            if x.get("r_multiple") is not None
        ]

        recent = evaluations[-20:]
        recent_r = [
            float(x["r_multiple"])
            for x in recent
            if x.get("r_multiple") is not None
        ]

        wins = sum(x["outcome"] == "WIN" for x in trades)

        stats[profile_name] = {
            "setups": len(evaluations),
            "trades": len(trades),
            "wins": wins,
            "losses": sum(x["outcome"] == "LOSS" for x in trades),
            "entries": len(entered),
            "no_entry": sum(x["outcome"] == "NO_ENTRY" for x in evaluations),
            "ambiguous": sum(
                x["outcome"] == "AMBIGUOUS" for x in evaluations
            ),
            "expired_after_entry": sum(
                x["outcome"] == "EXPIRED_AFTER_ENTRY"
                for x in evaluations
            ),
            "win_rate": round(wins / len(trades), 4) if trades else None,
            "expected_r": round(
                sum(r_values) / len(evaluations), 4
            ) if evaluations else None,
            "avg_r_per_trade": round(
                sum(r_values) / len(trades), 4
            ) if trades else None,
            "recent_expected_r": round(
                sum(recent_r) / len(recent), 4
            ) if recent else None,
        }

    return stats


def maybe_learn(state: dict) -> dict:
    stats = profile_stats(state)

    report = {
        "generated_at_ct": datetime.now(CHICAGO_TZ).isoformat(),
        "active_profile": state["active_profile"],
        "profiles": stats,
        "learning_applied": False,
        "reason": "Not enough completed setups.",
    }

    completed_count = max(
        (x["setups"] for x in stats.values()),
        default=0,
    )

    if completed_count < MIN_LEARNING_SETUPS:
        return report

    if completed_count % LEARNING_CHECKPOINT != 0:
        report["reason"] = "Not at a learning checkpoint."
        return report

    current_name = state["active_profile"]
    current = stats.get(current_name, {})

    if current.get("recent_expected_r") is None:
        report["reason"] = "Current profile has insufficient recent data."
        return report

    candidates = []

    for name, data in stats.items():
        if data["setups"] < MIN_LEARNING_SETUPS:
            continue

        if data["recent_expected_r"] is None:
            continue

        candidates.append((data["recent_expected_r"], data["expected_r"], name))

    if not candidates:
        report["reason"] = "No profile has enough observations."
        return report

    candidates.sort(reverse=True)

    best_recent, best_overall, best_name = candidates[0]

    if best_name == current_name:
        report["reason"] = "Current profile remains the best observed profile."
        return report

    improvement_recent = best_recent - current["recent_expected_r"]
    improvement_overall = (
        best_overall - (current.get("expected_r") or 0.0)
    )

    if improvement_recent < MIN_RECENT_EXPECTED_R_IMPROVEMENT:
        report["reason"] = "Recent improvement is below the safety threshold."
        return report

    if improvement_overall < MIN_EXPECTED_R_IMPROVEMENT:
        report["reason"] = "Overall improvement is below the safety threshold."
        return report

    previous = state["active_profile"]
    state["active_profile"] = best_name
    state["last_learning_update"] = datetime.now(CHICAGO_TZ).isoformat()

    update = {
        "timestamp_ct": state["last_learning_update"],
        "previous_profile": previous,
        "new_profile": best_name,
        "previous_recent_expected_r": current["recent_expected_r"],
        "new_recent_expected_r": best_recent,
        "previous_expected_r": current.get("expected_r"),
        "new_expected_r": best_overall,
    }

    state["learning_updates"].append(update)

    report.update({
        "learning_applied": True,
        "reason": "The new profile passed both recent and overall improvement thresholds.",
        "previous_profile": previous,
        "new_profile": best_name,
        "improvement_recent_expected_r": round(improvement_recent, 4),
        "improvement_overall_expected_r": round(improvement_overall, 4),
    })

    return report


def send_telegram_alert(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing — alert printed instead.")
        print(text)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        print(f"Telegram delivery failed (non-fatal): {exc}")
        print(text)
        return False


def format_message(setup: dict, report: dict) -> str:
    lines = [
        "⛓ <b>[ZW=F] Anti-Hunt Short Setup</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"📅 {datetime.now(CHICAGO_TZ).strftime('%A %Y-%m-%d %H:%M %Z')}",
        f"🧠 Profile: <b>{setup['profile']}</b>",
        f"🔵 Open: <b>{setup['daily_open']}</b>",
        f"📊 Current: <code>{setup['current_price']}</code>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🔻 ENTRY: <code>{setup['entry']}</code>",
        f"🛑 STOP: <code>{setup['stop']}</code>",
        f"🎯 TARGET: <code>{setup['target']}</code>",
        f"Risk: <code>{setup['risk']}</code> | "
        f"Reward: <code>{setup['reward']}</code> | "
        f"R:R <code>{setup['rr']}</code>",
    ]

    if setup["stop_atr_multiple"] is not None:
        lines.append(
            f"ATR: <code>{setup['atr_15m']}</code> | "
            f"Stop distance: <code>{setup['stop_atr_multiple']}x</code>"
        )

    if setup["vol_warning"]:
        lines.append("⚠️ Stop is inside the 2x ATR noise band.")

    if setup.get("stale_data_min", 0) > MAX_DATA_AGE_MIN:
        lines.append(
            f"⚠️ STALE DATA: {setup['stale_data_min']} min"
        )

    if report.get("learning_applied"):
        lines.append(
            f"🧠 <b>Profile changed: "
            f"{report['previous_profile']} → {report['new_profile']}</b>"
        )

    lines.append("⏳ Valid until 12:30 CT.")
    return "\n".join(lines)


def main() -> int:
    manual = is_manual()
    now_ct = datetime.now(CHICAGO_TZ)

    if not manual:
        ping_healthcheck()

        if now_ct.date() in HOLIDAYS:
            print(f"{now_ct.date()} is a CME holiday. Aborting.")
            return 0

        if not check_time_window():
            print(
                f"[{now_ct:%Y-%m-%d %H:%M %Z}] "
                "Outside the execution window. Aborting."
            )
            return 0

        if now_ct.time() > ENTRY_CUTOFF:
            print("Past the entry cutoff. Aborting.")
            return 0

    if manual:
        print("=" * 50)
        print("MANUAL TEST MODE")
        print("Normal time/holiday restrictions are bypassed.")
        print("Real ZW=F market data will still be fetched.")
        print("=" * 50)

    print("Fetching ZW=F data...")

    try:
        intraday, daily = fetch_market_data()
    except ValueError as exc:
        print(f"Data error: {exc}")
        return 1

    state = load_state()

    resolved = resolve_previous_setups(state, intraday)

    if resolved:
        print(f"Resolved {resolved} previous setup(s).")

    report = maybe_learn(state)
    save_json(REPORT_FILE, report)

    age = check_staleness(intraday)

    if age > MAX_DATA_AGE_MIN:
        print(
            f"WARNING: last 15m bar is {age:.0f} min old — "
            "alert will be flagged as stale."
        )

    daily_open = resolve_session_open(intraday, daily)
    current_price = float(intraday["Close"].iloc[-1])
    atr = compute_atr(intraday)

    profile_name = state["active_profile"]

    setup = build_setup(
        daily_open,
        current_price,
        atr,
        profile_name,
    )

    setup["manual_mode"] = manual
    setup["stale_data_min"] = round(age, 1)

    print(json.dumps(setup, indent=2))

    if setup["invalidated"]:
        print("Setup invalidated; no alert.")
        save_json(STATE_FILE, state)
        save_json(SETUP_FILE, setup)
        return 0

    # Manual runs do not create a new learning sample.
    if not manual:
        state["setups"].append(setup)

    save_json(SETUP_FILE, setup)
    save_json(STATE_FILE, state)

    # Keep existing manual-test behavior: send the alert if credentials exist.
    send_telegram_alert(format_message(setup, report))

    print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
