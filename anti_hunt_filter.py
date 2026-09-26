import json
import os
import sys
import time
from datetime import datetime, time as dt_time, date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# V4: bounded strategy + shadow ML
# ============================================================
# The ML model is intentionally SHADOW-ONLY in V4.
# It predicts the probability that the actual generated setup
# will reach TARGET before STOP. It does NOT change the trade.
#
# After enough out-of-sample observations, the ML layer can be
# promoted into the decision path by changing ML_SHADOW_ONLY.
# ============================================================

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

PROFILES = {
    "BASE": {"entry_mult": 0.994, "stop_mult": 1.012, "target_mult": 0.960},
    "ENTRY_993": {"entry_mult": 0.993, "stop_mult": 1.012, "target_mult": 0.960},
    "STOP_1010": {"entry_mult": 0.994, "stop_mult": 1.010, "target_mult": 0.960},
    "STOP_1014": {"entry_mult": 0.994, "stop_mult": 1.014, "target_mult": 0.960},
    "TARGET_958": {"entry_mult": 0.994, "stop_mult": 1.012, "target_mult": 0.958},
}

DEFAULT_PROFILE = "BASE"

# Existing bounded learner.
MIN_LEARNING_SETUPS = 30
LEARNING_CHECKPOINT = 10
MIN_EXPECTED_R_IMPROVEMENT = 0.05
MIN_RECENT_EXPECTED_R_IMPROVEMENT = 0.02

# ML guardrails.
ML_SHADOW_ONLY = True
ML_MIN_SAMPLES = 30
ML_MIN_CLASS_COUNT = 5
ML_CONFIDENCE_THRESHOLD = 0.60
ML_TEST_WINDOW = 10

STATE_FILE = "learning_state.json"
SETUP_FILE = "setup.json"
REPORT_FILE = "learning_report.json"
ML_REPORT_FILE = "ml_report.json"

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

ML_FEATURES = [
    "open_distance_pct",
    "atr_pct",
    "stop_atr_multiple",
    "rr",
    "hour_decimal",
    "stale_data_min",
    "vol_warning",
    "current_vs_open_pct",
]


def is_manual():
    return "--manual" in sys.argv


def is_resolve():
    return "--resolve" in sys.argv


def ping_healthcheck():
    if not HEALTHCHECK_URL or is_manual():
        return
    try:
        requests.get(HEALTHCHECK_URL, timeout=10)
    except requests.RequestException as exc:
        print(f"Healthcheck ping failed (non-fatal): {exc}")


def round_tick(price):
    return round(round(float(price) / TICK_SIZE) * TICK_SIZE, 4)


def chicago_index(df):
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(CHICAGO_TZ)


def check_time_window():
    now = datetime.now(CHICAGO_TZ)
    return now.weekday() < 5 and WINDOW_OPEN <= now.time() <= WINDOW_CLOSE


def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_market_data():
    last_error = None
    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            intraday = _flatten(yf.download(
                TICKER, period="2d", interval="15m",
                auto_adjust=False, progress=False
            ))
            daily = _flatten(yf.download(
                TICKER, period="5d", interval="1d",
                auto_adjust=False, progress=False
            ))
            if not intraday.empty and not daily.empty:
                return intraday, daily
            last_error = "Yahoo returned empty data."
        except Exception as exc:
            last_error = str(exc)
            print(f"Fetch attempt {attempt} failed: {exc}", file=sys.stderr)
        if attempt < MAX_FETCH_ATTEMPTS:
            time.sleep(10 * attempt)
    raise ValueError(f"Could not fetch usable ZW=F data: {last_error}")


def resolve_session_open(intraday, daily):
    now_ct = datetime.now(CHICAGO_TZ)
    idx = chicago_index(intraday)
    mask = [ts.date() == now_ct.date() for ts in idx]
    todays = intraday.loc[mask]
    if not todays.empty:
        return float(todays["Open"].iloc[0])
    return float(daily["Open"].iloc[-1])


def compute_atr(intraday):
    high = intraday["High"].astype(float)
    low = intraday["Low"].astype(float)
    close = intraday["Close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    value = tr.rolling(ATR_PERIOD).mean().iloc[-1]
    return float(value) if pd.notna(value) else None


def check_staleness(intraday):
    idx = chicago_index(intraday)
    return max(0.0, (datetime.now(CHICAGO_TZ) - idx[-1]).total_seconds() / 60.0)


def load_state():
    default = {
        "version": 4,
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
        state.setdefault("version", 4)
        state.setdefault("active_profile", DEFAULT_PROFILE)
        state.setdefault("setups", [])
        state.setdefault("last_learning_update", None)
        state.setdefault("learning_updates", [])
        if state["active_profile"] not in PROFILES:
            state["active_profile"] = DEFAULT_PROFILE
        return state
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Learning state unreadable; using safe defaults: {exc}")
        return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def build_setup(daily_open, current_price, atr, profile_name):
    params = PROFILES[profile_name]
    entry = round_tick(daily_open * params["entry_mult"])
    stop = round_tick(daily_open * params["stop_mult"])
    target = round_tick(daily_open * params["target_mult"])
    risk = round(stop - entry, 4)
    reward = round(entry - target, 4)
    now = datetime.now(CHICAGO_TZ)

    return {
        "ticker": TICKER,
        "profile": profile_name,
        "timestamp_ct": now.isoformat(),
        "valid_until_ct": datetime.combine(
            now.date(), WINDOW_CLOSE, tzinfo=CHICAGO_TZ
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


def feature_vector(setup):
    ts = datetime.fromisoformat(setup["timestamp_ct"])
    daily_open = float(setup["daily_open"])
    current = float(setup["current_price"])
    atr = float(setup["atr_15m"]) if setup.get("atr_15m") else 0.0
    return [
        (float(setup["entry"]) - daily_open) / daily_open * 100.0,
        (atr / daily_open * 100.0) if daily_open else 0.0,
        float(setup["stop_atr_multiple"] or 0.0),
        float(setup["rr"]),
        ts.hour + ts.minute / 60.0,
        float(setup.get("stale_data_min", 0.0)),
        1.0 if setup.get("vol_warning") else 0.0,
        (current - daily_open) / daily_open * 100.0 if daily_open else 0.0,
    ]


def historical_ml_rows(state):
    rows = []
    for setup in state.get("setups", []):
        outcome = setup.get("actual_outcome")
        if outcome not in {"WIN", "LOSS"}:
            continue
        try:
            rows.append((feature_vector(setup), 1 if outcome == "WIN" else 0))
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def train_ml_model(state):
    rows = historical_ml_rows(state)
    if len(rows) < ML_MIN_SAMPLES:
        return None, {
            "trained": False,
            "reason": f"Need at least {ML_MIN_SAMPLES} resolved WIN/LOSS samples.",
            "samples": len(rows),
        }

    y = [label for _, label in rows]
    if len(set(y)) < 2 or min(y.count(0), y.count(1)) < ML_MIN_CLASS_COUNT:
        return None, {
            "trained": False,
            "reason": "Both WIN and LOSS classes need enough samples.",
            "samples": len(rows),
            "wins": y.count(1),
            "losses": y.count(0),
        }

    X = [features for features, _ in rows]

    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    model.fit(X, y)

    # Time-ordered shadow evaluation: train on older rows, test on the newest rows.
    test_n = min(ML_TEST_WINDOW, max(5, len(rows) // 5))
    if len(rows) > test_n + 10:
        split = len(rows) - test_n
        test_model = Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        test_model.fit(X[:split], y[:split])
        probs = test_model.predict_proba(X[split:])[:, 1]
        preds = (probs >= ML_CONFIDENCE_THRESHOLD).astype(int)
        actual = y[split:]
        directional_accuracy = sum(int(p == a) for p, a in zip(preds, actual)) / len(actual)
        high_conf = [i for i, p in enumerate(probs) if p >= ML_CONFIDENCE_THRESHOLD or p <= 1 - ML_CONFIDENCE_THRESHOLD]
        high_conf_accuracy = None
        if high_conf:
            high_conf_accuracy = sum(
                int((probs[i] >= ML_CONFIDENCE_THRESHOLD) == bool(actual[i]))
                for i in high_conf
            ) / len(high_conf)
    else:
        directional_accuracy = None
        high_conf_accuracy = None

    report = {
        "trained": True,
        "samples": len(rows),
        "wins": y.count(1),
        "losses": y.count(0),
        "test_window": test_n,
        "time_ordered_directional_accuracy": round(directional_accuracy, 4) if directional_accuracy is not None else None,
        "high_confidence_accuracy": round(high_conf_accuracy, 4) if high_conf_accuracy is not None else None,
        "shadow_only": ML_SHADOW_ONLY,
    }
    return model, report


def add_ml_prediction(state, setup):
    model, report = train_ml_model(state)
    setup["ml_shadow"] = {
        "enabled": True,
        "trained": bool(model),
        "probability_win": None,
        "confidence_threshold": ML_CONFIDENCE_THRESHOLD,
        "decision": "INSUFFICIENT_DATA",
        "model_samples": report.get("samples", 0),
    }
    if model is not None:
        probability = float(model.predict_proba([feature_vector(setup)])[0][1])
        decision = "WATCH" if probability >= ML_CONFIDENCE_THRESHOLD else "LOW_CONFIDENCE"
        setup["ml_shadow"].update({
            "probability_win": round(probability, 4),
            "decision": decision,
        })
    return report


def completed_bars(intraday):
    idx = chicago_index(intraday)
    now = datetime.now(CHICAGO_TZ)
    mask = [(ts + timedelta(minutes=15)) <= now for ts in idx]
    return intraday.loc[mask].copy(), idx[mask]


def evaluate_geometry(setup, bars, idx, profile_name):
    daily_open = float(setup["daily_open"])
    valid_until = datetime.fromisoformat(setup["valid_until_ct"])
    setup_time = datetime.fromisoformat(setup["timestamp_ct"])
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

        if not entered and low <= entry <= high:
            entered = True
            entry_time = bar_time

            if high >= stop and low <= target:
                return {"outcome": "AMBIGUOUS", "r_multiple": None, "entry_time_ct": entry_time.isoformat()}
            if high >= stop:
                return {"outcome": "LOSS", "r_multiple": -1.0, "entry_time_ct": entry_time.isoformat()}
            if low <= target:
                return {"outcome": "WIN", "r_multiple": round(reward / risk, 4), "entry_time_ct": entry_time.isoformat()}

        elif entered:
            if high >= stop and low <= target:
                return {"outcome": "AMBIGUOUS", "r_multiple": None, "entry_time_ct": entry_time.isoformat()}
            if high >= stop:
                return {"outcome": "LOSS", "r_multiple": -1.0, "entry_time_ct": entry_time.isoformat()}
            if low <= target:
                return {"outcome": "WIN", "r_multiple": round(reward / risk, 4), "entry_time_ct": entry_time.isoformat()}

    if not entered:
        return {"outcome": "NO_ENTRY", "r_multiple": 0.0, "entry_time_ct": None}
    return {"outcome": "EXPIRED_AFTER_ENTRY", "r_multiple": 0.0, "entry_time_ct": entry_time.isoformat()}


def resolve_previous_setups(state, intraday):
    bars, idx = completed_bars(intraday)
    if bars.empty:
        return 0

    changed = 0
    for setup in state.get("setups", []):
        if setup.get("resolved_profiles"):
            continue

        try:
            valid_until = datetime.fromisoformat(setup["valid_until_ct"])
            if datetime.now(CHICAGO_TZ) <= valid_until:
                continue
        except (KeyError, ValueError):
            continue

        evaluations = {}
        for profile_name in PROFILES:
            result = evaluate_geometry(setup, bars, idx, profile_name)
            evaluations[profile_name] = result

        setup["resolved_profiles"] = evaluations

        actual_profile = setup.get("profile", DEFAULT_PROFILE)
        actual = evaluations.get(actual_profile)
        if actual:
            setup["actual_outcome"] = actual["outcome"]
            setup["actual_r_multiple"] = actual.get("r_multiple")
            setup["actual_entry_time_ct"] = actual.get("entry_time_ct")

        changed += 1

    return changed


def profile_stats(state):
    stats = {}
    for profile_name in PROFILES:
        evaluations = []
        for setup in state.get("setups", []):
            result = (setup.get("resolved_profiles") or {}).get(profile_name)
            if result:
                evaluations.append(result)

        trades = [x for x in evaluations if x["outcome"] in {"WIN", "LOSS"}]
        entered = [x for x in evaluations if x["outcome"] in {"WIN", "LOSS", "AMBIGUOUS", "EXPIRED_AFTER_ENTRY"}]
        r_values = [float(x["r_multiple"]) for x in evaluations if x.get("r_multiple") is not None]
        recent = evaluations[-20:]
        recent_r = [float(x["r_multiple"]) for x in recent if x.get("r_multiple") is not None]
        wins = sum(x["outcome"] == "WIN" for x in trades)

        stats[profile_name] = {
            "setups": len(evaluations),
            "trades": len(trades),
            "wins": wins,
            "losses": sum(x["outcome"] == "LOSS" for x in trades),
            "entries": len(entered),
            "no_entry": sum(x["outcome"] == "NO_ENTRY" for x in evaluations),
            "ambiguous": sum(x["outcome"] == "AMBIGUOUS" for x in evaluations),
            "expired_after_entry": sum(x["outcome"] == "EXPIRED_AFTER_ENTRY" for x in evaluations),
            "win_rate": round(wins / len(trades), 4) if trades else None,
            "expected_r": round(sum(r_values) / len(evaluations), 4) if evaluations else None,
            "avg_r_per_trade": round(sum(r_values) / len(trades), 4) if trades else None,
            "recent_expected_r": round(sum(recent_r) / len(recent), 4) if recent else None,
        }
    return stats


def maybe_learn(state):
    stats = profile_stats(state)
    report = {
        "generated_at_ct": datetime.now(CHICAGO_TZ).isoformat(),
        "active_profile": state["active_profile"],
        "profiles": stats,
        "learning_applied": False,
        "reason": "Not enough completed setups.",
    }

    completed_count = max((x["setups"] for x in stats.values()), default=0)
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
        if data["setups"] >= MIN_LEARNING_SETUPS and data["recent_expected_r"] is not None:
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
    improvement_overall = best_overall - (current.get("expected_r") or 0.0)

    if improvement_recent < MIN_RECENT_EXPECTED_R_IMPROVEMENT:
        report["reason"] = "Recent improvement is below the safety threshold."
        return report
    if improvement_overall < MIN_EXPECTED_R_IMPROVEMENT:
        report["reason"] = "Overall improvement is below the safety threshold."
        return report

    previous = state["active_profile"]
    state["active_profile"] = best_name
    state["last_learning_update"] = datetime.now(CHICAGO_TZ).isoformat()
    state["learning_updates"].append({
        "timestamp_ct": state["last_learning_update"],
        "previous_profile": previous,
        "new_profile": best_name,
        "previous_recent_expected_r": current["recent_expected_r"],
        "new_recent_expected_r": best_recent,
        "previous_expected_r": current.get("expected_r"),
        "new_expected_r": best_overall,
    })

    report.update({
        "learning_applied": True,
        "reason": "New bounded profile passed recent and overall improvement thresholds.",
        "previous_profile": previous,
        "new_profile": best_name,
        "improvement_recent_expected_r": round(improvement_recent, 4),
        "improvement_overall_expected_r": round(improvement_overall, 4),
    })
    return report


def ml_shadow_report(state):
    model, report = train_ml_model(state)
    report["generated_at_ct"] = datetime.now(CHICAGO_TZ).isoformat()
    report["model_type"] = "LogisticRegression + StandardScaler"
    report["features"] = ML_FEATURES
    report["confidence_threshold"] = ML_CONFIDENCE_THRESHOLD
    report["shadow_only"] = ML_SHADOW_ONLY
    return report


def send_telegram_alert(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing; alert printed instead.")
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
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        print(f"Telegram delivery failed (non-fatal): {exc}")
        print(text)
        return False


def format_message(setup, report):
    lines = [
        "⛓ <b>[ZW=F] Anti-Hunt Short Setup V4</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"📅 {datetime.now(CHICAGO_TZ).strftime('%A %Y-%m-%d %H:%M %Z')}",
        f"🧠 Profile: <b>{setup['profile']}</b>",
        f"🔵 Open: <b>{setup['daily_open']}</b>",
        f"📊 Current: <code>{setup['current_price']}</code>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🔻 ENTRY: <code>{setup['entry']}</code>",
        f"🛑 STOP: <code>{setup['stop']}</code>",
        f"🎯 TARGET: <code>{setup['target']}</code>",
        f"Risk: <code>{setup['risk']}</code> | Reward: <code>{setup['reward']}</code> | R:R <code>{setup['rr']}</code>",
    ]

    ml = setup.get("ml_shadow", {})
    if ml.get("probability_win") is not None:
        lines.append(
            f"🤖 ML SHADOW: <code>{ml['probability_win']:.1%}</code> "
            f"target-before-stop | <b>{ml['decision']}</b>"
        )
    else:
        lines.append("🤖 ML SHADOW: collecting training data")

    if setup.get("stop_atr_multiple") is not None:
        lines.append(
            f"ATR: <code>{setup['atr_15m']}</code> | Stop distance: <code>{setup['stop_atr_multiple']}x</code>"
        )
    if setup["vol_warning"]:
        lines.append("⚠️ Stop is inside the 2x ATR noise band.")
    if setup.get("stale_data_min", 0) > MAX_DATA_AGE_MIN:
        lines.append(f"⚠️ STALE DATA: {setup['stale_data_min']} min")
    if report.get("learning_applied"):
        lines.append(
            f"🧠 Bounded profile changed: {report['previous_profile']} → {report['new_profile']}"
        )
    lines.append("⏳ Valid until 12:30 CT.")
    lines.append("🤖 ML is SHADOW-ONLY in V4; it does not change the signal.")
    return "\n".join(lines)


def main():
    manual = is_manual()
    resolve_only = is_resolve() and not manual
    now_ct = datetime.now(CHICAGO_TZ)

    if resolve_only:
        ping_healthcheck()
        if now_ct.date() in HOLIDAYS or now_ct.weekday() >= 5:
            print("No weekday trading session to resolve.")
            return 0
        print("=" * 50)
        print("OUTCOME RESOLUTION + ML LEARNING MODE")
        print("No new setup will be created.")
        print("=" * 50)

    elif not manual:
        ping_healthcheck()
        if now_ct.date() in HOLIDAYS:
            print(f"{now_ct.date()} is a CME holiday. Aborting.")
            return 0
        if not check_time_window():
            print(f"[{now_ct:%Y-%m-%d %H:%M %Z}] Outside execution window. Aborting.")
            return 0
        if now_ct.time() > ENTRY_CUTOFF:
            print("Past entry cutoff. Aborting.")
            return 0

    if manual:
        print("=" * 50)
        print("MANUAL TEST MODE")
        print("Normal time/holiday restrictions are bypassed.")
        print("Real ZW=F data will still be fetched.")
        print("=" * 50)

    print("Fetching ZW=F data...")
    try:
        intraday, daily = fetch_market_data()
    except ValueError as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 1

    state = load_state()

    resolved = resolve_previous_setups(state, intraday)
    print(f"Resolved {resolved} previous setup(s).")

    report = maybe_learn(state)
    ml_report = ml_shadow_report(state)
    save_json(REPORT_FILE, report)
    save_json(ML_REPORT_FILE, ml_report)

    if resolve_only:
        save_json(STATE_FILE, state)
        print(json.dumps(report, indent=2))
        print(json.dumps(ml_report, indent=2))
        return 0

    age = check_staleness(intraday)
    if age > MAX_DATA_AGE_MIN:
        print(f"WARNING: last 15m bar is {age:.0f} min old — alert will be flagged as stale.")

    daily_open = resolve_session_open(intraday, daily)
    current_price = float(intraday["Close"].iloc[-1])
    atr = compute_atr(intraday)

    profile_name = state["active_profile"]
    setup = build_setup(daily_open, current_price, atr, profile_name)
    setup["manual_mode"] = manual
    setup["stale_data_min"] = round(age, 1)

    # Train only on prior resolved actual outcomes.
    ml_training_report = add_ml_prediction(state, setup)
    setup["ml_shadow"]["training_report"] = ml_training_report

    print(json.dumps(setup, indent=2))

    if setup["invalidated"]:
        print("Setup invalidated; no alert.")
        save_json(STATE_FILE, state)
        save_json(SETUP_FILE, setup)
        return 0

    # Manual tests do not create learning samples.
    if not manual:
        state["setups"].append(setup)

    save_json(SETUP_FILE, setup)
    save_json(STATE_FILE, state)

    send_telegram_alert(format_message(setup, report))
    print(json.dumps(report, indent=2))
    print(json.dumps(ml_report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
