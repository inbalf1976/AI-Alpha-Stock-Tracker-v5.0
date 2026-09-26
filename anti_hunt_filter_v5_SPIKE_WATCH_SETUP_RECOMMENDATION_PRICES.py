
import os
import sys
import time
from datetime import datetime, time as dt_time, date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# V5: bounded strategy + richer market features + walk-forward shadow ML
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

# Spike-watch layer: designed to detect a developing upside volatility expansion
# BEFORE the normal trade signal window. It is informational only and never
# changes the existing short setup or ML decision.
SPIKE_WATCH_OPEN = dt_time(7, 30)
SPIKE_WATCH_CLOSE = dt_time(12, 30)
SPIKE_SCORE_THRESHOLD = 5
SPIKE_LOOKBACK_BARS = 12
SPIKE_BREAKOUT_LOOKBACK = 20
SPIKE_COMPRESSION_THRESHOLD = 0.85
SPIKE_VOLUME_THRESHOLD = 1.25
SPIKE_ATR_EXPANSION_THRESHOLD = 1.10

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
ML_MIN_SAMPLES = 60
ML_MIN_CLASS_COUNT = 15
ML_CONFIDENCE_THRESHOLD = 0.60
ML_TEST_WINDOW = 20
ML_MIN_TRAIN_SAMPLES = 40

STATE_FILE = "learning_state.json"
OUTCOME_VERSION = 5
SPIKE_WATCH_VERSION = 2
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
    "rsi_14",
    "vwap_distance_pct",
    "volume_ratio",
    "momentum_4bar_pct",
    "trend_16bar_pct",
    "range_position",
    "opening_range_position",
    "prior_day_range_pct",
    "current_vs_prior_high_pct",
    "current_vs_prior_low_pct",
    "atr_regime_ratio",
    "trend_1h_pct",
    "regime_score",
    "daily_5bar_pct",
    "daily_20bar_pct",
    "daily_20d_breakout_pct",
    "daily_60d_breakout_pct",
    "daily_252d_breakout_pct",
    "daily_volatility_ratio",
    "daily_range_compression_ratio",
    "daily_volume_ratio",
    "weekly_4bar_pct",
    "weekly_13bar_pct",
]


def is_manual():
    return "--manual" in sys.argv


def is_resolve():
    return "--resolve" in sys.argv


def is_force_stale():
    return "--force-stale" in sys.argv


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
            hourly = _flatten(yf.download(
                TICKER, period="5d", interval="60m",
                auto_adjust=False, progress=False
            ))
            # Keep enough daily history to detect the type of multi-month/year
            # compression and breakout visible in the long-term charts. Five
            # days is not enough for this layer.
            daily_long = _flatten(yf.download(
                TICKER, period="5y", interval="1d",
                auto_adjust=False, progress=False
            ))
            if not intraday.empty and not hourly.empty and not daily_long.empty:
                return intraday, daily_long, hourly
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
        "version": 5,
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
        state.setdefault("version", 5)
        state["version"] = 5
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


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rsi(series, period=14):
    delta = series.astype(float).diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def session_slice(intraday, ts_date):
    idx = chicago_index(intraday)
    mask = [ts.date() == ts_date for ts in idx]
    return intraday.loc[mask].copy()


def market_context(intraday, daily, hourly, now_ct):
    df = intraday.copy()
    idx = chicago_index(df)
    df.index = idx
    df = df.sort_index()
    today = df.loc[df.index.date == now_ct.date()].copy()
    if today.empty:
        today = df.tail(32).copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(1.0, index=df.index)

    latest = float(close.iloc[-1])
    rsi14 = float(rsi(close, 14).iloc[-1])
    momentum_4 = (latest / float(close.iloc[-5]) - 1.0) * 100.0 if len(close) >= 5 else 0.0
    trend_16 = (latest / float(close.iloc[-17]) - 1.0) * 100.0 if len(close) >= 17 else 0.0

    recent_vol = volume.tail(20).replace([np.inf, -np.inf], np.nan).dropna()
    vol_ratio = latest_vol = float(volume.iloc[-1]) / float(recent_vol.median()) if not recent_vol.empty and recent_vol.median() > 0 else 1.0

    typical = (today["High"].astype(float) + today["Low"].astype(float) + today["Close"].astype(float)) / 3.0
    tv = today["Volume"].astype(float) if "Volume" in today.columns else pd.Series(1.0, index=today.index)
    denom = float(tv.sum())
    vwap = float((typical * tv).sum() / denom) if denom > 0 else latest
    vwap_distance = (latest - vwap) / vwap * 100.0 if vwap else 0.0

    session_high = float(today["High"].astype(float).max())
    session_low = float(today["Low"].astype(float).min())
    session_span = session_high - session_low
    range_position = (latest - session_low) / session_span if session_span > 0 else 0.5

    opening = today.head(4)
    opening_high = float(opening["High"].astype(float).max()) if not opening.empty else latest
    opening_low = float(opening["Low"].astype(float).min()) if not opening.empty else latest
    opening_span = opening_high - opening_low
    opening_position = (latest - opening_low) / opening_span if opening_span > 0 else 0.5

    prior = daily.iloc[-2] if len(daily) >= 2 else daily.iloc[-1]
    prior_high = _safe_float(prior.get("High"), latest)
    prior_low = _safe_float(prior.get("Low"), latest)
    prior_open = _safe_float(prior.get("Open"), latest)
    prior_range_pct = (prior_high - prior_low) / prior_open * 100.0 if prior_open else 0.0
    vs_prior_high = (latest - prior_high) / prior_high * 100.0 if prior_high else 0.0
    vs_prior_low = (latest - prior_low) / prior_low * 100.0 if prior_low else 0.0

    atr_series = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()
    atr_now = _safe_float(atr_series.iloc[-1], 0.0)
    atr_short = _safe_float(atr_series.tail(8).mean(), atr_now)
    atr_regime = atr_now / atr_short if atr_short > 0 else 1.0

    h = hourly.copy()
    h.index = chicago_index(h)
    hclose = h["Close"].astype(float)
    trend_1h = (float(hclose.iloc[-1]) / float(hclose.iloc[-4]) - 1.0) * 100.0 if len(hclose) >= 4 else 0.0

    trend_score = 1 if trend_16 > 0.20 else -1 if trend_16 < -0.20 else 0
    if atr_regime > 1.35:
        regime = "HIGH_VOL"
    elif atr_regime < 0.75:
        regime = "LOW_VOL"
    elif trend_score > 0:
        regime = "TREND_UP"
    elif trend_score < 0:
        regime = "TREND_DOWN"
    else:
        regime = "RANGE"

    return {
        "rsi_14": round(rsi14, 4),
        "vwap": round(vwap, 4),
        "vwap_distance_pct": round(vwap_distance, 6),
        "volume_ratio": round(vol_ratio, 4),
        "momentum_4bar_pct": round(momentum_4, 6),
        "trend_16bar_pct": round(trend_16, 6),
        "range_position": round(range_position, 6),
        "opening_range_position": round(opening_position, 6),
        "prior_day_range_pct": round(prior_range_pct, 6),
        "current_vs_prior_high_pct": round(vs_prior_high, 6),
        "current_vs_prior_low_pct": round(vs_prior_low, 6),
        "atr_regime_ratio": round(atr_regime, 6),
        "trend_1h_pct": round(trend_1h, 6),
        "regime": regime,
        "regime_score": trend_score,
    }

def daily_spike_context(daily, current_price, now_ct):
    """Measure the multi-week/month regime visible in the long-term charts.

    This is not a prediction by itself. It identifies compression, trend,
    breakout and volatility-expansion conditions on daily data so the
    intraday SPIKE WATCH can distinguish an ordinary move from a move that is
    occurring inside a larger expansion regime.
    """
    d = daily.copy()
    d.index = chicago_index(d)
    d = d.sort_index()
    # Use only completed daily bars for long-term measurements. The current
    # session is represented by current_price from the 15m feed and must not
    # leak a partial daily bar into the historical reference levels.
    completed = d.loc[d.index.date < now_ct.date()].copy()
    if not completed.empty:
        d = completed
    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    volume = d["Volume"].astype(float) if "Volume" in d.columns else pd.Series(1.0, index=d.index)

    if len(close) < 70:
        return {
            "enabled": True,
            "qualified": False,
            "stage": "INSUFFICIENT_DAILY_HISTORY",
            "reason": "Need at least 70 daily bars for long-term spike context.",
            "version": SPIKE_WATCH_VERSION,
        }

    def pct_from_lag(n):
        return (float(current_price) / float(close.iloc[-1 - n]) - 1.0) * 100.0 if len(close) > n else 0.0

    daily_5 = pct_from_lag(5)
    daily_20 = pct_from_lag(20)

    # Breakout distances use the highest completed daily bar before today.
    high20 = float(high.tail(20).max())
    high60 = float(high.tail(60).max())
    high252 = float(high.tail(min(252, len(high))).max())
    breakout20 = (float(current_price) / high20 - 1.0) * 100.0 if high20 else 0.0
    breakout60 = (float(current_price) / high60 - 1.0) * 100.0 if high60 else 0.0
    breakout252 = (float(current_price) / high252 - 1.0) * 100.0 if high252 else 0.0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr20 = tr.rolling(20).mean()
    atr60 = tr.rolling(60).mean()
    daily_volatility_ratio = float(atr20.iloc[-1] / atr60.iloc[-1]) if pd.notna(atr20.iloc[-1]) and pd.notna(atr60.iloc[-1]) and atr60.iloc[-1] > 0 else 1.0

    range20 = (high.tail(20) - low.tail(20)).mean()
    range60 = (high.tail(60) - low.tail(60)).mean()
    compression_ratio = float(range20 / range60) if range60 > 0 else 1.0

    recent_volume = float(volume.tail(5).mean())
    base_volume = float(volume.tail(60).head(55).median()) if len(volume) >= 60 else recent_volume
    volume_ratio = recent_volume / base_volume if base_volume > 0 else 1.0

    weekly = close.resample("W-FRI").last().dropna()
    weekly_4 = (float(weekly.iloc[-1]) / float(weekly.iloc[-5]) - 1.0) * 100.0 if len(weekly) >= 5 else 0.0
    weekly_13 = (float(weekly.iloc[-1]) / float(weekly.iloc[-14]) - 1.0) * 100.0 if len(weekly) >= 14 else 0.0

    # Directional daily regime used by the spike alert. This is descriptive
    # context, not a trade trigger: the spike direction is compared against
    # the prevailing daily direction in the alert recommendation.
    if daily_20 >= 0.50 and weekly_13 >= -0.50:
        daily_direction = "UP"
    elif daily_20 <= -0.50 and weekly_13 <= 0.50:
        daily_direction = "DOWN"
    else:
        daily_direction = "NEUTRAL"

    checks = {
        "daily_compression": compression_ratio <= 0.85,
        "daily_volatility_expansion": daily_volatility_ratio >= 1.15,
        "daily_20d_breakout": breakout20 >= 0.0,
        "daily_60d_breakout": breakout60 >= 0.0,
        "daily_252d_breakout": breakout252 >= -0.5,
        "daily_momentum": daily_20 >= 2.0,
        "volume_expansion": volume_ratio >= 1.15,
        "weekly_trend": weekly_13 > 3.0,
    }
    score = sum(bool(v) for v in checks.values())

    if score >= 6:
        stage = "MACRO_SPIKE_REGIME"
    elif score >= 4:
        stage = "MACRO_SPIKE_WATCH"
    else:
        stage = "NORMAL_REGIME"

    return {
        "enabled": True,
        "version": SPIKE_WATCH_VERSION,
        "qualified": score >= 4,
        "stage": stage,
        "score": score,
        "max_score": len(checks),
        "checks": checks,
        "daily_5bar_pct": round(daily_5, 4),
        "daily_20bar_pct": round(daily_20, 4),
        "daily_20d_breakout_pct": round(breakout20, 4),
        "daily_60d_breakout_pct": round(breakout60, 4),
        "daily_252d_breakout_pct": round(breakout252, 4),
        "daily_volatility_ratio": round(daily_volatility_ratio, 4),
        "daily_range_compression_ratio": round(compression_ratio, 4),
        "daily_volume_ratio": round(volume_ratio, 4),
        "weekly_4bar_pct": round(weekly_4, 4),
        "weekly_13bar_pct": round(weekly_13, 4),
        "daily_direction": daily_direction,
        "month": now_ct.month,
        "week_of_year": int(now_ct.isocalendar().week),
    }

def spike_watch_context(intraday, daily, hourly, now_ct, macro=None):
    """Detect directional spike conditions: UP or DOWN. Shadow/informational only."""
    df = intraday.copy()
    idx = chicago_index(df)
    df.index = idx
    df = df.sort_index()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(1.0, index=df.index)
    if len(df) < 35:
        return {"enabled": True, "qualified": False, "score": 0, "max_score": 10,
                "stage": "INSUFFICIENT_DATA", "direction": "NONE",
                "reason": "Need at least 35 completed 15m bars."}
    latest = float(close.iloc[-1])
    bar_range = (high - low).replace([np.inf, -np.inf], np.nan)
    recent_range = _safe_float(bar_range.tail(8).mean(), 0.0)
    baseline_range = _safe_float(bar_range.tail(32).head(24).mean(), recent_range)
    compression_ratio = recent_range / baseline_range if baseline_range > 0 else 1.0
    compressed = compression_ratio <= SPIKE_COMPRESSION_THRESHOLD
    recent_volume = volume.tail(4).mean()
    base_volume = volume.tail(24).head(20).median()
    volume_ratio = recent_volume / base_volume if base_volume and base_volume > 0 else 1.0
    volume_expansion = volume_ratio >= SPIKE_VOLUME_THRESHOLD
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_series = tr.rolling(ATR_PERIOD).mean()
    atr_now = _safe_float(atr_series.iloc[-1], 0.0)
    atr_prev = _safe_float(atr_series.iloc[-5], atr_now)
    atr_expansion_ratio = atr_now / atr_prev if atr_prev > 0 else 1.0
    atr_expansion = atr_expansion_ratio >= SPIKE_ATR_EXPANSION_THRESHOLD
    macro = macro or daily_spike_context(daily, latest, now_ct)
    macro_qualified = bool(macro.get("qualified"))

    recent_lows = low.tail(8).to_numpy()
    recent_highs_8 = high.tail(8).to_numpy()
    higher_low_count = int(np.sum(np.diff(recent_lows) > 0))
    lower_high_count = int(np.sum(np.diff(recent_highs_8) < 0))
    higher_lows = higher_low_count >= 4
    lower_highs = lower_high_count >= 4

    prior_20_high = high.rolling(SPIKE_BREAKOUT_LOOKBACK).max().shift(1)
    prior_20_low = low.rolling(SPIKE_BREAKOUT_LOOKBACK).min().shift(1)
    recent_resistance = prior_20_high.tail(SPIKE_LOOKBACK_BARS)
    recent_support = prior_20_low.tail(SPIKE_LOOKBACK_BARS)
    recent_highs = high.tail(SPIKE_LOOKBACK_BARS)
    recent_lows_window = low.tail(SPIKE_LOOKBACK_BARS)
    valid_resistance = recent_resistance.notna()
    valid_support = recent_support.notna()
    resistance_tests = int(((recent_highs[valid_resistance] >= recent_resistance[valid_resistance] * 0.997)).sum()) if valid_resistance.any() else 0
    support_tests = int(((recent_lows_window[valid_support] <= recent_support[valid_support] * 1.003)).sum()) if valid_support.any() else 0
    resistance_pressure = resistance_tests >= 2
    support_pressure = support_tests >= 2
    prior_resistance = _safe_float(prior_20_high.iloc[-1], latest)
    prior_support = _safe_float(prior_20_low.iloc[-1], latest)
    breakout_up_pct = (latest / prior_resistance - 1.0) * 100.0 if prior_resistance else 0.0
    breakout_down_pct = (latest / prior_support - 1.0) * 100.0 if prior_support else 0.0
    breakout_up = breakout_up_pct >= 0.15
    breakout_down = breakout_down_pct <= -0.15

    mom_now = (latest / float(close.iloc[-5]) - 1.0) * 100.0
    prior_close = float(close.iloc[-9])
    mom_prev = (float(close.iloc[-5]) / prior_close - 1.0) * 100.0 if prior_close else 0.0
    momentum_up = mom_now > 0.10 and mom_now > mom_prev
    momentum_down = mom_now < -0.10 and mom_now < mom_prev

    h = hourly.copy()
    h.index = chicago_index(h)
    hclose = h["Close"].astype(float)
    trend_1h = (float(hclose.iloc[-1]) / float(hclose.iloc[-4]) - 1.0) * 100.0 if len(hclose) >= 4 else 0.0
    trend_16 = (latest / float(close.iloc[-17]) - 1.0) * 100.0
    trend_up = trend_16 > 0.15 and trend_1h > 0.10
    trend_down = trend_16 < -0.15 and trend_1h < -0.10

    today = df.loc[df.index.date == now_ct.date()]
    if today.empty:
        today = df.tail(32)
    day_low = float(today["Low"].min())
    day_high = float(today["High"].max())
    day_span = day_high - day_low
    recovery_position = (latest - day_low) / day_span if day_span > 0 else 0.5
    morning_reversal_up = recovery_position >= 0.70 and latest > day_low
    morning_reversal_down = recovery_position <= 0.30 and latest < day_high

    neutral = {"compression": compressed, "volume_expansion": volume_expansion,
               "atr_expansion": atr_expansion, "macro_spike_regime": macro_qualified}
    up_checks = {**neutral, "higher_lows": higher_lows, "resistance_pressure": resistance_pressure,
                 "breakout": breakout_up, "momentum_acceleration": momentum_up,
                 "trend_alignment": trend_up, "morning_reversal": morning_reversal_up}
    down_checks = {**neutral, "lower_highs": lower_highs, "support_pressure": support_pressure,
                   "breakdown": breakout_down, "momentum_acceleration": momentum_down,
                   "trend_alignment": trend_down, "morning_reversal": morning_reversal_down}
    up_score = sum(bool(v) for v in up_checks.values())
    down_score = sum(bool(v) for v in down_checks.values())

    if up_score >= SPIKE_SCORE_THRESHOLD and breakout_up:
        direction, score, checks, stage = "UP", up_score, up_checks, "SPIKE_CONFIRMED"
    elif down_score >= SPIKE_SCORE_THRESHOLD and breakout_down:
        direction, score, checks, stage = "DOWN", down_score, down_checks, "SPIKE_CONFIRMED"
    elif max(up_score, down_score) >= SPIKE_SCORE_THRESHOLD - 1:
        if up_score >= down_score:
            direction, score, checks = "UP", up_score, up_checks
        else:
            direction, score, checks = "DOWN", down_score, down_checks
        stage = "SPIKE_WATCH"
    elif max(up_score, down_score) >= 3:
        if up_score >= down_score:
            direction, score, checks = "UP", up_score, up_checks
        else:
            direction, score, checks = "DOWN", down_score, down_checks
        stage = "EARLY_WATCH"
    else:
        direction, score, checks, stage = "NONE", max(up_score, down_score), {}, "NO_SPIKE_SIGNAL"

    return {"enabled": True, "qualified": stage in {"SPIKE_CONFIRMED", "SPIKE_WATCH"},
            "score": score, "up_score": up_score, "down_score": down_score, "max_score": 10,
            "stage": stage, "direction": direction, "checks": checks,
            "compression_ratio": round(compression_ratio, 4), "higher_low_count": higher_low_count,
            "lower_high_count": lower_high_count, "resistance_tests": resistance_tests,
            "support_tests": support_tests, "breakout_pct": round(breakout_up_pct if direction == "UP" else breakout_down_pct, 4),
            "breakout_up_pct": round(breakout_up_pct, 4), "breakout_down_pct": round(breakout_down_pct, 4),
            "volume_ratio": round(volume_ratio, 4), "momentum_4bar_pct": round(mom_now, 4),
            "momentum_prev_4bar_pct": round(mom_prev, 4), "atr_expansion_ratio": round(atr_expansion_ratio, 4),
            "trend_16bar_pct": round(trend_16, 4), "trend_1h_pct": round(trend_1h, 4),
            "recovery_position": round(recovery_position, 4), "threshold": SPIKE_SCORE_THRESHOLD,
            "macro_context": macro, "version": SPIKE_WATCH_VERSION + 1}


def format_spike_watch_message(spike, now_ct):
    stage = spike.get("stage", "NO_SPIKE_SIGNAL")
    direction = spike.get("direction", "NONE")
    score = spike.get("score", 0)
    max_score = spike.get("max_score", 10)
    lines = [
        "🚨 <b>[ZW=F] SPIKE WATCH V5</b>", "━━━━━━━━━━━━━━━━━━━━",
        f"📅 {now_ct.strftime('%A %Y-%m-%d %H:%M %Z')}",
        f"⚡ Stage: <b>{stage}</b>", f"🧭 Direction: <b>{direction}</b>",
        f"📈 Precursor score: <b>{score}/{max_score}</b>",
        f"⬆️ UP score: <code>{spike.get('up_score', 0)}/{max_score}</code> | ⬇️ DOWN score: <code>{spike.get('down_score', 0)}/{max_score}</code>",
        "━━━━━━━━━━━━━━━━━━━━"]
    labels = {"compression":"Compression","higher_lows":"Higher lows","lower_highs":"Lower highs",
              "resistance_pressure":"Resistance pressure","support_pressure":"Support pressure",
              "breakout":"Breakout","breakdown":"Breakdown","volume_expansion":"Volume expansion",
              "momentum_acceleration":"Momentum acceleration","atr_expansion":"ATR expansion",
              "trend_alignment":"15m/1h trend alignment","morning_reversal":"Morning reversal",
              "macro_spike_regime":"Multi-month spike regime"}
    for key, label in labels.items():
        if key in spike.get("checks", {}):
            lines.append(f"{'✅' if spike['checks'].get(key) else '▫️'} {label}")
    daily_direction = spike.get("macro_context", {}).get("daily_direction", "NEUTRAL")
    counter_trend = direction in {"UP", "DOWN"} and daily_direction in {"UP", "DOWN"} and direction != daily_direction
    aligned = direction in {"UP", "DOWN"} and daily_direction == direction

    if stage == "SPIKE_CONFIRMED":
        if counter_trend:
            setup_recommendation = f"WAIT — SPIKE {direction} IS AGAINST DAILY DIRECTION {daily_direction}"
            setup_reason = (
                f"⚠️ COUNTER-TREND SPIKE. The {direction} spike is against the daily {daily_direction} direction. "
                f"Do not chase the {direction} move. Wait for exhaustion/rejection and for daily {daily_direction} control to regain confirmation."
            )
        elif aligned:
            setup_recommendation = f"SPIKE {direction} ALIGNED WITH DAILY DIRECTION {daily_direction}"
            setup_reason = f"✅ The {direction} spike agrees with the daily {daily_direction} direction. Continue monitoring the V5 setup; do not chase the spike."
        else:
            setup_recommendation = f"WAIT — SPIKE {direction} CONFIRMED; do not chase the move"
            setup_reason = f"A {direction.lower()} spike is confirmed, while the daily direction is {daily_direction}. Wait for confirmation/rejection before considering the normal setup."
    elif stage == "SPIKE_WATCH":
        if counter_trend:
            setup_recommendation = f"WAIT — SPIKE WATCH {direction} IS AGAINST DAILY DIRECTION {daily_direction}"
            setup_reason = f"⚠️ Counter-trend spike watch. The {direction} move is against daily {daily_direction}; wait for exhaustion/rejection and confirmation."
        elif aligned:
            setup_recommendation = f"WAIT — SPIKE WATCH {direction} ALIGNED WITH DAILY DIRECTION {daily_direction}"
            setup_reason = f"The developing {direction} spike agrees with daily {daily_direction}; wait for confirmation and do not chase."
        else:
            setup_recommendation = f"WAIT — SPIKE WATCH {direction}; no entry yet"
            setup_reason = f"A {direction.lower()} spike pattern is developing; daily direction is {daily_direction}. Wait for confirmation or rejection."
    elif stage == "EARLY_WATCH":
        setup_recommendation = f"WATCH — {direction} spike developing; no trade yet"
        setup_reason = f"Daily direction: {daily_direction}. Monitor for confirmation; the normal Anti-Hunt setup is not active premarket."
    else:
        setup_recommendation = "NO TRADE — no qualifying spike pattern"
        setup_reason = f"No actionable directional spike precursor detected. Daily direction: {daily_direction}."
    lines.extend([
        "━━━━━━━━━━━━━━━━━━━━",
        f"📊 <b>Daily Direction: {daily_direction}</b>",
        f"🧭 <b>Spike Direction: {direction}</b>",
        "🎯 <b>SETUP RECOMMENDATION</b>",
        f"<b>{setup_recommendation or 'NO TRADE — recommendation unavailable'}</b>",
        f"ℹ️ {setup_reason or 'No additional setup guidance is available.'}",
        "🔻 Normal short setup: <b>NOT ACTIVE in premarket</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Breakout/Breakdown: <code>{spike.get('breakout_pct', 0):.2f}%</code> | Volume: <code>{spike.get('volume_ratio', 1):.2f}x</code>",
        f"ATR expansion: <code>{spike.get('atr_expansion_ratio', 1):.2f}x</code> | Recovery: <code>{spike.get('recovery_position', 0):.0%}</code>",
        f"Macro regime: <b>{spike.get('macro_context', {}).get('stage', 'N/A')}</b> <code>{spike.get('macro_context', {}).get('score', 0)}/{spike.get('macro_context', {}).get('max_score', 0)}</code>",
        "⚠️ Spike Watch is shadow/informational only — it does not create or modify a trade."])
    return "\n".join(lines)


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
        "⛓ <b>[ZW=F] Anti-Hunt Short Setup V5</b>",
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

    if setup.get("regime"):
        lines.append(f"🌡 Regime: <code>{setup['regime']}</code> | RSI <code>{setup.get('rsi_14', 0):.1f}</code> | VWAP dist <code>{setup.get('vwap_distance_pct', 0):.2f}%</code>")
    spike = setup.get("spike_watch", {})
    if spike:
        lines.append(
            f"🚨 SPIKE WATCH: <b>{spike.get('stage', 'N/A')}</b> "
            f"<code>{spike.get('score', 0)}/{spike.get('max_score', 0)}</code>"
        )
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
    lines.append("🤖 ML is SHADOW-ONLY in V5; it does not change the signal.")
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
        # The spike-watch layer is allowed to run earlier than the normal
        # trade window so it can warn before the breakout. The normal setup
        # remains restricted to WINDOW_OPEN/WINDOW_CLOSE and ENTRY_CUTOFF.
        in_spike_window = SPIKE_WATCH_OPEN <= now_ct.time() <= SPIKE_WATCH_CLOSE
        in_trade_window = check_time_window() and now_ct.time() <= ENTRY_CUTOFF
        if not in_spike_window:
            print(f"[{now_ct:%Y-%m-%d %H:%M %Z}] Outside spike-watch window. Aborting.")
            return 0

    if manual:
        print("=" * 50)
        print("MANUAL TEST MODE")
        print("Normal time/holiday restrictions are bypassed.")
        print("Real ZW=F data will still be fetched.")
        print("=" * 50)

    print("Fetching ZW=F data...")
    try:
        intraday, daily, hourly = fetch_market_data()
    except ValueError as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 1

    # Calculate both the long-term regime and intraday precursor. The two
    # layers are intentionally separate: a spike is more interesting when
    # intraday acceleration occurs inside a multi-week/month expansion regime.
    current_price_for_context = float(intraday["Close"].iloc[-1])
    macro = daily_spike_context(daily, current_price_for_context, now_ct)
    spike = spike_watch_context(intraday, daily, hourly, now_ct, macro=macro)
    print(json.dumps({"spike_watch": spike}, indent=2))

    state = load_state()

    resolved = resolve_previous_setups(state, intraday)
    print(f"Resolved {resolved} previous setup(s).")

    # Premarket / early-session spike warning. This path intentionally does
    # not create a trade setup or learning sample.
    in_trade_window = WINDOW_OPEN <= now_ct.time() <= WINDOW_CLOSE and now_ct.time() <= ENTRY_CUTOFF
    if not manual and not in_trade_window:
        if spike.get("qualified"):
            last_key = state.get("last_spike_watch_key")
            current_key = f"{now_ct.date().isoformat()}:{spike.get('stage')}:{spike.get('score')}"
            if last_key != current_key:
                send_telegram_alert(format_spike_watch_message(spike, now_ct))
                state["last_spike_watch_key"] = current_key
                save_json(STATE_FILE, state)
            else:
                print("Spike-watch alert already sent for this stage/score.")
        else:
            print("No qualifying premarket spike-watch pattern.")
        return 0

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
        print(f"WARNING: last 15m bar is {age:.0f} min old (max allowed {MAX_DATA_AGE_MIN} min).")
        if not (manual and is_force_stale()):
            print(
                "ABORT: data too stale — no setup will be built, no alert sent."
                + (" Use --manual --force-stale to test with stale data anyway." if manual else "")
            )
            save_json(STATE_FILE, state)
            return 0
        print("--force-stale set: continuing with stale data for manual test only.")

    daily_open = resolve_session_open(intraday, daily)
    current_price = float(intraday["Close"].iloc[-1])
    atr = compute_atr(intraday)

    profile_name = state["active_profile"]
    setup = build_setup(daily_open, current_price, atr, profile_name)
    setup["manual_mode"] = manual
    setup["stale_data_min"] = round(age, 1)
    context = market_context(intraday, daily, hourly, now_ct)
    setup["market_context"] = context
    setup.update(context)
    setup["macro_spike_context"] = macro
    for macro_key in ML_FEATURES:
        if macro_key in macro:
            setup[macro_key] = macro[macro_key]
    setup["spike_watch"] = spike

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
