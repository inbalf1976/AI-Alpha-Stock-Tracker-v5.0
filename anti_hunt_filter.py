"""
anti_hunt_filter.py  (v2 - Production Fixed)
===========================================
Institutional open anti-stop-hunting Short filter for Chicago SRW Wheat (ZW=F).

Runs during the safe institutional window (8:45 AM - 12:30 PM America/Chicago,
weekdays only), anchors to the session Opening Price, and calculates a protected
Short setup engineered to sit outside typical high-frequency sweep zones.
"""

import os
import sys
import json
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, time as dt_time, date
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKER = "ZW=F"
CHICAGO_TZ = ZoneInfo("America/Chicago")
TICK_SIZE = 0.25          # ZW minimum price increment ($/bu)
ATR_PERIOD = 16           # 15m bars (~4h of session)
ATR_STOP_MIN_MULT = 2.0   # warn if stop distance < 2x ATR (noise-level)
MAX_DATA_AGE_MIN = 45     # staleness threshold for the last 15m bar
MAX_FETCH_ATTEMPTS = 3

WINDOW_OPEN = dt_time(8, 45)    # 8:45 AM Chicago  (grains open 8:30 CT)
WINDOW_CLOSE = dt_time(12, 30)  # 12:30 PM Chicago
ENTRY_CUTOFF = dt_time(11, 30)  # no fresh setups sent after this

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
HEALTHCHECK_URL = os.environ.get("HEALTHCHECK_URL", "").strip()

# Anti-hunt geometry (multipliers applied to the open anchor)
ENTRY_MULT = 0.994   # Sell Limit: waits for a brief pump into the entry zone
STOP_MULT = 1.012    # Stop-Loss: tucked above the institutional open/high ceiling
TARGET_MULT = 0.960  # Target: macro support zone

# CME full-day closures observed for grain futures. Update yearly.
HOLIDAYS = {
    date(2026, 1, 1),  date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),  date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1),  date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5),  date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def ping_healthcheck() -> None:
    if not HEALTHCHECK_URL:
        return
    try:
        requests.get(HEALTHCHECK_URL, timeout=10)
    except requests.RequestException as exc:
        print(f"Healthcheck ping failed (non-fatal): {exc}", file=sys.stderr)


def round_tick(price: float, tick: float = TICK_SIZE) -> float:
    return round(round(price / tick) * tick, 4)


def check_time_window() -> bool:
    now_ct = datetime.now(CHICAGO_TZ)
    if now_ct.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    return WINDOW_OPEN <= now_ct.time() <= WINDOW_CLOSE


def is_cme_holiday(d: date) -> bool:
    return d in HOLIDAYS


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    intraday = daily = None
    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            intraday = _flatten(yf.download(
                TICKER, period="2d", interval="15m",
                auto_adjust=False, progress=False,
            ))
            daily = _flatten(yf.download(
                TICKER, period="5d", interval="1d",
                auto_adjust=False, progress=False,
            ))
        except Exception as exc:
            print(f"Fetch attempt {attempt} raised: {exc}", file=sys.stderr)
        if intraday is not None and daily is not None \
                and not intraday.empty and not daily.empty:
            return intraday, daily
        time.sleep(10 * attempt)

    raise ValueError(
        "yfinance returned no usable data after "
        f"{MAX_FETCH_ATTEMPTS} attempts."
    )


def resolve_session_open(intraday: pd.DataFrame, daily: pd.DataFrame) -> float:
    now_ct = datetime.now(CHICAGO_TZ)
    idx = intraday.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    todays_bars = intraday[idx.tz_convert(CHICAGO_TZ).date == now_ct.date()]
    if not todays_bars.empty:
        return float(todays_bars["Open"].iloc[-1]) # FIXED: Brackets and index tracking fully added
    return float(daily["Open"].iloc[-1])


def compute_atr(intraday: pd.DataFrame) -> float | None:
    if len(intraday) < ATR_PERIOD + 1:
        return None
    high, low, close = intraday["High"], intraday["Low"], intraday["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(ATR_PERIOD).mean().iloc[-1])


def check_staleness(intraday: pd.DataFrame) -> float:
    idx = intraday.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    last_bar_ct = idx[-1].tz_convert(CHICAGO_TZ)
    return (datetime.now(CHICAGO_TZ) - last_bar_ct).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Telegram alerting
# ---------------------------------------------------------------------------
def send_telegram_alert(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing — alert printed to stdout instead.")
        print(text)
        return False

    url = f"https://telegram.org{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            print("📲 Telegram alert delivered successfully!")
            return True
        else:
            print(f"❌ Telegram API returned error: {response.text}", file=sys.stderr)
            return False
    except Exception as exc:
        print(f"❌ Network error trying to call Telegram: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Engine core
# ---------------------------------------------------------------------------
def run_anti_hunt_logic(bypass_gates=False) -> None:
    now_ct = datetime.now(CHICAGO_TZ)
    
    # 1. Structural Environment Verification
    if not bypass_gates:
        if is_cme_holiday(now_ct.date()):
            print(f"[{now_ct.date()}] CME Holiday observed. Aborting gracefully.")
            return
        if not check_time_window():
            print(f"[{now_ct.strftime('%Y-%m-%d %H:%M %Z')}] Outside the safe institutional window (Mon-Fri, 08:45-12:30 America/Chicago). Aborting gracefully.")
            return

    print("🧪 RUNNING IN FORCED MANUAL TEST MODE (Bypassing Time, Day, and Staleness Gates)...")
    print(f"⚡ Institutional Core Analysis Active [{now_ct.strftime('%H:%M:%S %Z')}]")

    # 2. Ingest Data Stream
    try:
        intraday_data, daily_data = fetch_market_data()
    except Exception as e:
        print(f"🚫 Critical data error: {e}", file=sys.stderr)
        return

    # 3. Data Integrity Constraints Check
    staleness_min = check_staleness(intraday_data)
    if not bypass_gates:
        if staleness_min > MAX_DATA_AGE_MIN:
            print(f"🚫 Pipeline Stalled: Data age is {staleness_min:.1f} minutes. Maximum allowed is {MAX_DATA_AGE_MIN}m.", file=sys.stderr)
            return

    # 4. Resolve Boundary Anchors & Metrics
    session_open = resolve_session_open(intraday_data, daily_data)
    current_price = float(intraday_data["Close"].iloc[-1])
    atr_value = compute_atr(intraday_data)

    # 5. Execute Geometry Computations
    entry_level = round_tick(session_open * ENTRY_MULT)
    stop_level  = round_tick(session_open * STOP_MULT)
    target_level = round_tick(session_open * TARGET_MULT)

    # Risk Metrics Review
    stop_distance_points = stop_level - entry_level
    noise_floor_warn = False
    if atr_value is not None:
        if stop_distance_points < (ATR_STOP_MIN_MULT * atr_value):
            noise_floor_warn = True

    # 6. Invalidation Logic Checks
    if current_price >= stop_level:
        print(f"❌ Setup Cancelled: Current price ({current_price}) is already trading above calculated stop ({stop_level}). Strategy invalidated.")
        return

    if not bypass_gates and now_ct.time() > ENTRY_CUTOFF:
        print(f"⏰ Execution Alert Threshold Reached: Current time is past entry cutoff ({ENTRY_CUTOFF.strftime('%H:%M')}). No new setups generated.")
        return

    # 7. Formulate Delivery Payload
    atr_display = f"{atr_value:.2f}c" if atr_value is not None else "N/A"
    warning_block = ""
    if noise_floor_warn:
        warning_block = f"\n⚠️ <b>RISK WARNING:</b> Stop distance ({stop_distance_points:.2f}c) is thinner than 2x ATR volatility threshold ({ATR_STOP_MIN_MULT * atr_value:.2f}c). Noise hunt risk high."

    msg = (
        f"🌾 <b>ANTI-HUNT WHEAT FILTER (v2)</b>\n"
        f"🕒 Time: <code>{now_ct.strftime('%H:%M:%S')} CST</code>\n\n"
        f"📈 Session Open Anchor: <code>{session_open:.2f}c</code>\n"
        f"💵 Current Spot Price: <code>{current_price:.2f}c</code>\n"
        f"📊 15m ATR Volatility: <code>{atr_display}</code>\n"
        f"⏱️ Bar Latency Age: <code>{staleness_min:.1f} min</code>\n{warning_block}\n"
        f"🛡️ <b>THE PROTECTED GRID SETUP:</b>\n"
