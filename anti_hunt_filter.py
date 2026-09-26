"""
anti_hunt_filter.py  (v2 - Sanitized Production)
==============================================
Institutional open anti-stop-hunting Short filter for Chicago SRW Wheat (ZW=F).
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
TICK_SIZE = 0.25          
ATR_PERIOD = 16           
ATR_STOP_MIN_MULT = 2.0   
MAX_DATA_AGE_MIN = 45     
MAX_FETCH_ATTEMPTS = 3

WINDOW_OPEN = dt_time(8, 45)    
WINDOW_CLOSE = dt_time(12, 30)  
ENTRY_CUTOFF = dt_time(11, 30)  

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
HEALTHCHECK_URL = os.environ.get("HEALTHCHECK_URL", "").strip()

ENTRY_MULT = 0.994   
STOP_MULT = 1.012    
TARGET_MULT = 0.960  

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
    if now_ct.weekday() >= 5:  
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

    raise ValueError("yfinance returned no usable data.")


def resolve_session_open(intraday: pd.DataFrame, daily: pd.DataFrame) -> float:
    now_ct = datetime.now(CHICAGO_TZ)
    idx = intraday.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    todays_bars = intraday[idx.tz_convert(CHICAGO_TZ).date == now_ct.date()]
    if not todays_bars.empty:
        return float(todays_bars["Open"].iloc[-1]) 
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
# Telegram alerting (With Auto-Sanitize Bug Fix)
# ---------------------------------------------------------------------------
def send_telegram_alert(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing — alert printed to stdout instead.")
        print(text)
        return False

    # 🛡️ THE CRITICAL AUTO-SANITY FIX:
    # Strips out any full URLs or trailing text that leaked into the secret variable
    clean_token = TELEGRAM_BOT_TOKEN.strip()
    if "bot" in clean_token:
        clean_token = clean_token.split("bot")[-1]
    if "api.telegram.org" in clean_token:
        clean_token = clean_token.split("/")[-1]

    url = f"https://telegram.org{clean_token}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID.strip(),
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
            print(f"❌ Telegram API error: {response.text}", file=sys.stderr)
            return False
    except Exception as exc:
        print(f"❌ Network error: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Engine core
# ---------------------------------------------------------------------------
def run_anti_hunt_logic(bypass_gates=False) -> None:
    now_ct = datetime.now(CHICAGO_TZ)
    
    if not bypass_gates:
        if is_cme_holiday(now_ct.date()):
            print(f"[{now_ct.date()}] CME Holiday observed. Aborting gracefully.")
            return
        if not check_time_window():
            print(f"[{now_ct.strftime('%Y-%m-%d %H:%M %Z')}] Outside the safe window. Aborting.")
            return

    print("🧪 RUNNING IN FORCED MANUAL TEST MODE...")
    
    try:
        intraday_data, daily_data = fetch_market_data()
    except Exception as e:
        print(f"🚫 Critical data error: {e}", file=sys.stderr)
        return

    staleness_min = check_staleness(intraday_data)
    if not bypass_gates and staleness_min > MAX_DATA_AGE_MIN:
        print(f"🚫 Pipeline Stalled: Data age is {staleness_min:.1f} minutes.", file=sys.stderr)
        return

    session_open = resolve_session_open(intraday_data, daily_data)
    current_price = float(intraday_data["Close"].iloc[-1])
    atr_value = compute_atr(intraday_data)

    entry_level = round_tick(session_open * ENTRY_MULT)
    stop_level  = round_tick(session_open * STOP_MULT)
    target_level = round_tick(session_open * TARGET_MULT)

    stop_distance_points = stop_level - entry_level
    noise_floor_warn = False
    if atr_value is not None:
        if stop_distance_points < (ATR_STOP_MIN_MULT * atr_value):
            noise_floor_warn = True

    if current_price >= stop_level:
        print(f"❌ Setup Cancelled: Price ({current_price}) above stop ({stop_level}).")
        return

    if not bypass_gates and now_ct.time() > ENTRY_CUTOFF:
        print("⏰ Execution Alert Threshold Reached: Past cutoff.")
        return

    atr_display = f"{atr_value:.2f}c" if atr_value is not None else "N/A"
    warning_block = ""
    if noise_floor_warn:
        warning_block = f"\n⚠️ <b>RISK WARNING:</b> Stop distance ({stop_distance_points:.2f}c) is thinner than 2x ATR threshold."

    msg = f"""🌾 <b>ANTI-HUNT WHEAT FILTER (v2)</b>
🕒 Time: <code>{now_ct.strftime('%H:%M:%S')} CST</code>

📈 Session Open Anchor: <code>{session_open:.2f}c</code>
💵 Current Spot Price: <code>{current_price:.2f}c</code>
📊 15m ATR Volatility: <code>{atr_display}</code>
⏱️ Bar Latency Age: <code>{staleness_min:.1f} min</code>
{warning_block}
🛡️ <b>THE PROTECTED GRID SETUP:</b>
📥 <b>ENTRY (Sell Limit):</b> <code>{entry_level:.2f}c</code>
🛑 <b>STOP-LOSS:</b> <code>{stop_level:.2f}c</code>
🎯 <b>TARGET:</b> <code>{target_level:.2f}c</code>

<i>Note: Orders mapped to 0.25c tick size. Stop sits above daily ceiling.</i>"""

    state_payload = {
        "timestamp": now_ct.isoformat(),
        "session_open": session_open,
        "current_price": current_price,
        "atr_15m": atr_value,
        "entry": entry_level,
        "stop": stop_level,
        "target": target_level,
        "warning_triggered": noise_floor_warn
    }
    with open("setup.json", "w") as f:
        json.dump(state_payload, f, indent=2)
    print("💾 Analysis state cached successfully inside setup.json.")

    send_telegram_alert(msg)
    print("🏁 Execution script run complete.")
    ping_healthcheck()


if __name__ == "__main__":
    # 🧪 KEPT ON TRUE FOR IMMEDIATE WEEKEND TESTING:
    FORCE_WEEKEND_TEST = True
    
    if FORCE_WEEKEND_TEST:
        run_anti_hunt_logic(bypass_gates=True)
    else:
        run_anti_hunt_logic(bypass_gates=False)
