"""
anti_hunt_filter.py  (v2 + manual test mode)
================================================
Institutional open anti-stop-hunting Short filter for Chicago SRW Wheat (ZW=F).

NORMAL MODE:
    Runs during the safe institutional window
    (8:45 AM - 12:30 PM America/Chicago, weekdays only).

MANUAL MODE:
    Run with:
        python anti_hunt_filter.py --manual

    Manual mode bypasses:
        - CME holiday check
        - Trading-window check
        - 11:30 CT entry cutoff

    It still:
        - Fetches real ZW=F market data
        - Calculates the setup
        - Performs data-staleness checks
        - Writes setup.json
        - Can send Telegram alerts

Dependencies:
    yfinance
    pandas
    requests

Environment:
    TELEGRAM_BOT_TOKEN   optional for manual/local testing
    TELEGRAM_CHAT_ID     optional for manual/local testing
    HEALTHCHECK_URL      optional
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


# ---------------------------------------------------------------------------
# Manual mode
# ---------------------------------------------------------------------------
#
# Normal automated run:
#
#     python anti_hunt_filter.py
#
# Manual/local test:
#
#     python anti_hunt_filter.py --manual
#
# Manual mode intentionally bypasses the time/holiday restrictions while
# keeping the actual market-data calculation intact.
# ---------------------------------------------------------------------------

MANUAL_MODE = "--manual" in sys.argv


# ---------------------------------------------------------------------------
# Anti-hunt geometry
# ---------------------------------------------------------------------------

ENTRY_MULT = 0.994
STOP_MULT = 1.012
TARGET_MULT = 0.960


# ---------------------------------------------------------------------------
# CME full-day closures observed for grain futures.
# Update yearly.
# ---------------------------------------------------------------------------

HOLIDAYS = {
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),

    date(2027, 1, 1),
    date(2027, 1, 18),
    date(2027, 2, 15),
    date(2027, 3, 26),
    date(2027, 5, 31),
    date(2027, 6, 18),
    date(2027, 7, 5),
    date(2027, 9, 6),
    date(2027, 11, 25),
    date(2027, 12, 24),
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def ping_healthcheck() -> None:
    """
    Dead-man switch:
    A monitoring service can alert us if this never fires.
    """

    if not HEALTHCHECK_URL:
        return

    try:
        requests.get(HEALTHCHECK_URL, timeout=10)

    except requests.RequestException as exc:
        print(
            f"Healthcheck ping failed (non-fatal): {exc}",
            file=sys.stderr
        )


def round_tick(price: float, tick: float = TICK_SIZE) -> float:
    """
    Round to the exchange tick so levels are actually fillable.
    """

    return round(round(price / tick) * tick, 4)


def check_time_window() -> bool:
    """
    True only if current America/Chicago time is a weekday between
    8:45 AM and 12:30 PM.
    """

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
    """
    Newer yfinance versions can return MultiIndex columns.
    Flatten them to normal column names.
    """

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def fetch_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull 2 days of 15-minute bars and 5 days of daily bars for ZW=F.

    Uses retry/backoff because Yahoo can intermittently block requests.
    """

    intraday = None
    daily = None

    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):

        try:

            intraday = _flatten(
                yf.download(
                    TICKER,
                    period="2d",
                    interval="15m",
                    auto_adjust=False,
                    progress=False,
                )
            )

            daily = _flatten(
                yf.download(
                    TICKER,
                    period="5d",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                )
            )

        except Exception as exc:

            print(
                f"Fetch attempt {attempt} raised: {exc}",
                file=sys.stderr
            )

        if (
            intraday is not None
            and daily is not None
            and not intraday.empty
            and not daily.empty
        ):
            return intraday, daily

        # Don't wait after the final attempt.
        if attempt < MAX_FETCH_ATTEMPTS:
            wait_seconds = 10 * attempt

            print(
                f"No usable data. Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise ValueError(
        "yfinance returned no usable data after "
        f"{MAX_FETCH_ATTEMPTS} attempts "
        "(holiday, blockage, or outage)."
    )


def resolve_session_open(
    intraday: pd.DataFrame,
    daily: pd.DataFrame
) -> float:
    """
    Anchor:

    1. First 15m candle of the CURRENT Chicago session.
    2. Falls back to the latest daily bar's open when the intraday
       series doesn't yet contain today's session.
    """

    now_ct = datetime.now(CHICAGO_TZ)

    idx = intraday.index

    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    todays_bars = intraday[
        idx.tz_convert(CHICAGO_TZ).date == now_ct.date()
    ]

    if not todays_bars.empty:
        return float(todays_bars["Open"].iloc[0])

    return float(daily["Open"].iloc[-1])


def compute_atr(intraday: pd.DataFrame) -> float | None:
    """
    ATR(PERIOD) on 15m bars.
    Returns None if there is not enough history.
    """

    if len(intraday) < ATR_PERIOD + 1:
        return None

    high = intraday["High"]
    low = intraday["Low"]
    close = intraday["Close"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return float(
        tr.rolling(ATR_PERIOD).mean().iloc[-1]
    )


def check_staleness(intraday: pd.DataFrame) -> float:
    """
    Return age of the last 15m bar in minutes.
    """

    idx = intraday.index

    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    last_bar_ct = idx[-1].tz_convert(CHICAGO_TZ)

    return (
        datetime.now(CHICAGO_TZ) - last_bar_ct
    ).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Telegram alerting
# ---------------------------------------------------------------------------

def send_telegram_alert(text: str) -> bool:
    """
    POST an HTML message to the Telegram Bot API.

    If Telegram credentials are missing, print the alert instead.

    Analysis success does NOT depend on Telegram delivery.
    """

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:

        print(
            "Telegram credentials missing — "
            "alert printed to stdout instead."
        )

        print(text)

        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

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

        print(
            f"Telegram delivery failed (non-fatal): {exc}",
            file=sys.stderr
        )

        print("----- UNDELIVERED ALERT -----")
        print(text)

        return False


# ---------------------------------------------------------------------------
# Setup construction
# ---------------------------------------------------------------------------

def build_setup(
    daily_open: float,
    current_price: float,
    atr: float | None,
) -> dict:

    stop = round_tick(
        daily_open * STOP_MULT
    )

    entry = round_tick(
        daily_open * ENTRY_MULT
    )

    target = round_tick(
        daily_open * TARGET_MULT
    )

    risk = round(
        stop - entry,
        4
    )

    reward = round(
        entry - target,
        4
    )

    return {
        "ticker": TICKER,

        "manual_mode": MANUAL_MODE,

        "timestamp_ct": datetime.now(
            CHICAGO_TZ
        ).isoformat(),

        "valid_until_ct": datetime.combine(
            datetime.now(CHICAGO_TZ).date(),
            WINDOW_CLOSE,
            tzinfo=CHICAGO_TZ,
        ).isoformat(),

        "daily_open": round(
            daily_open,
            4
        ),

        "current_price": round(
            current_price,
            4
        ),

        "entry": entry,

        "stop": stop,

        "target": target,

        "risk": risk,

        "reward": reward,

        "rr": round(
            reward / risk,
            2
        ) if risk > 0 else 0.0,

        "atr_15m": round(
            atr,
            4
        ) if atr else None,

        "stop_atr_multiple": round(
            risk / atr,
            2
        ) if (
            atr
            and atr > 0
        ) else None,

        "vol_warning": bool(
            atr
            and risk < ATR_STOP_MIN_MULT * atr
        ),

        "invalidated": current_price > stop,
    }


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

def format_message(setup: dict) -> str:

    mode_label = (
        "MANUAL TEST"
        if MANUAL_MODE
        else "AUTOMATED"
    )

    lines = [

        f"⛓ <b>[ZW=F] Anti-Hunt Short Setup — "
        f"Institutional Open ({mode_label})</b>",

        "━━━━━━━━━━━━━━━━━━━━",

        (
            f"📅 Session: "
            f"{datetime.now(CHICAGO_TZ).strftime('%A %Y-%m-%d %H:%M %Z')}"
        ),

        f"🔵 Daily Open (Anchor): "
        f"<b>{setup['daily_open']}</b>",

        f"📊 Current Price: "
        f"<code>{setup['current_price']}</code>",

        "━━━━━━━━━━━━━━━━━━━━",

        "<b>Protected Setup | Short Bias</b>",

        f"🔻 ENTRY (Sell Limit): "
        f"<code>{setup['entry']}</code>",

        f"🛑 STOP-LOSS: "
        f"<code>{setup['stop']}</code>",

        f"🎯 TARGET: "
        f"<code>{setup['target']}</code>",

        "━━━━━━━━━━━━━━━━━━━━",

        (
            f"Risk: <code>{setup['risk']}</code> | "
            f"Reward: <code>{setup['reward']}</code> | "
            f"R:R ≈ <code>{setup['rr']}</code>"
        ),
    ]

    if setup["stop_atr_multiple"] is not None:

        lines.append(
            f"Stop distance: "
            f"{setup['stop_atr_multiple']}x ATR(16, 15m) "
            f"({setup['atr_15m']})"
        )

    if setup["vol_warning"]:

        lines.append(
            "⚠️ <b>VOL WARNING:</b> "
            "stop inside 2x ATR noise band — "
            "size down or stand aside."
        )

    lines.append(
        "⏳ Levels valid until 12:30 CT today."
    )

    lines.append(
        "<i>Geometry engineered outside HFT sweep zones. "
        "CFD fills will differ from exchange prices — "
        "mind the spread.</i>"
    )

    if MANUAL_MODE:

        lines.append(
            "🧪 <b>MANUAL TEST MODE:</b> "
            "normal time/holiday restrictions bypassed."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:

    now_ct = datetime.now(CHICAGO_TZ)

    if MANUAL_MODE:

        print(
            "=================================================="
        )

        print(
            "MANUAL TEST MODE"
        )

        print(
            "Normal trading-window restrictions are bypassed."
        )

        print(
            "Real ZW=F market data will still be fetched."
        )

        print(
            "=================================================="
        )

    # -----------------------------------------------------------------------
    # Dead-man switch
    # -----------------------------------------------------------------------
    #
    # This remains enabled in both normal and manual mode.
    #
    ping_healthcheck()

    # -----------------------------------------------------------------------
    # CME holiday check
    # -----------------------------------------------------------------------

    if not MANUAL_MODE:

        if is_cme_holiday(now_ct.date()):

            print(
                f"{now_ct.date()} is a CME holiday. "
                f"Market closed — aborting."
            )

            return 0

    # -----------------------------------------------------------------------
    # Trading window check
    # -----------------------------------------------------------------------

    if not MANUAL_MODE:

        if not check_time_window():

            print(
                f"[{now_ct.strftime('%Y-%m-%d %H:%M %Z')}] "
                f"Outside the safe institutional window "
                f"(Mon-Fri, 08:45-12:30 America/Chicago). "
                f"Aborting gracefully."
            )

            return 0

    # -----------------------------------------------------------------------
    # Entry cutoff
    # -----------------------------------------------------------------------

    if not MANUAL_MODE:

        if now_ct.time() > ENTRY_CUTOFF:

            print(
                f"Past the "
                f"{ENTRY_CUTOFF.strftime('%H:%M')} "
                f"entry cutoff — "
                f"no fresh setups this late in the window. "
                f"Aborting."
            )

            return 0

    # -----------------------------------------------------------------------
    # Fetch market data
    # -----------------------------------------------------------------------

    print(
        "Fetching ZW=F market data..."
    )

    try:

        intraday, daily = fetch_market_data()

    except ValueError as exc:

        print(
            f"Data error: {exc}",
            file=sys.stderr
        )

        return 1

    # -----------------------------------------------------------------------
    # Staleness check
    # -----------------------------------------------------------------------

    age = check_staleness(intraday)

    if age > MAX_DATA_AGE_MIN:

        print(
            f"WARNING: last 15m bar is "
            f"{age:.0f} min old — "
            f"alert will be flagged as stale.",
            file=sys.stderr
        )

    # -----------------------------------------------------------------------
    # Resolve market values
    # -----------------------------------------------------------------------

    daily_open = resolve_session_open(
        intraday,
        daily,
    )

    current_price = float(
        intraday["Close"].iloc[-1]
    )

    atr = compute_atr(
        intraday
    )

    # -----------------------------------------------------------------------
    # Build setup
    # -----------------------------------------------------------------------

    setup = build_setup(
        daily_open,
        current_price,
        atr,
    )

    if age > MAX_DATA_AGE_MIN:

        setup["stale_data_min"] = round(
            age,
            1
        )

    # -----------------------------------------------------------------------
    # Print setup
    # -----------------------------------------------------------------------

    print(
        json.dumps(
            setup,
            indent=2
        )
    )

    # -----------------------------------------------------------------------
    # Invalidated setup
    # -----------------------------------------------------------------------

    if setup["invalidated"]:

        print(
            f"Price {current_price} already above "
            f"stop {setup['stop']} — "
            f"setup invalidated. Skipping alert."
        )

        return 0

    # -----------------------------------------------------------------------
    # Persist setup.json
    # -----------------------------------------------------------------------

    try:

        with open(
            "setup.json",
            "w"
        ) as fh:

            json.dump(
                setup,
                fh,
                indent=2
            )

    except OSError as exc:

        print(
            f"Could not write setup.json "
            f"(non-fatal): {exc}",
            file=sys.stderr
        )

    # -----------------------------------------------------------------------
    # Telegram alert
    # -----------------------------------------------------------------------

    send_telegram_alert(
        format_message(setup)
    )

    return 0


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
```
