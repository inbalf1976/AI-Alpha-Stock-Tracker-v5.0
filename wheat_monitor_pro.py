#!/usr/bin/env python3
"""
WHEAT MONITOR v4.0 — Main System Engine
Integrates Wheat Range Engine (WRE), Technical Indicator Pipeline,
News-Adjusted Conviction Gate, State Caching, and Telegram Alerting.
"""

import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import requests

# ── LOGGING & CONFIGURATION ──────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("WheatMonitor")

WEEKLY_CACHE_FILE = Path("weekly_cache.json")
DAILY_LOG_FILE = Path("daily_performance_log.json")


# ── INDICATORS & FEATURE ENGINEERING ─────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators for trend, momentum, volatility, and volume.
    Used by Daily Ensemble models and Conviction Gate logic.
    Note: Volume indicators relying on unstable continuous futures feeds (vol_low)
    are structurally excluded to prevent false signals.
    """
    df = df.copy()

    # Moving averages & Trend
    df['SMA5'] = df['Close'].rolling(5).mean()
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    # Returns & Momentum
    df['Ret1'] = df['Close'].pct_change(1)
    df['Ret3'] = df['Close'].pct_change(3)
    df['Ret5'] = df['Close'].pct_change(5)

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Average True Range (ATR 14)
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift(1))
    low_cp = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # Bollinger Bands (20, 2)
    df['BB_Mid'] = df['SMA20']
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Mid'] - (bb_std * 2)

    return df


# ── CONVICTION GATE ENGINE ───────────────────────────────────────────────────

class ConvictionGate:
    """
    Conviction Gate v4.0 with News Nudge & Conflict Override Penalty.
    Combines technical conviction scores with macro headline sentiment.
    """
    def __init__(
        self,
        news_bullish_threshold: float = 0.60,
        news_bearish_threshold: float = -0.60,
        nudge_boost: float = 0.03,
        conflict_penalty: float = 0.05
    ):
        self.news_bullish_thresh = news_bullish_threshold
        self.news_bearish_thresh = news_bearish_threshold
        self.nudge_boost = nudge_boost
        self.conflict_penalty = conflict_penalty

    def evaluate_conviction(
        self,
        tech_score: float,          # Base technical score (0.00 to 1.00)
        tech_direction: str,        # 'LONG' or 'SHORT'
        news_score: float,          # News sentiment score (-1.00 to +1.00)
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Evaluates final conviction score, tier classification, and divergence state.
        """
        adjusted_score = tech_score
        divergence_detected = False
        divergence_reason = None
        applied_adjustment = 0.0

        news_bullish = news_score >= self.news_bullish_thresh
        news_bearish = news_score <= self.news_bearish_thresh

        # 1. Supplemental Nudge (High Confidence Alignment)
        if tech_direction == 'LONG' and news_bullish:
            applied_adjustment = self.nudge_boost
            adjusted_score += applied_adjustment
            logger.info(f"🟢 [Nudge] Bullish alignment boost applied (+{self.nudge_boost:.2f}).")

        elif tech_direction == 'SHORT' and news_bearish:
            applied_adjustment = self.nudge_boost
            adjusted_score += applied_adjustment
            logger.info(f"🟢 [Nudge] Bearish alignment boost applied (+{self.nudge_boost:.2f}).")

        # 2. Conflict Override / Penalty (Sharply Contradictory Macro News)
        elif (tech_direction == 'LONG' and news_bearish) or (tech_direction == 'SHORT' and news_bullish):
            applied_adjustment = -self.conflict_penalty
            adjusted_score -= self.conflict_penalty
            divergence_detected = True
            
            direction_conflict = (
                "Bullish Technicals vs. Bearish News" if tech_direction == 'LONG' 
                else "Bearish Technicals vs. Bullish News"
            )
            divergence_reason = f"DIVERGENCE_CAUTION: {direction_conflict} (News Score: {news_score:+.2f})"
            
            logger.warning(
                f"⚠️ [Divergence Alert] {divergence_reason}. "
                f"Applied penalty (-{self.conflict_penalty:.2f}). Score reduced to {adjusted_score:.2f}."
            )

        final_score = float(min(max(adjusted_score, 0.0), 1.0))

        # 3. Tier Classification
        if final_score >= 0.75:
            tier = "TIER_1"
        elif final_score >= 0.55:
            tier = "TIER_2"
        else:
            tier = "TIER_3"

        metadata = {
            "base_tech_score": round(tech_score, 3),
            "news_score": round(news_score, 3),
            "applied_adjustment": round(applied_adjustment, 3),
            "final_score": round(final_score, 3),
            "divergence_detected": divergence_detected,
            "divergence_reason": divergence_reason,
            "status_flag": "CAUTION" if divergence_detected else "NORMAL"
        }

        return final_score, tier, metadata


# ── WHEAT RANGE ENGINE ───────────────────────────────────────────────────────

class WheatRangeEngine:
    """
    Wheat Range Engine (WRE) v4.0
    Generates weekly range forecasts, calculates conviction-weighted 
    volatility buffers, and constructs trade execution levels.
    """
    def __init__(self, atr_multiplier: float = 1.8, min_rr_ratio: float = 1.5):
        self.atr_multiplier = atr_multiplier
        self.min_rr_ratio = min_rr_ratio

    def predict_next_week(
        self,
        df: pd.DataFrame,
        current_price: float,
        cost_floor_cents: float = 490.0,
        forced_direction: Optional[str] = None,
        daily_direction_hint: Optional[str] = None,
        backtest_tier: str = "TIER_1",
        backtest_accuracy: float = 0.65,
        news_signal: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculates projected weekly range bounds and trade execution parameters.
        """
        if 'ATR' not in df.columns or 'SMA20' not in df.columns:
            df = add_indicators(df)

        last_row = df.iloc[-1]
        atr = last_row['ATR'] if not np.isnan(last_row['ATR']) else current_price * 0.025
        sma20 = last_row['SMA20'] if not np.isnan(last_row['SMA20']) else current_price

        # 1. Determine Bias Direction
        if forced_direction in ['LONG', 'SHORT']:
            bias = forced_direction
        elif daily_direction_hint in ['LONG', 'SHORT']:
            bias = daily_direction_hint
        else:
            bias = 'LONG' if current_price >= sma20 else 'SHORT'

        # 2. Adjust Range Volatility Buffer
        tier_weights = {"TIER_1": 1.2, "TIER_2": 1.0, "TIER_3": 0.8}
        tier_factor = tier_weights.get(backtest_tier, 1.0)
        news_factor = 1.0 + (np.clip(news_signal, -1.0, 1.0) * 0.15)
        range_buffer = atr * self.atr_multiplier * tier_factor * news_factor

        # 3. Derive Weekly Range Boundaries
        proj_high = round(max(current_price + range_buffer, current_price * 1.01), 2)
        proj_low = round(min(current_price - range_buffer, current_price * 0.99), 2)

        # Apply Hard Physical Cost Floor Guardrail
        if proj_low < cost_floor_cents:
            proj_low = float(cost_floor_cents)

        # 4. Construct Geometry & Execution Points
        if bias == 'LONG':
            entry = round(current_price, 2)
            stop = round(max(proj_low, entry - (atr * 1.2)), 2)
            target = round(proj_high, 2)
            
            if entry - stop < 4.0:
                stop = entry - 4.0
                
            risk = entry - stop
            reward = target - entry
        else:  # SHORT
            entry = round(current_price, 2)
            stop = round(entry + (atr * 1.2), 2)
            target = round(proj_low, 2)
            
            if stop - entry < 4.0:
                stop = entry + 4.0
                
            risk = stop - entry
            reward = entry - target

        rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

        return {
            "bias": bias,
            "current_price": current_price,
            "proj_high": proj_high,
            "proj_low": proj_low,
            "entry": entry,
            "stop_loss": stop,
            "target": target,
            "risk_reward": rr_ratio,
            "atr": round(atr, 2),
            "tier": backtest_tier,
            "accuracy": round(backtest_accuracy * 100, 1)
        }


# ── CACHING & PERFORMANCE LOGGING ───────────────────────────────────────────

def log_daily_performance(iso_year: int, iso_week: int, current_price: float, weekly: Dict[str, Any]):
    """Tracks daily price relative to the frozen weekly projected range."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "iso_year": iso_year,
        "iso_week": iso_week,
        "current_price": current_price,
        "proj_high": weekly.get("proj_high"),
        "proj_low": weekly.get("proj_low"),
        "bias": weekly.get("bias")
    }
    try:
        data = []
        if DAILY_LOG_FILE.exists():
            data = json.loads(DAILY_LOG_FILE.read_text())
        data.append(log_entry)
        DAILY_LOG_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"Failed to write daily performance log: {e}")


def get_frozen_weekly_plan(
    df: pd.DataFrame,
    current_price: float,
    cost_floor_cents: float = 490.0,
    daily_direction: Optional[str] = None,
    break_type: Optional[str] = None,
    backtest_tier: str = "TIER_1",
    backtest_accuracy: float = 0.65,
    news_signal: float = 0.0
) -> Dict[str, Any]:
    """
    Retrieves or updates the weekly plan cached by ISO week.
    Flips directional bias automatically if a stop loss break occurs.
    """
    now = datetime.datetime.now()
    iso_year, iso_week, _ = now.isocalendar()
    wre = WheatRangeEngine()

    cached_data = None
    if WEEKLY_CACHE_FILE.exists():
        try:
            cached_data = json.loads(WEEKLY_CACHE_FILE.read_text())
        except Exception as e:
            logger.error(f"Failed to read weekly cache: {e}")

    if cached_data and cached_data.get('iso_year') == iso_year and cached_data.get('iso_week') == iso_week:
        old_weekly = cached_data.get('weekly', {})
        if break_type == 'stop':
            new_final_call = 'SHORT' if old_weekly.get('bias') == 'LONG' else 'LONG'
            logger.warning(f"🔄 Stop-loss break detected! Flipping weekly bias to {new_final_call}.")
            weekly = wre.predict_next_week(
                df, current_price, cost_floor_cents,
                forced_direction=new_final_call,
                backtest_tier=backtest_tier,
                backtest_accuracy=backtest_accuracy,
                news_signal=news_signal
            )
        else:
            weekly = old_weekly
    else:
        # Fresh week initialization
        weekly = wre.predict_next_week(
            df, current_price, cost_floor_cents,
            forced_direction=None,
            daily_direction_hint=daily_direction,
            backtest_tier=backtest_tier,
            backtest_accuracy=backtest_accuracy,
            news_signal=news_signal
        )

    # Freeze/update cache state
    try:
        WEEKLY_CACHE_FILE.write_text(json.dumps({
            'iso_year': iso_year,
            'iso_week': iso_week,
            'weekly': weekly
        }, indent=2))
    except Exception as e:
        logger.error(f"Failed to save weekly cache: {e}")

    log_daily_performance(iso_year, iso_week, current_price, weekly)
    return weekly


# ── TELEGRAM ALERTING PIPELINE ───────────────────────────────────────────────

def format_telegram_alert(
    weekly_plan: Dict[str, Any],
    symbol: str = "ZW=F (CBOT Wheat)",
    alert_type: str = "WEEKLY_PLAN",
    status_flag: str = "NORMAL"
) -> str:
    """
    Formats trading plans into clean HTML messages for Telegram.
    Includes CAUTION badges when divergence is detected.
    """
    bias = weekly_plan.get("bias", "NEUTRAL")
    direction_emoji = "🟢" if bias == "LONG" else "🔴"
    
    header_title = {
        "WEEKLY_PLAN": "🌾 <b>WHEAT MONITOR v4.0 — WEEKLY PLAN</b>",
        "DIRECTION_FLIP": "⚠️ <b>DIRECTIONAL FLIP TRIGGERED</b>",
        "DAILY_UPDATE": "📊 <b>DAILY REGIME UPDATE</b>"
    }.get(alert_type, "🌾 <b>WHEAT MONITOR ALERT</b>")

    entry = weekly_plan.get("entry", 0.0)
    stop = weekly_plan.get("stop_loss", 0.0)
    target = weekly_plan.get("target", 0.0)
    rr = weekly_plan.get("risk_reward", 0.0)
    
    risk_pts = round(abs(entry - stop), 2)
    reward_pts = round(abs(target - entry), 2)

    msg = f"{header_title}\n"
    if status_flag == "CAUTION":
        msg += "⚠️ <b>STATUS: DIVERGENCE / CAUTION DETECTED</b>\n"
    
    msg += f"<b>Asset:</b> {requests.utils.quote(symbol) if False else symbol}\n"
    msg += f"<b>Bias:</b> {direction_emoji} <b>{bias}</b>\n\n"

    msg += f"<b>📍 Execution Levels (USd/Bu):</b>\n"
    msg += f"  • <b>Entry:</b> <code>{entry:.2f}</code>\n"
    msg += f"  • <b>Stop Loss:</b> <code>{stop:.2f}</code> (Risk: {risk_pts:.2f}¢)\n"
    msg += f"  • <b>Take Profit:</b> <code>{target:.2f}</code> (Reward: {reward_pts:.2f}¢)\n"
    msg += f"  • <b>R:R Ratio:</b> <code>{rr:.2f}</code>\n\n"

    msg += f"<b>📐 Forecasted Weekly Range:</b>\n"
    msg += f"  • <b>High Barrier:</b> <code>{weekly_plan.get('proj_high', 0.0):.2f}</code>\n"
    msg += f"  • <b>Low Floor:</b> <code>{weekly_plan.get('proj_low', 0.0):.2f}</code>\n"
    msg += f"  • <b>ATR (14):</b> <code>{weekly_plan.get('atr', 0.0):.2f}</code>\n\n"

    msg += f"<b>⚙️ Model Status:</b>\n"
    msg += f"  • <b>Backtest Tier:</b> {weekly_plan.get('tier', 'N/A')}\n"
    msg += f"  • <b>Historical Accuracy:</b> {weekly_plan.get('accuracy', 0.0)}%\n"
    
    return msg


def send_telegram_alert(
    message_text: str,
    bot_token: str,
    chat_id: str
) -> bool:
    """Dispatches formatted HTML alerts to Telegram via HTTP API."""
    if not bot_token or not chat_id:
        logger.error("Telegram dispatch failed: Missing BOT_TOKEN or CHAT_ID.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        res_data = response.json()
        if res_data.get("ok"):
            logger.info("✅ Telegram alert sent successfully.")
            return True
        else:
            logger.error(f"❌ Telegram API Error: {res_data.get('description')}")
            return False
    except Exception as e:
        logger.error(f"❌ Network error sending Telegram alert: {e}")
        return False


# ── EXECUTION DEMO / ENTRY POINT ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Initializing WHEAT MONITOR v4.0 Pipeline Test...")

    # 1. Generate synthetic OHLC data for demonstration
    dates = pd.date_range(end=datetime.datetime.now(), periods=100)
    np.random.seed(42)
    close_prices = 530.0 + np.cumsum(np.random.randn(100) * 3.5)
    
    df_raw = pd.DataFrame({
        'Open': close_prices - 1.0,
        'High': close_prices + 4.0,
        'Low': close_prices - 4.0,
        'Close': close_prices,
        'Volume': np.random.randint(1000, 50000, size=100)
    }, index=dates)

    # 2. Add indicators
    df_proc = add_indicators(df_raw)
    current_live_price = float(df_proc['Close'].iloc[-1])

    # 3. Evaluate Conviction Gate (with technicals vs news divergence)
    gate = ConvictionGate(nudge_boost=0.03, conflict_penalty=0.05)
    tech_score = 0.73
    tech_direction = 'LONG'
    news_score = -0.70  # Contradictory macro headline

    final_score, tier, meta = gate.evaluate_conviction(
        tech_score=tech_score,
        tech_direction=tech_direction,
        news_score=news_score
    )

    logger.info(f"Conviction Result: Score={final_score}, Tier={tier}, Meta={meta}")

    # 4. Generate Frozen Weekly Plan
    plan = get_frozen_weekly_plan(
        df=df_proc,
        current_price=current_live_price,
        cost_floor_cents=490.0,
        daily_direction=tech_direction,
        backtest_tier=tier,
        backtest_accuracy=0.68,
        news_signal=news_score
    )

    # 5. Format Telegram Alert
    alert_msg = format_telegram_alert(
        weekly_plan=plan,
        alert_type="WEEKLY_PLAN",
        status_flag=meta["status_flag"]
    )

    print("\n--- GENERATED TELEGRAM ALERT PREVIEW ---")
    print(alert_msg)
