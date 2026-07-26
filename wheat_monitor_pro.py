import os
import logging
import datetime
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("WheatMonitor")

# ==========================================
# 1. INDICATORS & MARKET DATA PROCESSING
# ==========================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches OHLC DataFrame with technical indicators."""
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI_14'] = calculate_rsi(df['Close'], period=14)
    df['ATR_14'] = calculate_atr(df, period=14)
    return df

# ==========================================
# 2. CONVICTION GATE ENGINE
# ==========================================

class ConvictionGate:
    def __init__(self, nudge_boost: float = 0.03, conflict_penalty: float = 0.05):
        self.nudge_boost = nudge_boost
        self.conflict_penalty = conflict_penalty

    def evaluate_conviction(
        self,
        tech_score: float,
        tech_direction: str,
        news_score: float
    ) -> tuple[float, str, dict]:
        """
        Evaluates technical score against macro news sentiment to determine final tier & adjustments.
        """
        applied_adjustment = 0.0
        divergence_detected = False
        divergence_reason = ""
        status_flag = "OK"

        # Neutral news -> small alignment boost
        if abs(news_score) <= 0.15:
            applied_adjustment += self.nudge_boost

        # Conflicting sentiment check
        is_bullish_tech = tech_direction.upper() == 'LONG'
        is_bearish_news = news_score < -0.30
        is_bearish_tech = tech_direction.upper() == 'SHORT'
        is_bullish_news = news_score > 0.30

        if (is_bullish_tech and is_bearish_news) or (is_bearish_tech and is_bullish_news):
            applied_adjustment -= self.conflict_penalty
            divergence_detected = True
            status_flag = "CAUTION"
            news_type = "Bearish" if is_bearish_news else "Bullish"
            tech_type = "Bullish" if is_bullish_tech else "Bearish"
            divergence_reason = (
                f"DIVERGENCE_CAUTION: {tech_type} Technicals vs. "
                f"{news_type} News (News Score: {news_score:.2f})"
            )
            logger.warning(f"⚠️ [Divergence Alert] {divergence_reason}. Applied penalty (-{self.conflict_penalty}).")

        # Precision float evaluation and normalization
        raw_score = tech_score + applied_adjustment
        final_score = round(max(0.0, min(1.0, raw_score)), 2)

        # Classification Tier Assignment
        if final_score >= 0.80:
            tier = "TIER_1"
        elif final_score >= 0.65:
            tier = "TIER_2"
        elif final_score >= 0.50:
            tier = "TIER_3"
        else:
            tier = "NO_TRADE"

        meta = {
            'base_tech_score': tech_score,
            'news_score': news_score,
            'applied_adjustment': applied_adjustment,
            'final_score': final_score,
            'divergence_detected': divergence_detected,
            'divergence_reason': divergence_reason,
            'status_flag': status_flag
        }

        logger.info(f"Conviction Result: Score={raw_score}, Tier={tier}, Meta={meta}")
        return final_score, tier, meta

# ==========================================
# 3. FROZEN WEEKLY PLAN GENERATOR
# ==========================================

def get_frozen_weekly_plan(
    df: pd.DataFrame,
    current_price: float,
    cost_floor_cents: float = 490.0,
    daily_direction: str = 'LONG',
    backtest_tier: str = 'TIER_2',
    backtest_accuracy: float = 0.68,
    news_signal: float = -0.70
) -> dict:
    """Calculates entry, stop loss, take profit, and forecasted range limits."""
    atr_val = float(df['ATR_14'].dropna().iloc[-1]) if 'ATR_14' in df and not df['ATR_14'].dropna().empty else 8.0

    if daily_direction.upper() == 'LONG':
        # Hard floor protection check
        low_floor = max(cost_floor_cents, current_price - (1.2 * atr_val))
        stop_loss = max(cost_floor_cents - 0.35, current_price - (0.5 * atr_val))
        entry = current_price
        take_profit = current_price + (1.6 * atr_val)
        high_barrier = take_profit
    else:
        entry = current_price
        stop_loss = current_price + (0.5 * atr_val)
        take_profit = max(cost_floor_cents, current_price - (1.6 * atr_val))
        low_floor = take_profit
        high_barrier = current_price + (1.2 * atr_val)

    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    rr_ratio = reward / risk if risk > 0 else 0.0

    return {
        'asset': 'ZW=F (CBOT Wheat)',
        'bias': daily_direction.upper(),
        'entry': round(entry, 2),
        'stop_loss': round(stop_loss, 2),
        'take_profit': round(take_profit, 2),
        'risk': round(risk, 2),
        'reward': round(reward, 2),
        'rr_ratio': round(rr_ratio, 2),
        'high_barrier': round(high_barrier, 2),
        'low_floor': round(low_floor, 2),
        'atr_14': round(atr_val, 2),
        'tier': backtest_tier,
        'accuracy': round(backtest_accuracy * 100, 1),
        'news_signal': news_signal
    }

# ==========================================
# 4. TELEGRAM FORMATTING & DISPATCH
# ==========================================

def format_telegram_alert(weekly_plan: dict, alert_type: str = "WEEKLY_PLAN", status_flag: str = "OK") -> str:
    """Formats market plan payload into styled HTML for Telegram."""
    status_header = ""
    if status_flag == "CAUTION":
        status_header = "⚠️ <b>STATUS: DIVERGENCE / CAUTION DETECTED</b>\n\n"

    bias_emoji = "🟢" if weekly_plan['bias'] == 'LONG' else "🔴"

    msg = (
        f"🌾 <b>WHEAT MONITOR v4.0 — WEEKLY PLAN</b>\n\n"
        f"{status_header}"
        f"<b>Asset:</b> {weekly_plan['asset']}\n"
        f"<b>Bias:</b> {bias_emoji} <b>{weekly_plan['bias']}</b>\n\n"
        f"<b>📍 Execution Levels (USd/Bu):</b>\n"
        f"  • <b>Entry:</b> <code>{weekly_plan['entry']:.2f}</code>\n"
        f"  • <b>Stop Loss:</b> <code>{weekly_plan['stop_loss']:.2f}</code> (Risk: {weekly_plan['risk']:.2f}¢)\n"
        f"  • <b>Take Profit:</b> <code>{weekly_plan['take_profit']:.2f}</code> (Reward: {weekly_plan['reward']:.2f}¢)\n"
        f"  • <b>R:R Ratio:</b> <code>{weekly_plan['rr_ratio']:.2f}</code>\n\n"
        f"<b>📐 Forecasted Weekly Range:</b>\n"
        f"  • <b>High Barrier:</b> <code>{weekly_plan['high_barrier']:.2f}</code>\n"
        f"  • <b>Low Floor:</b> <code>{weekly_plan['low_floor']:.2f}</code>\n"
        f"  • <b>ATR (14):</b> <code>{weekly_plan['atr_14']:.2f}</code>\n\n"
        f"<b>⚙️ Model Status:</b>\n"
        f"  • <b>Backtest Tier:</b> {weekly_plan['tier']}\n"
        f"  • <b>Historical Accuracy:</b> {weekly_plan['accuracy']}%"
    )
    return msg

def send_telegram_alert(message_text: str, bot_token: str, chat_id: str) -> bool:
    """Sends the formatted alert to Telegram via HTTP Bot API."""
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
        if response.status_code == 200 and res_data.get("ok"):
            logger.info("✅ Telegram alert dispatched successfully!")
            return True
        else:
            logger.error(f"❌ Telegram delivery failed: {res_data.get('description', 'Unknown Error')}")
            return False
    except Exception as e:
        logger.error(f"❌ Exception occurred during Telegram dispatch: {e}")
        return False

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    logger.info("Initializing WHEAT MONITOR v4.0 Pipeline Test...")

    # Load credentials from environment variables or .env file
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # 1. Generate Synthetic Synthetic OHLC Data
    dates = pd.date_range(end=datetime.datetime.now(), periods=100)
    np.random.seed(42)
    close_prices = 490.0 + np.cumsum(np.random.randn(100) * 0.5)

    df_raw = pd.DataFrame({
        'Open': close_prices - 0.5,
        'High': close_prices + 2.0,
        'Low': close_prices - 2.0,
        'Close': close_prices,
        'Volume': np.random.randint(1000, 50000, size=100)
    }, index=dates)

    # Force current price to test scenario target
    df_raw.iloc[-1, df_raw.columns.get_loc('Close')] = 493.65

    # 2. Add indicators
    df_proc = add_indicators(df_raw)
    current_live_price = float(df_proc['Close'].iloc[-1])

    # 3. Evaluate Conviction Gate
    gate = ConvictionGate(nudge_boost=0.03, conflict_penalty=0.05)
    tech_score = 0.73
    tech_direction = 'LONG'
    news_score = -0.70

    final_score, tier, meta = gate.evaluate_conviction(
        tech_score=tech_score,
        tech_direction=tech_direction,
        news_score=news_score
    )

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
    print("----------------------------------------\n")

    # 6. Dispatch Alert to Telegram
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        logger.info("Attempting Telegram dispatch...")
        send_telegram_alert(
            message_text=alert_msg,
            bot_token=TELEGRAM_BOT_TOKEN,
            chat_id=TELEGRAM_CHAT_ID
        )
    else:
        logger.warning(
            "⚠️ Telegram dispatch skipped: Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. "
            "Set them in your environment or inside a .env file."
        )
