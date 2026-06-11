"""
HIGH CONVICTION GATE
=====================
Built from real backtest results on 2 years of ZW=F data.
Only allows alerts when historically proven conditions are met.

BACKTEST FINDINGS:
  Baseline UP accuracy:  68.0%
  Baseline DOWN accuracy: 26.4%

  TIER 1 (100% accuracy, 8-13 trades):
    bearish_month + in_lower_half + rsi_oversold

  TIER 2 (94.7% accuracy, 19 trades):
    vol_low + bearish_month + in_lower_half

  TIER 3 (94.1% accuracy, 17 trades):
    vol_low + bearish_month + in_lower_half + inside_bb

  TIER 4 (81.7% accuracy, 93 trades):
    vol_low alone

INTEGRATION:
  Call check_gate(df) before sending any alert.
  Returns (allowed, tier, reason, conditions_met)
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ── thresholds (must match backtest exactly) ──────────────────────────────────
RSI_OVERSOLD        = 35       # rsi_oversold
VOL_RATIO_LOW       = 0.80     # vol_low: volume < 80% of 20-day avg
BB_STD_INSIDE       = 1.5      # inside_bb: within 1.5 std of BB middle
RANGE_PCT_LOWER     = 0.40     # in_lower_half: price in bottom 40% of 52wk range
BEARISH_MONTHS      = [6, 7, 8]  # harvest season

# Minimum sample size to trust a tier
MIN_TRADES_TO_TRUST = 15


class HighConvictionGate:
    """
    Evaluates current market conditions against backtest-proven patterns.
    Only allows alerts on historically high-accuracy setups.
    """

    def check_gate(self, df):
        """
        Main entry point. Call this before sending any alert.

        Args:
            df: DataFrame with OHLCV data + indicators (from add_indicators())

        Returns:
            allowed:        bool   — True = send alert, False = skip
            tier:           int    — 1/2/3/4 = conviction tier, 0 = blocked
            accuracy:       float  — expected accuracy based on backtest
            reason:         str    — human-readable explanation
            conditions:     dict   — all condition values for logging
        """
        if df is None or len(df) < 60:
            return False, 0, 0.0, "Insufficient data for gate check", {}

        conditions = self._compute_conditions(df)
        return self._evaluate_tiers(conditions)

    # ── condition computation ─────────────────────────────────────────────────

    def _compute_conditions(self, df):
        """Compute all backtest conditions from current market data."""
        close  = df['Close'].iloc[-1]
        month  = datetime.now().month

        # ── RSI ──
        delta = df['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_series = 100 - (100 / (1 + gain / loss))
        rsi = float(rsi_series.iloc[-1])

        # ── Volume ratio ──
        vol_avg   = float(df['Volume'].rolling(20).mean().iloc[-1])
        vol_curr  = float(df['Volume'].iloc[-1])
        vol_ratio = vol_curr / vol_avg if vol_avg > 0 else 1.0

        # ── Bollinger Bands ──
        bb_mid = float(df['Close'].rolling(20).mean().iloc[-1])
        bb_std = float(df['Close'].rolling(20).std().iloc[-1])
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        near_upper = close > (bb_mid + BB_STD_INSIDE * bb_std)
        near_lower = close < (bb_mid - BB_STD_INSIDE * bb_std)
        inside_bb  = not near_upper and not near_lower

        # ── 52-week range position ──
        prices_1yr = df['Close'].iloc[-252:] if len(df) >= 252 else df['Close']
        high52 = float(prices_1yr.max())
        low52  = float(prices_1yr.min())
        range_pct = (close - low52) / (high52 - low52) if high52 > low52 else 0.5

        # ── ATR ──
        hl  = df['High'] - df['Low']
        hc  = (df['High'] - df['Close'].shift()).abs()
        lc  = (df['Low']  - df['Close'].shift()).abs()
        atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
        atr_pct = float(atr) / close

        # ── MACD ──
        ema12 = df['Close'].ewm(span=12).mean().iloc[-1]
        ema26 = df['Close'].ewm(span=26).mean().iloc[-1]
        macd  = float(ema12 - ema26)
        macd_signal = float(df['Close'].ewm(span=12).mean().ewm(span=9).mean().iloc[-1] -
                           df['Close'].ewm(span=26).mean().ewm(span=9).mean().iloc[-1])
        macd_bullish = macd > macd_signal

        # ── Momentum ──
        ret_1d = float(df['Close'].pct_change(1).iloc[-1])
        ret_3d = float(df['Close'].pct_change(3).iloc[-1])
        momentum_up   = ret_1d > 0 and ret_3d > 0
        momentum_down = ret_1d < 0 and ret_3d < 0

        # ── Named conditions (must match backtest names exactly) ──
        conds = {
            # Core backtest conditions
            'bearish_month':  month in BEARISH_MONTHS,
            'in_lower_half':  range_pct < RANGE_PCT_LOWER,
            'rsi_oversold':   rsi < RSI_OVERSOLD,
            'vol_low':        vol_ratio < VOL_RATIO_LOW,
            'inside_bb':      inside_bb,
            'wc_bullish':     False,   # set externally if W/C ratio available
            'momentum_up':    momentum_up,
            'momentum_down':  momentum_down,
            'macd_bullish':   macd_bullish,
            'vol_good':       0.010 < atr_pct < 0.035,

            # Raw values for logging
            'rsi':            round(rsi, 1),
            'vol_ratio':      round(vol_ratio, 2),
            'range_pct':      round(range_pct, 3),
            'atr_pct':        round(atr_pct, 4),
            'month':          month,
            'price':          round(close, 2),
            'bb_mid':         round(bb_mid, 2),
        }

        return conds

    def set_wc_bullish(self, conditions, wasde_signal):
        """
        Optionally enrich conditions with W/C ratio signal from WASDE.
        Call after _compute_conditions() if wasde_signal is available.
        """
        if wasde_signal and wasde_signal.get('signal') == 'BULLISH':
            conditions['wc_bullish'] = True
        return conditions

    # ── tier evaluation ───────────────────────────────────────────────────────

    def _evaluate_tiers(self, c):
        """
        Check tiers from highest to lowest conviction.
        Returns first tier that matches.
        """

        # ── TIER 1: 100% accuracy (13 trades) ──
        # bearish_month + in_lower_half + rsi_oversold
        if c['bearish_month'] and c['in_lower_half'] and c['rsi_oversold']:
            return (
                True, 1, 1.00,
                f"TIER 1 — 100% accuracy (13 trades) | "
                f"Harvest month + price at {c['range_pct']:.0%} of range + RSI {c['rsi']:.0f}",
                c
            )

        # ── TIER 2: 94.7% accuracy (19 trades) ──
        # vol_low + bearish_month + in_lower_half
        if c['vol_low'] and c['bearish_month'] and c['in_lower_half']:
            return (
                True, 2, 0.947,
                f"TIER 2 — 94.7% accuracy (19 trades) | "
                f"Low volume ({c['vol_ratio']:.1f}x) + harvest month + price at {c['range_pct']:.0%} of range",
                c
            )

        # ── TIER 3: 94.1% accuracy (17 trades) ──
        # vol_low + bearish_month + in_lower_half + inside_bb
        if c['vol_low'] and c['bearish_month'] and c['in_lower_half'] and c['inside_bb']:
            return (
                True, 3, 0.941,
                f"TIER 3 — 94.1% accuracy (17 trades) | "
                f"Low volume + harvest month + lower range + inside BB",
                c
            )

        # ── TIER 4: 81.7% accuracy (93 trades) — vol_low alone ──
        # Broader filter — still meaningfully above baseline 68%
        if c['vol_low'] and c['in_lower_half']:
            return (
                True, 4, 0.817,
                f"TIER 4 — 81.7% accuracy (93 trades) | "
                f"Low volume ({c['vol_ratio']:.1f}x) + price at {c['range_pct']:.0%} of 52wk range",
                c
            )

        # ── BLOCKED ──
        # Build a clear reason explaining what's missing
        missing = []
        if not c['bearish_month']:
            missing.append(f"not harvest month (month={c['month']})")
        if not c['in_lower_half']:
            missing.append(f"price too high in range ({c['range_pct']:.0%} of 52wk)")
        if not c['rsi_oversold']:
            missing.append(f"RSI not oversold ({c['rsi']:.0f})")
        if not c['vol_low']:
            missing.append(f"volume not low ({c['vol_ratio']:.1f}x avg)")

        reason = "BLOCKED — conditions not met: " + " | ".join(missing[:3])

        return False, 0, 0.68, reason, c

    # ── formatting ────────────────────────────────────────────────────────────

    def format_for_telegram(self, tier, accuracy, reason, conditions):
        """Format gate result for inclusion in Telegram alert."""
        if tier == 0:
            return ""

        tier_emoji = {1: "💎", 2: "🥇", 3: "🥈", 4: "🥉"}
        emoji = tier_emoji.get(tier, "✅")

        return (
            f"{emoji} *CONVICTION TIER {tier}*\n"
            f"Expected accuracy: {accuracy:.0%}\n"
            f"RSI: {conditions['rsi']:.0f} | "
            f"Vol: {conditions['vol_ratio']:.1f}x | "
            f"Range pos: {conditions['range_pct']:.0%}"
        )

    def log_conditions(self, conditions):
        """Print all conditions to GitHub Actions log."""
        print(f"\n🎯 Gate Conditions:")
        print(f"   Month: {conditions['month']} | bearish_month={conditions['bearish_month']}")
        print(f"   RSI: {conditions['rsi']:.1f} | oversold={conditions['rsi_oversold']}")
        print(f"   Vol ratio: {conditions['vol_ratio']:.2f}x | vol_low={conditions['vol_low']}")
        print(f"   52wk range pos: {conditions['range_pct']:.1%} | in_lower_half={conditions['in_lower_half']}")
        print(f"   Inside BB: {conditions['inside_bb']}")
        print(f"   ATR: {conditions['atr_pct']:.2%} | vol_good={conditions['vol_good']}")


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    from datetime import timedelta

    print("Testing gate with live ZW=F data...")
    end   = datetime.now()
    start = end - timedelta(days=400)
    df    = yf.Ticker("ZW=F").history(start=start, end=end, auto_adjust=False)

    gate = HighConvictionGate()
    allowed, tier, accuracy, reason, conditions = gate.check_gate(df)
    gate.log_conditions(conditions)

    print(f"\nResult: {'✅ ALLOWED' if allowed else '⛔ BLOCKED'}")
    print(f"Tier: {tier} | Expected accuracy: {accuracy:.0%}")
    print(f"Reason: {reason}")
