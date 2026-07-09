"""
HIGH CONVICTION GATE
=====================
REBUILT 2026-07-09 to match the real, holdout-validated numbers now
used in wheat_monitor_pro.py's ConvictionGate class.

WHY THIS CHANGED:
  The previous version of this file claimed:
    TIER 1: 100% accuracy (13 trades) — bearish_month + in_lower_half + rsi_oversold
    TIER 2: 94.7% accuracy (19 trades)
    TIER 3: 94.1% accuracy (17 trades)
    TIER 4: 81.7% accuracy (93 trades) — vol_low alone

  These numbers came from searching hundreds of condition combinations
  against a single 2-year dataset and reporting the best result found —
  a textbook overfitting setup. When tested against a holdout period
  those combos were never fitted to (see backtest.py, backtest_results.json),
  the results were:
    - bearish_month + in_lower_half + rsi_oversold: NEVER OCCURRED in
      the last 4 months of data at all
    - Its close cousins that did occur collapsed to 54.5%-62.5%,
      barely better than a coin flip
    - rsi_oversold alone collapsed to 50.0% and flipped direction
    - in_lower_half never occurred once in the holdout period —
      consistent with this year's drought-driven price strength
      keeping wheat out of the bottom of its 52-week range

BACKTEST FINDINGS (real, holdout-validated, as of 2026-07-09):
  Baseline UP accuracy: 67.46% (wheat trended up most of this 2yr window,
  so this is the real bar — anything below it adds zero value)

  Only 4 SINGLE conditions actually beat baseline on holdout data:
    vol_low       : 84.8% UP (n=33 holdout) — strongest real signal
    momentum_up   : 84.0% UP (n=25 holdout) — strong
    macd_bullish  : 70.0% UP (n=30 holdout) — modest but real
    bearish_month : 68.0% UP (n=25 holdout) — barely above baseline

  EXCLUDED (proven unreliable on holdout — do not re-add):
    rsi_oversold, momentum_down, near_bb_lower, in_lower_half,
    wc_bullish, rsi_neutral, inside_bb, vol_good (last four held up
    but scored BELOW baseline, meaning no real predictive value alone)

  No combination-based tiers are used anymore. Searching combinations
  reintroduces the exact overfitting problem that caused this rewrite.
  If you want a combo-based tier in the future, it must be discovered
  AND holdout-validated via backtest.py first — never wired in from a
  single train-only search result.

INTEGRATION:
  Call check_gate(df) before sending any alert.
  Returns (allowed, tier, accuracy, reason, conditions_met)
  — same interface as before, so nothing calling this needs to change.

MAINTENANCE:
  Re-run backtest.py periodically and update HOLDOUT_ACCURACY /
  BASELINE_UP below to match. Do not let these numbers go stale the
  way the old hardcoded "100%" numbers did.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ── holdout-validated accuracies (update when you re-run backtest.py) ─────────
HOLDOUT_ACCURACY = {
    'vol_low':       0.848,
    'momentum_up':   0.840,
    'macd_bullish':  0.700,
    'bearish_month': 0.680,
}
BASELINE_UP = 0.6746  # from backtest_results.json baseline_up_full

VOL_RATIO_LOW  = 0.80     # vol_low: volume < 80% of 20-day avg
BEARISH_MONTHS = [6, 7, 8]  # harvest season

# Minimum sample size to trust a tier (informational — see docstring)
MIN_TRADES_TO_TRUST = 15


class HighConvictionGate:
    """
    Evaluates current market conditions against HOLDOUT-VALIDATED
    single conditions only. No searched combinations — see module
    docstring for why that approach was abandoned.
    """

    def check_gate(self, df):
        """
        Main entry point. Call this before sending any alert.

        Args:
            df: DataFrame with OHLCV data + indicators (from add_indicators())

        Returns:
            allowed:        bool   — True = send alert, False = skip
            tier:           int    — 2/1 = conviction tier, 0 = blocked/baseline
            accuracy:       float  — real holdout-tested accuracy
            reason:         str    — human-readable explanation
            conditions:     dict   — all condition values for logging
        """
        if df is None or len(df) < 60:
            return False, 0, 0.0, "Insufficient data for gate check", {}

        conditions = self._compute_conditions(df)
        return self._evaluate_tiers(conditions)

    # ── condition computation ─────────────────────────────────────────────────

    def _compute_conditions(self, df):
        """Compute only the conditions that survived holdout validation."""
        close = df['Close'].iloc[-1]
        month = datetime.now().month

        # ── Volume ratio (for vol_low) ──
        vol_avg   = float(df['Volume'].rolling(20).mean().iloc[-1])
        vol_curr  = float(df['Volume'].iloc[-1])
        vol_ratio = vol_curr / vol_avg if vol_avg > 0 else 1.0
        vol_low   = vol_ratio < VOL_RATIO_LOW

        # ── Momentum (for momentum_up) ──
        ret_1d = float(df['Close'].pct_change(1).iloc[-1])
        ret_3d = float(df['Close'].pct_change(3).iloc[-1])
        momentum_up = ret_1d > 0 and ret_3d > 0

        # ── MACD (for macd_bullish) ──
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        macd_bullish = float(macd.iloc[-1]) > float(macd_signal.iloc[-1])

        # ── Month (for bearish_month) ──
        bearish_month = month in BEARISH_MONTHS

        conds = {
            'vol_low':       vol_low,
            'momentum_up':   momentum_up,
            'macd_bullish':  macd_bullish,
            'bearish_month': bearish_month,

            # Raw values for logging
            'vol_ratio': round(vol_ratio, 2),
            'ret_1d':    round(ret_1d, 4),
            'ret_3d':    round(ret_3d, 4),
            'month':     month,
            'price':     round(float(close), 2),
        }

        return conds

    # ── tier evaluation ───────────────────────────────────────────────────────

    def _evaluate_tiers(self, c):
        """
        Rank whichever validated conditions are currently active by
        their real holdout accuracy. Highest wins. No combinations —
        each condition was validated alone, so only single-condition
        claims are made.
        """
        active = [(name, acc) for name, acc in HOLDOUT_ACCURACY.items() if c.get(name)]

        if not active:
            return (
                False, 0, BASELINE_UP,
                f"BLOCKED — no validated condition active (baseline only, {BASELINE_UP:.1%})",
                c
            )

        active.sort(key=lambda x: x[1], reverse=True)
        best_name, best_acc = active[0]
        active_names = " + ".join(name for name, _ in active)

        if best_acc >= 0.80:
            tier = 2
            reason = (f"TIER 2 — holdout-validated {best_acc:.1%} | "
                       f"{active_names}")
        elif best_acc >= 0.68:
            tier = 1
            reason = (f"TIER 1 — holdout-validated {best_acc:.1%} | "
                       f"{active_names}")
        else:
            tier = 0
            reason = f"WEAK — {active_names} ({best_acc:.1%}, near baseline)"

        allowed = tier > 0
        return allowed, tier, best_acc, reason, c

    # ── formatting ────────────────────────────────────────────────────────────

    def format_for_telegram(self, tier, accuracy, reason, conditions):
        """Format gate result for inclusion in Telegram alert."""
        if tier == 0:
            return ""

        tier_emoji = {1: "🥈", 2: "🥇"}
        emoji = tier_emoji.get(tier, "✅")

        return (
            f"{emoji} *CONVICTION TIER {tier}* (holdout-validated)\n"
            f"Expected accuracy: {accuracy:.0%}\n"
            f"Vol: {conditions['vol_ratio']:.1f}x | "
            f"Momentum up: {conditions['momentum_up']} | "
            f"MACD bullish: {conditions['macd_bullish']}"
        )

    def log_conditions(self, conditions):
        """Print all conditions to GitHub Actions log."""
        print(f"\n🎯 Gate Conditions (holdout-validated set only):")
        print(f"   Month: {conditions['month']} | bearish_month={conditions['bearish_month']}")
        print(f"   Vol ratio: {conditions['vol_ratio']:.2f}x | vol_low={conditions['vol_low']}")
        print(f"   ret_1d={conditions['ret_1d']:+.4f} ret_3d={conditions['ret_3d']:+.4f} | "
              f"momentum_up={conditions['momentum_up']}")


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
    print(f"Tier: {tier} | Real holdout accuracy: {accuracy:.0%}")
    print(f"Reason: {reason}")
