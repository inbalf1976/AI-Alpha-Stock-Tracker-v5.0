"""
HIGH CONVICTION GATE
=====================
REBUILT 2026-07-09 to match the real, holdout-validated numbers now
used in wheat_monitor_pro.py's ConvictionGate class.
UPDATED 2026-07-10: vol_low REMOVED — see below.

WHY vol_low WAS REMOVED (2026-07-10):
  A diagnostic (volume_lag_check.py) showed ZW=F's Yahoo daily Volume
  field takes roughly 1-2 WEEKS to fully backfill for this continuous
  futures contract. Dates within the last ~10 days showed volume
  readings of single/low-double digits (e.g. 7, 48, 136 contracts) on
  the most liquid wheat contract in the world — obviously incomplete
  data, not real trading activity. Since vol_low (ratio < 0.80)
  compares current volume against a 20-day average that is ITSELF
  partly built from these same artificially-low recent values, it was
  almost certainly firing as effectively-always-true on any recent
  date — meaning its 84.8% holdout accuracy likely measured "is this
  date recent" rather than any genuine market behavior. This is a
  STRUCTURAL problem that recurs every single day this runs, not a
  rare glitch — removed rather than patched.

BACKTEST FINDINGS (real, holdout-validated, as of 2026-07-09):
  Baseline UP accuracy: 67.46%

  Conditions that beat baseline on holdout data AND do not depend on
  the unreliable Volume field:
    momentum_up   : 84.0% UP (n=25 holdout) — strongest reliable signal
    macd_bullish  : 70.0% UP (n=30 holdout) — modest but real
    bearish_month : 68.0% UP (n=25 holdout) — barely above baseline

  REMOVED:
    vol_low — structural data lag, see above.

  EXCLUDED (proven unreliable on holdout — do not re-add):
    rsi_oversold, momentum_down, near_bb_lower, in_lower_half,
    wc_bullish, rsi_neutral, inside_bb, vol_good

INTEGRATION:
  Call check_gate(df) before sending any alert.
  Returns (allowed, tier, accuracy, reason, conditions_met)
"""

import pandas as pd
import numpy as np
from datetime import datetime


HOLDOUT_ACCURACY = {
    'momentum_up':   0.840,
    'macd_bullish':  0.700,
    'bearish_month': 0.680,
}
BASELINE_UP = 0.6746

BEARISH_MONTHS = [6, 7, 8]
MIN_TRADES_TO_TRUST = 15


class HighConvictionGate:
    """
    Evaluates current market conditions against HOLDOUT-VALIDATED
    single conditions only. vol_low removed due to a confirmed
    structural data reliability problem — see module docstring.
    """

    def check_gate(self, df):
        if df is None or len(df) < 60:
            return False, 0, 0.0, "Insufficient data for gate check", {}

        conditions = self._compute_conditions(df)
        return self._evaluate_tiers(conditions)

    def _compute_conditions(self, df):
        close = df['Close'].iloc[-1]
        month = datetime.now().month

        ret_1d = float(df['Close'].pct_change(1).iloc[-1])
        ret_3d = float(df['Close'].pct_change(3).iloc[-1])
        momentum_up = ret_1d > 0 and ret_3d > 0

        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        macd_bullish = float(macd.iloc[-1]) > float(macd_signal.iloc[-1])

        bearish_month = month in BEARISH_MONTHS

        conds = {
            'momentum_up':   momentum_up,
            'macd_bullish':  macd_bullish,
            'bearish_month': bearish_month,
            'ret_1d':    round(ret_1d, 4),
            'ret_3d':    round(ret_3d, 4),
            'month':     month,
            'price':     round(float(close), 2),
        }

        return conds

    def _evaluate_tiers(self, c):
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
            reason = f"TIER 2 — holdout-validated {best_acc:.1%} | {active_names}"
        elif best_acc >= 0.68:
            tier = 1
            reason = f"TIER 1 — holdout-validated {best_acc:.1%} | {active_names}"
        else:
            tier = 0
            reason = f"WEAK — {active_names} ({best_acc:.1%}, near baseline)"

        allowed = tier > 0
        return allowed, tier, best_acc, reason, c

    def format_for_telegram(self, tier, accuracy, reason, conditions):
        if tier == 0:
            return ""

        tier_emoji = {1: "🥈", 2: "🥇"}
        emoji = tier_emoji.get(tier, "✅")

        return (
            f"{emoji} *CONVICTION TIER {tier}* (holdout-validated)\n"
            f"Expected accuracy: {accuracy:.0%}\n"
            f"Momentum up: {conditions['momentum_up']} | "
            f"MACD bullish: {conditions['macd_bullish']}"
        )

    def log_conditions(self, conditions):
        print(f"\n🎯 Gate Conditions (holdout-validated set only, vol_low removed):")
        print(f"   Month: {conditions['month']} | bearish_month={conditions['bearish_month']}")
        print(f"   ret_1d={conditions['ret_1d']:+.4f} ret_3d={conditions['ret_3d']:+.4f} | "
              f"momentum_up={conditions['momentum_up']}")
        print(f"   macd_bullish={conditions['macd_bullish']}")


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
