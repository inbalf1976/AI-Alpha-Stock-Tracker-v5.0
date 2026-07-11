"""
HIGH CONVICTION GATE
=====================
UPDATED 2026-07-10: now loads from validated_conditions.json
(auto-generated weekly by generate_validated_conditions.py from
fresh backtest.py output), matching the same pattern as
ConvictionGate in wheat_monitor_pro.py. Both gates now read from
ONE shared, auto-updating source instead of each needing manual
edits — this was the actual gap that let the old fabricated "100%"
numbers drift out of sync with reality for so long.

vol_low remains permanently excluded (STRUCTURAL_EXCLUSIONS) — see
generate_validated_conditions.py docstring for the confirmed Yahoo
volume data lag that caused this.

INTEGRATION:
  Call check_gate(df) before sending any alert.
  Returns (allowed, tier, accuracy, reason, conditions_met)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


FALLBACK_HOLDOUT_ACCURACY = {
    'momentum_up':   0.840,
    'macd_bullish':  0.700,
    'bearish_month': 0.680,
}
FALLBACK_BASELINE_UP = 0.6746

STRUCTURAL_EXCLUSIONS = {'vol_low', 'vol_good', 'vol_high'}
BEARISH_MONTHS = [6, 7, 8]


def _load_holdout_accuracy():
    """Same loading logic as ConvictionGate — see wheat_monitor_pro.py."""
    path = Path("validated_conditions.json")
    if not path.exists():
        print("   ⚠️ [HighConvictionGate] validated_conditions.json not found — using fallback")
        return dict(FALLBACK_HOLDOUT_ACCURACY), FALLBACK_BASELINE_UP, "FALLBACK (no file)"

    try:
        data = json.loads(path.read_text())
        loaded = data.get('validated_conditions', {})
        baseline = data.get('baseline_up', FALLBACK_BASELINE_UP)

        cleaned = {k: v for k, v in loaded.items() if k not in STRUCTURAL_EXCLUSIONS}
        removed = set(loaded.keys()) & STRUCTURAL_EXCLUSIONS
        if removed:
            print(f"   ⚠️ [HighConvictionGate] Ignored structurally-excluded condition(s): {removed}")

        if not cleaned:
            print("   ⚠️ [HighConvictionGate] No usable conditions in file — using fallback")
            return dict(FALLBACK_HOLDOUT_ACCURACY), FALLBACK_BASELINE_UP, "FALLBACK (empty file)"

        generated_at = data.get('generated_at', 'unknown date')
        print(f"   [HighConvictionGate] Loaded {len(cleaned)} validated condition(s) "
              f"(generated {generated_at})")
        return cleaned, baseline, f"LIVE (generated {generated_at})"

    except Exception as e:
        print(f"   ⚠️ [HighConvictionGate] Failed to load ({e}) — using fallback")
        return dict(FALLBACK_HOLDOUT_ACCURACY), FALLBACK_BASELINE_UP, "FALLBACK (load error)"


class HighConvictionGate:
    """
    Evaluates current market conditions against whichever conditions
    are currently validated in validated_conditions.json (auto-
    updated weekly). vol_low structurally excluded regardless of
    what any backtest says — see module docstring.
    """

    def __init__(self):
        self.HOLDOUT_ACCURACY, self.BASELINE_UP, self._source = _load_holdout_accuracy()

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
        active = [(name, acc) for name, acc in self.HOLDOUT_ACCURACY.items() if c.get(name)]

        if not active:
            return (
                False, 0, self.BASELINE_UP,
                f"BLOCKED — no validated condition active (baseline only, {self.BASELINE_UP:.1%})",
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
            f"{emoji} *CONVICTION TIER {tier}* (holdout-validated, {self._source})\n"
            f"Expected accuracy: {accuracy:.0%}\n"
            f"Momentum up: {conditions['momentum_up']} | "
            f"MACD bullish: {conditions['macd_bullish']}"
        )

    def log_conditions(self, conditions):
        print(f"\n🎯 Gate Conditions (source: {self._source}):")
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
