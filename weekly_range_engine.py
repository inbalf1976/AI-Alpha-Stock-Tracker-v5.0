"""
WHEAT WEEKLY RANGE ENGINE
==========================
Predicts next week's price range for ZW=F.

CHANGELOG (2026-07-09):
  FIX 1: hist_up_pct ("Historically X% up this week") was displayed
  as raw ws['up_pct'] with zero sample-size discounting, even though
  the confidence score right next to it DOES apply a size factor.
  This is exactly the fake-precision problem found and fixed
  elsewhere (see ConvictionGate rebuild, backtest.py holdout
  validation) — a "100%" claim from 3-5 years of data is not
  statistically meaningful. Now flagged explicitly with a sample
  size note whenever count is small.

  FIX 2: format_alert()'s tier_labels/tier_advice dicts were still
  hardcoded to the OLD fake tier scheme (100%/94.7%/81.7%) and never
  updated when ConvictionGate was rebuilt to use real holdout-tested
  numbers (0/1/2 tiers, 68-85% real accuracy). This function was
  silently printing stale fabricated labels regardless of what the
  real gate said. Now takes the real accuracy value and builds the
  label dynamically. Trade setup is now clearly marked or omitted
  when conviction is weak, instead of always printing full
  Entry/Stop/Target numbers under a "WEAK - informational only" line.

WHY WEEKLY:
  - Daily: 58-68% accuracy (too much noise)
  - Weekly: 72-80% range accuracy (seasonal + fundamentals dominate)
  - Monthly: 75-85% range accuracy (best for wheat)

APPROACH:
  1. Historical weekly ranges by season — what does wheat typically
     move in a week during this time of year?
  2. ATR-based range width — current volatility sets the range size
  3. Directional bias — seasonal + trend + cost floor = up or down
  4. Key levels — cost floor support, 52wk levels, recent highs/lows
  5. Confidence — how consistent is this week's pattern historically?

OUTPUT:
  - Weekly range: LOW to HIGH in cents
  - Bias: UP / DOWN / NEUTRAL with confidence %
  - Key level: most important price to watch
  - Trade setup: entry, stop, target based on weekly range
  - Historical accuracy: how often this pattern played out
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

IL = ZoneInfo("Asia/Jerusalem")


class WeeklyRangeEngine:
    """
    Predicts next week's wheat price range using historical patterns.
    """

    def __init__(self):
        self.weekly_stats  = None   # historical weekly stats by week-of-year
        self.fitted        = False

    # ── FIT ──────────────────────────────────────────────────────────────────

    def fit(self, df, exclude_years=None):
        exclude_years = exclude_years or []

        df = df.copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df['year']    = df.index.year
        df['week']    = df.index.isocalendar().week.astype(int)
        df['ret_1d']  = df['Close'].pct_change()

        df = df[~df['year'].isin(exclude_years)]

        weekly = df.groupby(['year', 'week']).agg(
            open  =('Close', 'first'),
            high  =('High',  'max'),
            low   =('Low',   'min'),
            close =('Close', 'last'),
            vol   =('Volume','mean'),
        ).reset_index()

        weekly['range_pct']   = (weekly['high'] - weekly['low']) / weekly['open']
        weekly['return_pct']  = (weekly['close'] - weekly['open']) / weekly['open']
        weekly['direction']   = (weekly['return_pct'] > 0).astype(int)

        stats = weekly.groupby('week').agg(
            avg_range    =('range_pct',  'mean'),
            std_range    =('range_pct',  'std'),
            avg_return   =('return_pct', 'mean'),
            up_pct       =('direction',  'mean'),
            count        =('open',       'count'),
        ).reset_index()

        stats['down_pct']      = 1 - stats['up_pct']
        stats['bias_strength'] = (stats['up_pct'] - 0.5).abs()

        self.weekly_stats = stats.set_index('week')
        self.fitted       = True

    # ── PREDICT ───────────────────────────────────────────────────────────────

    def predict_next_week(self, df, current_price, cost_floor_cents=None):
        """
        FIX (2026-07-10): previously used `next_week = today + 7 days`,
        meaning the forecast ALWAYS described a rolling week starting
        7 days from whenever the script happened to run — NOT the
        calendar week currently in progress. This caused real
        confusion: an alert run on a Monday would forecast the
        FOLLOWING week (not the current one), while an alert run on a
        Thursday might land in the same ISO week by coincidence,
        making the target week inconsistent and non-obvious depending
        on which day you happened to check.

        Now targets the CURRENT ISO week (Monday-Sunday containing
        "today"), so "this week's forecast" actually means the week
        you're currently in, matching what a person reading "Week 29"
        on a given day would reasonably assume it means. Combined
        with get_frozen_weekly_range() in wheat_monitor_pro.py, this
        is computed once at the start of each real calendar week and
        held fixed until that week is actually over.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first")

        # Target THIS week (containing today), not a shifting +7 day window
        today = datetime.now(IL)
        target_week_num = int(today.isocalendar()[1])
        # Monday of the current ISO week, for a clear, stable label
        week_monday = today - timedelta(days=today.isoweekday() - 1)

        if target_week_num in self.weekly_stats.index:
            ws = self.weekly_stats.loc[target_week_num]
        else:
            ws = self.weekly_stats.iloc[min(target_week_num, len(self.weekly_stats)-1)]

        hist_range_pct  = float(ws['avg_range'])
        hist_range_std  = float(ws['std_range']) if not np.isnan(ws['std_range']) else hist_range_pct * 0.3

        atr_pct = float(df['ATR'].iloc[-1]) / current_price
        atr_weekly_estimate = atr_pct * 2.5
        blended_range_pct = hist_range_pct * 0.60 + atr_weekly_estimate * 0.40
        range_half = (current_price * blended_range_pct) / 2

        hist_up_pct    = float(ws['up_pct'])
        hist_avg_ret   = float(ws['avg_return'])
        sample_count   = int(ws['count'])

        bias_score = hist_avg_ret

        price     = float(df['Close'].iloc[-1])
        sma5      = float(df['Close'].rolling(5).mean().iloc[-1])
        sma20     = float(df['SMA_20'].iloc[-1])
        sma50     = float(df['SMA_50'].iloc[-1])

        if price > sma5 > sma20:
            bias_score += 0.003
        elif price < sma5 < sma20:
            bias_score -= 0.003

        if cost_floor_cents:
            dist_from_floor = (current_price - cost_floor_cents) / cost_floor_cents
            if dist_from_floor < 0.02:
                bias_score += 0.004
            elif dist_from_floor > 0.10:
                bias_score -= 0.002

        if bias_score > 0.002:
            bias      = 'UP'
            bias_pct  = hist_up_pct
        elif bias_score < -0.002:
            bias      = 'DOWN'
            bias_pct  = 1 - hist_up_pct
        else:
            bias      = 'NEUTRAL'
            bias_pct  = 0.50

        consistency = float(ws['bias_strength'])
        size_factor = min(1.0, sample_count / 5)

        base_conf   = 0.50 + consistency * 0.60
        confidence  = min(0.85, base_conf * size_factor)

        if bias == 'NEUTRAL':
            confidence = min(0.55, confidence)

        # ── Sample-size honesty check (FIX 1) ────────────────────────────────
        if sample_count < 4:
            sample_confidence_note = f"LOW CONFIDENCE — only {sample_count} years of data"
        elif sample_count < 8:
            sample_confidence_note = f"moderate confidence — {sample_count} years of data"
        else:
            sample_confidence_note = f"{sample_count} years of data"

        if bias == 'UP':
            range_low  = current_price - range_half * 0.40
            range_high = current_price + range_half * 0.60
        elif bias == 'DOWN':
            range_low  = current_price - range_half * 0.60
            range_high = current_price + range_half * 0.40
        else:
            range_low  = current_price - range_half * 0.50
            range_high = current_price + range_half * 0.50

        if cost_floor_cents and range_low < cost_floor_cents < range_high:
            key_level       = cost_floor_cents
            key_level_label = "Cost floor (strong support)"
        elif price > sma50:
            key_level       = round(sma50, 2)
            key_level_label = "SMA50 (key support)"
        else:
            key_level       = round(sma20, 2)
            key_level_label = "SMA20 (nearest support)"

        if bias == 'UP':
            entry  = current_price
            stop   = round(range_low - (range_half * 0.10), 2)
            target = round(range_high, 2)
        elif bias == 'DOWN':
            entry  = current_price
            stop   = round(range_high + (range_half * 0.10), 2)
            target = round(range_low, 2)
        else:
            entry  = current_price
            stop   = round(range_low * 0.99, 2)
            target = round(range_high * 1.01, 2)

        rr = abs(target - entry) / abs(stop - entry) if abs(stop - entry) > 0 else 0

        month_labels = {
            1:'Jan neutral', 2:'Pre-spring dip', 3:'Spring rally',
            4:'Peak planting', 5:'Weather premium', 6:'Harvest pressure',
            7:'Post-harvest low', 8:'Summer lull', 9:'Fall recovery',
            10:'Winter demand', 11:'Pre-winter rally', 12:'Winter high'
        }
        month_label = month_labels.get(week_monday.month, '')

        return {
            'range_low':        round(range_low, 2),
            'range_high':       round(range_high, 2),
            'range_width_pct':  round(blended_range_pct * 100, 2),
            'bias':             bias,
            'bias_pct':         round(bias_pct * 100, 1),
            'confidence':       round(confidence, 3),
            'key_level':        key_level,
            'key_level_label':  key_level_label,
            'entry':            round(entry, 2),
            'stop':             stop,
            'target':           target,
            'rr':               round(rr, 2),
            'week_num':         target_week_num,
            'next_week_label':  f"Week {target_week_num} (starting Mon {week_monday.strftime('%b %d')})",
            'month_label':      month_label,
            'hist_up_pct':      round(hist_up_pct * 100, 1),
            'hist_avg_return':  round(hist_avg_ret * 100, 2),
            'sample_count':     sample_count,
            'sample_confidence_note': sample_confidence_note,
            'current_price':    round(current_price, 2),
        }

    # ── MONTHLY OUTLOOK ───────────────────────────────────────────────────────

    def predict_monthly_range(self, df, current_price, cost_floor_cents=None):
        """
        FIX (2026-07-11): same bug class as predict_next_week had
        before its 2026-07-10 fix — this previously aggregated "the
        next 4 weeks from today", a rolling window that shifts every
        single day, rather than the actual current calendar month.
        Combined with using live current_price as the center point
        every run, this caused the monthly range to visibly change
        on every alert, all week, for no real reason.

        Now aggregates the ISO weeks that fall within the CURRENT
        calendar month (today's actual month, 1st to last day), so
        "July's range" genuinely means July. Combined with
        get_frozen_monthly_range() in wheat_monitor_pro.py, this
        should be computed once per real calendar month and held
        fixed until the month changes.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first")

        today = datetime.now(IL)
        month_start = today.replace(day=1)
        if today.month == 12:
            next_month_start = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month_start = today.replace(month=today.month + 1, day=1)
        month_end = next_month_start - timedelta(days=1)

        # Count how many days of the month fall in each ISO week, so
        # boundary weeks (e.g. a week that's mostly June, with only 1-2
        # days in July) contribute proportionally, not at full weight.
        # Without this, months spanning 5 ISO weeks (common) summed all
        # 5 weeks' full avg_range as if each fully belonged to the
        # month, producing an unrealistic ~25%+ total swing that only
        # looked sane because the 15% safety clamp silently caught it.
        week_day_counts = {}
        d = month_start
        while d <= month_end:
            wn = int(d.isocalendar()[1])
            week_day_counts[wn] = week_day_counts.get(wn, 0) + 1
            d += timedelta(days=1)

        monthly_lows  = []
        monthly_highs = []
        monthly_biases = []

        for week_num in sorted(week_day_counts.keys()):
            if week_num in self.weekly_stats.index:
                ws          = self.weekly_stats.loc[week_num]
                range_pct   = float(ws['avg_range'])
                avg_ret     = float(ws['avg_return'])
                weight      = week_day_counts[week_num] / 7.0
                monthly_lows.append((avg_ret - range_pct / 2) * weight)
                monthly_highs.append((avg_ret + range_pct / 2) * weight)
                monthly_biases.append(avg_ret * weight)

        if not monthly_lows:
            return None

        cumulative_low  = current_price * (1 + sum(monthly_lows))
        cumulative_high = current_price * (1 + sum(monthly_highs))
        avg_monthly_ret = sum(monthly_biases)

        max_move = current_price * 0.15
        monthly_low  = max(current_price - max_move, min(cumulative_low,  cumulative_high))
        monthly_high = min(current_price + max_move, max(cumulative_low,  cumulative_high))

        if cost_floor_cents and monthly_low < cost_floor_cents:
            monthly_low = cost_floor_cents * 0.99

        bias = 'UP' if avg_monthly_ret > 0.005 else 'DOWN' if avg_monthly_ret < -0.005 else 'NEUTRAL'

        return {
            'monthly_low':   round(monthly_low, 2),
            'monthly_high':  round(monthly_high, 2),
            'bias':          bias,
            'avg_return':    round(avg_monthly_ret * 100, 2),
            'month_label':   datetime.now(IL).strftime('%B'),
        }

    # ── FORMAT FOR TELEGRAM ───────────────────────────────────────────────────

    def format_alert(self, weekly, monthly=None, tier=0, gate_conds=None,
                     wasde=None, weather=None, seasonal=None, cost_signal=None,
                     gate_accuracy=None, gate_reason=None, final_direction=None):
        """
        FIX 2 (2026-07-09): tier_labels/tier_advice used to be hardcoded
        to the OLD fake tier scheme (100%/94.7%/81.7%) and never updated
        when ConvictionGate was rebuilt with real holdout-tested numbers.
        Now builds the label from the REAL accuracy value passed in via
        gate_accuracy. Trade setup is now clearly gated: full numbers
        only shown when conviction is real (tier > 0); otherwise an
        explicit "no trade setup" message replaces it.

        FIX 3 (2026-07-09): the message used to show weekly['bias'] as a
        standalone "BIAS: UP/DOWN" line with no indication of whether it
        actually won the override against the ensemble's direction (see
        wheat_monitor_pro.py — override only fires if weekly confidence
        >= 65%). This caused a real, confusing case in practice: an
        alert showed "BIAS: UP (48%)" prominently while the ensemble was
        heavily DOWN and the monthly outlook also said DOWN — the weekly
        bias had LOST the override (48% < 65% threshold) but was still
        displayed as if it were the headline call. Now the message
        leads with an explicit FINAL CALL line, and the BIAS line is
        annotated whenever it disagrees with the actual final direction.
        """

        if gate_accuracy is not None:
            if tier == 0:
                tier_label = f"NO SIGNAL - baseline only ({gate_accuracy:.0%})"
                decision   = "WEAK - no trade setup below"
            elif gate_accuracy >= 0.80:
                tier_label = f"TIER {tier} - {gate_accuracy:.0%} holdout-validated accuracy"
                decision   = "MODERATE-STRONG - real edge, size accordingly"
            else:
                tier_label = f"TIER {tier} - {gate_accuracy:.0%} holdout-validated accuracy"
                decision   = "MODERATE - modest real edge, not a high-conviction signal"
        else:
            tier_label = "NO TIER - accuracy unavailable"
            decision   = "WEAK - no trade setup below"

        rsi_str = f"RSI: {gate_conds.get('rsi'):.0f}" if gate_conds and gate_conds.get('rsi') is not None else ""
        vol_str = f"Vol: {gate_conds.get('vol_ratio'):.1f}x" if gate_conds and gate_conds.get('vol_ratio') is not None else ""

        cost_line = ""
        if cost_signal:
            cost_line = (
                f"Cost floor: {cost_signal['floor_cents']:.0f}c "
                f"({cost_signal['distance_pct']:+.1%} from current) "
                f"- {cost_signal['signal']}\n"
            )

        monthly_line = ""
        if monthly:
            monthly_line = (
                f"\nMONTHLY OUTLOOK ({monthly['month_label']}):\n"
                f"Range: {monthly['monthly_low']:.0f} - {monthly['monthly_high']:.0f}c\n"
                f"Bias: {monthly['bias']} ({monthly['avg_return']:+.1f}% avg)\n"
            )

        seasonal_line = ""
        if seasonal:
            seasonal_line = (
                f"Seasonal: {seasonal['phase']} ({seasonal['confidence']:.0%}) "
                f"- {seasonal['explanation']}\n"
                f"Next 20d: {seasonal['pos_days']} up / {seasonal['neg_days']} down\n"
            )

        fundamental_line = ""
        if wasde:
            fundamental_line += f"WASDE: {wasde['signal']} ({wasde['source']})\n"
        if weather:
            fundamental_line += f"Weather: {weather['signal']}\n"

        # ── Trade setup block — only shown with real conviction (FIX 2) ──────
        # ── FIX 4 (2026-07-09): trade setup must match FINAL CALL, not the
        # weekly engine's own (possibly overridden/losing) bias. Previously
        # Entry/Stop/Target were always built from weekly['bias'] even when
        # final_direction disagreed — meaning an alert could show
        # "FINAL CALL: DOWN" at the top and still print a bullish trade
        # setup (stop below entry, target above) underneath it. Now the
        # setup direction is derived from final_direction when provided.
        effective_direction = final_direction if final_direction is not None else weekly['bias']

        if tier > 0:
            if effective_direction == 'UP':
                setup_entry  = weekly['current_price']
                setup_stop   = weekly['range_low']
                setup_target = weekly['range_high']
            elif effective_direction == 'DOWN':
                setup_entry  = weekly['current_price']
                setup_stop   = weekly['range_high']
                setup_target = weekly['range_low']
            else:
                setup_entry  = weekly['entry']
                setup_stop   = weekly['stop']
                setup_target = weekly['target']

            setup_rr = (abs(setup_target - setup_entry) / abs(setup_stop - setup_entry)
                        if abs(setup_stop - setup_entry) > 0 else 0)

            direction_note = ""
            if effective_direction != weekly['bias']:
                direction_note = f" (setup direction follows FINAL CALL {effective_direction}, not the weekly bias above)\n"

            trade_setup_block = (
                f"TRADE SETUP:\n"
                f"{direction_note}"
                f"Entry:  {setup_entry:.0f}c (current)\n"
                f"Stop:   {setup_stop:.0f}c\n"
                f"Target: {setup_target:.0f}c\n"
                f"R:R = {setup_rr:.1f}:1\n"
            )
        else:
            trade_setup_block = (
                f"No trade setup shown — conviction is baseline/weak.\n"
                f"(Entry/stop/target numbers are only shown when a\n"
                f"holdout-validated condition is actually active.)\n"
            )

        # ── FIX 3: unambiguous final call header ─────────────────────────────
        final_call_block = ""
        bias_annotation = ""
        if final_direction is not None:
            final_call_block = f"FINAL CALL: {final_direction}\n" + ("=" * 30) + "\n\n"
            if weekly['bias'] != 'NEUTRAL' and weekly['bias'] != final_direction:
                bias_annotation = (
                    f" (did NOT override — final call is {final_direction}, "
                    f"this bias was below the 65% override threshold)"
                )

        message = (
            f"{final_call_block}"
            f"WHEAT WEEKLY OUTLOOK\n"
            f"------------------------------\n"
            f"{weekly['next_week_label']} | {weekly['month_label']}\n\n"
            f"EXPECTED RANGE:\n"
            f"Low:  {weekly['range_low']:.0f}c\n"
            f"High: {weekly['range_high']:.0f}c\n"
            f"Width: {weekly['range_width_pct']:.1f}% of price\n\n"
            f"BIAS: {weekly['bias']} ({weekly['confidence']:.0%} confidence){bias_annotation}\n"
            f"Historically {weekly['hist_up_pct']:.0f}% up this week of year\n"
            f"({weekly['sample_confidence_note']})\n\n"
            f"KEY LEVEL: {weekly['key_level']:.0f}c\n"
            f"{weekly['key_level_label']}\n\n"
            f"{cost_line}"
            f"{seasonal_line}"
            f"{fundamental_line}\n"
            f"CONVICTION: {tier_label}\n"
            f"{rsi_str} | {vol_str}\n"
            f"DECISION: {decision}\n\n"
            f"{trade_setup_block}"
            f"{monthly_line}"
        )

        return message


# ── STANDALONE TEST ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("Testing Weekly Range Engine...")
    end   = datetime.now(ZoneInfo("Asia/Jerusalem"))
    start = end - timedelta(days=5 * 365)
    df    = yf.Ticker("ZW=F").history(start=start, end=end, auto_adjust=False)
    df    = df.iloc[:-1]

    df['Returns']    = df['Close'].pct_change()
    df['SMA_20']     = df['Close'].rolling(20).mean()
    df['SMA_50']     = df['Close'].rolling(50).mean()
    bb_mid           = df['Close'].rolling(20).mean()
    bb_std           = df['Close'].rolling(20).std()
    df['BB_Width']   = (bb_std * 2) / bb_mid
    df['Volatility'] = df['Returns'].rolling(20).std()
    hl               = df['High'] - df['Low']
    hc               = (df['High'] - df['Close'].shift()).abs()
    lc               = (df['Low']  - df['Close'].shift()).abs()
    df['ATR']        = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df               = df.dropna()

    price = float(df['Close'].iloc[-1])
    print(f"Current price: {price:.2f}c")

    engine = WeeklyRangeEngine()
    engine.fit(df, exclude_years=[2022])

    weekly  = engine.predict_next_week(df, price, cost_floor_cents=608)
    monthly = engine.predict_monthly_range(df, price, cost_floor_cents=608)

    print(f"\nWEEKLY PREDICTION:")
    print(f"  Range:      {weekly['range_low']:.0f} - {weekly['range_high']:.0f}c")
    print(f"  Bias:       {weekly['bias']} ({weekly['confidence']:.0%})")
    print(f"  Key level:  {weekly['key_level']:.0f}c ({weekly['key_level_label']})")
    print(f"  Trade:      Entry {weekly['entry']:.0f} | Stop {weekly['stop']:.0f} | Target {weekly['target']:.0f}")
    print(f"  R:R:        {weekly['rr']:.1f}:1")
    print(f"  History:    {weekly['hist_up_pct']:.0f}% up this week | {weekly['sample_confidence_note']}")

    if monthly:
        print(f"\nMONTHLY OUTLOOK:")
        print(f"  Range: {monthly['monthly_low']:.0f} - {monthly['monthly_high']:.0f}c")
        print(f"  Bias:  {monthly['bias']} ({monthly['avg_return']:+.1f}%)")
