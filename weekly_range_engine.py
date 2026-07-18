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

    def compute_setup_from_entry(self, entry, direction):
        """
        NEW (2026-07-14): replaces the old ATR/historical-blended range
        for the actual TRADE SETUP (not the informational EXPECTED
        RANGE, which still uses seasonal history separately). Uses a
        fixed, asymmetric -15%/+25% band from the entry price —
        deliberately wide and skewed upward, reflecting the recent
        reality that Black Sea supply shocks have repeatedly blown
        through narrower historical-based ranges this month, with
        upside moves (698 high) larger than downside ones.

        direction: 'UP' or 'DOWN' — determines which side is the
        stop and which is the target.
        """
        entry = float(entry)
        if direction == 'UP':
            low  = round(entry * 0.85, 2)
            high = round(entry * 1.25, 2)
            stop, target = low, high
        elif direction == 'DOWN':
            low  = round(entry * 0.75, 2)
            high = round(entry * 1.15, 2)
            stop, target = high, low
        else:  # NEUTRAL fallback — symmetric-ish, no strong directional lean
            low  = round(entry * 0.85, 2)
            high = round(entry * 1.15, 2)
            stop, target = low, high

        rr = abs(target - entry) / abs(stop - entry) if abs(stop - entry) > 0 else 0

        return {
            'range_low': low, 'range_high': high,
            'entry': round(entry, 2), 'stop': stop, 'target': target,
            'rr': round(rr, 2),
        }

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

    def predict_next_week(self, df, current_price, cost_floor_cents=None,
                           forced_direction=None, daily_direction_hint=None):
        """
        FIX (2026-07-10): targets the CURRENT ISO week, not a shifting
        +7 day window — see prior changelog entries for full history.

        REBUILT (2026-07-14) per corrected design: the model computes
        its OWN real weekly HIGH/LOW from historical seasonal pattern
        + current ATR (same as the original design), and that
        forecast can be ANY width — it is only CLAMPED to a hard
        outer safety boundary of -15%/+25% from current price, never
        forced to exactly those numbers. A quiet, typical week might
        forecast a narrow 8-12% range; only if the raw calculation
        would exceed the -15%/+25% outer limits does clamping kick
        in. (An earlier version of this fix mistakenly hardcoded the
        range to always be exactly -15%/+25% — corrected here.)

        forced_direction: used only when regenerating after a broken
        setup (win → same direction, loss → flip) — overrides the
        data-derived bias so the forced direction still gets a real,
        clamped range appropriate to it, rather than a fixed formula.

        daily_direction_hint: today's backtest-informed daily ensemble
        read (ConvictionGate/EnsemblePredictor), fed in as a small
        additional nudge to the weekly bias score — per design note,
        the weekly forecast should also draw on the same
        backtest-validated signals the daily alert uses, not only
        seasonal/trend/cost-floor data in isolation.
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

        # OUTER SAFETY BOUNDS ONLY — the real forecast below can be
        # (and usually should be) much narrower than this; these are
        # just the hard ceiling/floor it may never exceed.
        MAX_DOWN_PCT = 0.15
        MAX_UP_PCT   = 0.25

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

        # Small nudge from today's backtest-informed daily direction,
        # per design note — the weekly call should partly draw on the
        # same validated signals the daily alert already uses.
        if daily_direction_hint == 'UP':
            bias_score += 0.0015
        elif daily_direction_hint == 'DOWN':
            bias_score -= 0.0015

        if forced_direction in ('UP', 'DOWN'):
            bias = forced_direction
            bias_pct = hist_up_pct if bias == 'UP' else (1 - hist_up_pct)
        elif bias_score > 0.002:
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

        # Real forecast, skewed toward the bias direction (as before)
        if bias == 'UP':
            range_low  = current_price - range_half * 0.40
            range_high = current_price + range_half * 0.60
        elif bias == 'DOWN':
            range_low  = current_price - range_half * 0.60
            range_high = current_price + range_half * 0.40
        else:
            range_low  = current_price - range_half * 0.50
            range_high = current_price + range_half * 0.50

        # ── OUTER BOUND CLAMP (the actual fix) ────────────────────────────────
        # The real forecast above may be narrower than these limits
        # (typical/quiet weeks) — it just may never be WIDER than them.
        floor_price   = current_price * (1 - MAX_DOWN_PCT)
        ceiling_price = current_price * (1 + MAX_UP_PCT)
        raw_low, raw_high = range_low, range_high  # DIAGNOSTIC: pre-clamp values
        clamped = False
        if range_low < floor_price:
            range_low = floor_price
            clamped = True
        if range_high > ceiling_price:
            range_high = ceiling_price
            clamped = True

        if clamped:
            print(f"   ⚠️ Weekly range CLAMPED: raw forecast was {raw_low:.0f}-{raw_high:.0f}c "
                  f"({(raw_high-raw_low)/current_price:.1%} width) -> clamped to "
                  f"{range_low:.0f}-{range_high:.0f}c ({(range_high-range_low)/current_price:.1%} width). "
                  f"range_half(pre-clamp)={range_half:.2f}c, atr_pct={atr_pct:.4f}, "
                  f"hist_range_pct={hist_range_pct:.4f}, blended_range_pct={blended_range_pct:.4f}")

        blended_range_pct = (range_high - range_low) / current_price
        range_half = (range_high - range_low) / 2  # recompute AFTER clamp, for a consistent stop buffer

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
        #
        # SECOND FIX (same day): the weighting fix above helped only
        # marginally (156c -> still ~156c wide) because the deeper
        # problem wasn't boundary weeks — it was summing each week's
        # FULL range linearly across 4-5 weeks. Range/volatility does
        # NOT add up linearly across periods; it scales with the
        # square root of time (standard random-walk/volatility scaling
        # in finance). Naively summing 5 weeks' full ranges massively
        # overstates real monthly range — that's why the 15% safety
        # clamp was silently doing all the work both times, on both
        # sides of the range. Now uses a weighted AVERAGE weekly range
        # (not a sum), scaled by sqrt(effective number of weeks),
        # applied once — the statistically defensible way to do this.
        week_day_counts = {}
        d = month_start
        while d <= month_end:
            wn = int(d.isocalendar()[1])
            week_day_counts[wn] = week_day_counts.get(wn, 0) + 1
            d += timedelta(days=1)

        weighted_range_pcts = []
        weighted_returns    = []
        weights             = []

        for week_num, days_in_month in week_day_counts.items():
            if week_num in self.weekly_stats.index:
                ws        = self.weekly_stats.loc[week_num]
                range_pct = float(ws['avg_range'])
                avg_ret   = float(ws['avg_return'])
                weight    = days_in_month / 7.0
                weighted_range_pcts.append(range_pct * weight)
                weighted_returns.append(avg_ret * weight)
                weights.append(weight)

        if not weights:
            return None

        total_weight = sum(weights)
        # Weighted average weekly range (not a sum) — the typical
        # week's range this month, on average
        avg_weekly_range_pct = sum(weighted_range_pcts) / total_weight
        # Returns DO compound roughly additively (fine to sum) —
        # this is the expected total drift over the month
        avg_monthly_ret = sum(weighted_returns)
        # Volatility scales with sqrt(time), not linear time —
        # this is the actual fix
        monthly_range_pct = avg_weekly_range_pct * (total_weight ** 0.5)

        center = current_price * (1 + avg_monthly_ret)
        half_range = current_price * monthly_range_pct / 2
        cumulative_low  = center - half_range
        cumulative_high = center + half_range

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
                     gate_accuracy=None, gate_reason=None, final_direction=None,
                     daily_direction=None, status_line=None):
        """
        UPDATED 2026-07-14 — new weekly plan design:
          - weekly['final_call'] is now the FROZEN weekly direction
            (only changes when the week's TP/SL setup breaks), shown
            as the top-level "WEEKLY FINAL CALL" header.
          - daily_direction is a SEPARATE, freshly-computed daily
            technical read (ensemble/trend/seasonal), shown on its
            own line, distinct from the frozen weekly call.
          - status_line shows whether today's real price is inside
            or outside the current frozen setup's range.
          - Trade setup (entry/stop/target) now uses a fixed -15%/
            +25% band from entry (see compute_setup_from_entry),
            not the old ATR/historical blend — EXPECTED RANGE above
            it still uses the historical/seasonal calc separately,
            since that section answers a different question ("what's
            typical this week historically") than the trade setup
            ("what am I actually risking against right now").

        FIX 2 (2026-07-09): tier_labels/tier_advice used to be hardcoded
        to the OLD fake tier scheme (100%/94.7%/81.7%) and never updated
        when ConvictionGate was rebuilt with real holdout-tested numbers.
        Now builds the label from the REAL accuracy value passed in via
        gate_accuracy. Trade setup is now clearly gated: full numbers
        only shown when conviction is real (tier > 0); otherwise an
        explicit "no trade setup" message replaces it.
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
        # ── NEW (2026-07-14): trade setup now comes directly from the
        # weekly dict's own entry/stop/target — set once via
        # compute_setup_from_entry() when the week (or a mid-week
        # breakout regeneration) was frozen, in wheat_monitor_pro.py.
        # No longer recalculated here from bias/current_price.
        weekly_final_call = weekly.get('final_call', final_direction)

        if tier > 0 and weekly.get('entry') is not None:
            trade_setup_block = (
                f"TRADE SETUP:\n"
                f"Entry:  {weekly['entry']:.0f}c\n"
                f"Stop:   {weekly['stop']:.0f}c\n"
                f"Target: {weekly['target']:.0f}c\n"
                f"R:R = {weekly['rr']:.1f}:1\n"
            )
        else:
            trade_setup_block = (
                f"No trade setup shown — conviction is baseline/weak.\n"
                f"(Entry/stop/target numbers are only shown when a\n"
                f"holdout-validated condition is actually active.)\n"
            )

        final_call_block = f"WEEKLY FINAL CALL: {weekly_final_call}\n" + ("=" * 30) + "\n\n"
        daily_direction_line = f"daily direction: {daily_direction}\n\n" if daily_direction else ""
        status_block = f"\nstatus- {status_line}\n" if status_line else ""

        message = (
            f"{final_call_block}"
            f"{daily_direction_line}"
            f"WHEAT WEEKLY OUTLOOK\n"
            f"------------------------------\n"
            f"{weekly['next_week_label']} | {weekly['month_label']}\n\n"
            f"EXPECTED RANGE:\n"
            f"Low:  {weekly['range_low']:.0f}c\n"
            f"High: {weekly['range_high']:.0f}c\n\n"
            f"BIAS: {weekly['bias']} ({weekly['confidence']:.0%} confidence)\n"
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
            f"{status_block}"
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
