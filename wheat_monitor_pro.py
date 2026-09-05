"""
WHEAT MONITOR v4.0 - CLEAN REBUILD
=====================================
Built from scratch using everything learned over the past month.

DESIGN PRINCIPLES:
  1. Seasonal truth first — 5 years of ZW=F history defines the calendar
     2022 excluded (Ukraine war = global anomaly)
  2. Real price always — uses LIVE current price (not stale daily bar)
  3. Trend respect — never fights a confirmed multi-day trend
  4. Conviction gate — only alerts on historically proven setups
  5. Honest confidence — no artificial boosting, real probabilities only
  6. One alert per day — no duplicates, no noise

SIGNAL HIERARCHY (in order of weight):
  1. Seasonal phase      — derived from 5yr history, hard override
  2. Trend direction     — 5/10/20 day MA alignment
  3. Conviction tier     — backtest-proven condition combinations
  4. Ensemble models     — LSTM + RF + XGB with daily-sensitive features
  5. Fundamental context — WASDE multi-grain, weather, volume

ACCURACY TARGET: 80%+ on Tier 1/2 setups (~6-10 alerts/month)

CHANGELOG (this version):
  - FIX: current_price now comes from a live quote (get_live_price),
    not the last daily bar. The old logic dropped "today's" candle
    to avoid using an incomplete bar, but that meant current_price
    could silently lag by days around weekends/holidays. Now the
    daily bars still drive all indicators/seasonal/trend calcs —
    only the single "current price" used for entry/stop/target is
    live. If the live fetch fails, this is now flagged explicitly
    (⚠️ STALE) instead of failing silently.
  - CHANGE (2026-08-07): full formatted alert now sends EVERY day
    the alert gate is open, regardless of tier — restoring the
    original daily visibility. Only log_prediction() (the accuracy
    tracking) stays gated on tier > 0, so Tier 0 days are still
    excluded from win-rate stats (per the 2026-07-29 fix that
    stopped Tier 0 from dragging live accuracy down to 43.5%), but
    no longer go silent/heartbeat-only. Sending and logging are now
    fully decoupled.
"""

import os, sys, json, warnings, requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

IL = ZoneInfo("Asia/Jerusalem")   # Israel timezone — used everywhere

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKER          = "ZW=F"
CORN_TICKER     = "ZC=F"
SOY_TICKER      = "ZS=F"
STOP_PCT        = 0.015   # 1.5%
TARGET_PCT      = 0.025   # 2.5%
MIN_CONFIDENCE  = 0.58
STATE_FILE      = Path("wheat_monitor_state.json")

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID")

# Years to exclude from seasonal calculation (global anomalies)
EXCLUDE_YEARS   = [2022]

# ── LIVE PRICE FETCH ──────────────────────────────────────────────────────────

def get_live_price(ticker=TICKER):
    """
    Fetches the actual live/last-traded price, separate from the
    daily historical bars used for indicators. Daily bars can lag
    by days around holidays/weekends or while today's session is
    still forming — this pulls the real current quote instead.

    UPDATED 2026-08-18: now tries the current front-month SPECIFIC
    contract (e.g. ZWU26.CBT) first, via get_front_month_ticker() —
    the same fix already proven for volume (see get_accurate_volume()
    below, confirmed 2026-07-10/11: ZW=F's continuous series has a
    real data lag/discrepancy vs. the specific contract). Falls back
    to the plain TICKER (ZW=F) only if the front-month fetch fails,
    with a clear log line either way. This was flagged as a real live
    discrepancy on 2026-08-18 — a Telegram alert showed 688c current
    price while the actual quote was 674.75c, an ~1.9% gap large
    enough to materially change the trade setup's entry/stop/target
    math, not just a display rounding issue.

    Returns (price, is_live). is_live=False means every live fetch
    attempt failed and the caller should fall back to the last daily
    close, while flagging it clearly rather than trusting it silently.
    """
    front_month = get_front_month_ticker()
    sources = [front_month] if ticker == front_month else [front_month, ticker]

    for i, t in enumerate(sources):
        try:
            fast = yf.Ticker(t).fast_info
            live = fast.get('last_price') or fast.get('lastPrice')
            if live and live > 0:
                if i > 0:
                    print(f"   ⚠️ Live price from FALLBACK source ({t}) — front-month ({front_month}) fetch failed")
                return float(live), True
        except Exception as e:
            print(f"   fast_info live price failed ({t}): {e}")

    # Fallback: try 1-minute intraday bars for today, same source order
    for t in sources:
        try:
            intraday = yf.Ticker(t).history(period='1d', interval='1m')
            if not intraday.empty:
                return float(intraday['Close'].iloc[-1]), True
        except Exception as e:
            print(f"   intraday fallback failed ({t}): {e}")

    return None, False


# ── SEASONAL ENGINE ───────────────────────────────────────────────────────────

class SeasonalEngine:
    """
    Derives the wheat seasonal calendar directly from 5 years of
    ZW=F price history. No hardcoded assumptions — the data speaks.
    Excludes 2022 (Ukraine war anomaly).
    """

    def __init__(self):
        self.seasonal_returns = None
        self.phase            = None
        self.bias             = 0.0
        self.confidence       = 0.0

    def fit(self, df):
        """Calculate average return by day-of-year across 5 years, excluding anomaly years."""
        df = df.copy()
        df['doy']  = df.index.dayofyear
        df['year'] = df.index.year
        df['ret1'] = df['Close'].pct_change(1)

        # Exclude anomaly years
        df = df[~df['year'].isin(EXCLUDE_YEARS)]

        # Average return by day-of-year
        self.seasonal_returns = df.groupby('doy')['ret1'].mean()

        # Smooth with 7-day rolling average
        self.seasonal_returns = self.seasonal_returns.rolling(7, center=True, min_periods=1).mean()

    def get_current_phase(self):
        """
        Returns seasonal phase for today based on historical patterns.
        Looks at next 20 trading days to determine trend direction.
        """
        if self.seasonal_returns is None:
            return {'phase': 'UNKNOWN', 'bias': 0.0, 'confidence': 0.0, 'explanation': 'No data'}

        today_doy = datetime.now(IL).timetuple().tm_yday

        # Look at next 20 days of seasonal returns
        forward_days = []
        for offset in range(1, 21):
            doy = ((today_doy + offset - 1) % 365) + 1
            if doy in self.seasonal_returns.index:
                forward_days.append(self.seasonal_returns[doy])

        if not forward_days:
            return {'phase': 'NEUTRAL', 'bias': 0.0, 'confidence': 0.5, 'explanation': 'No seasonal data'}

        avg_forward = np.mean(forward_days)
        pos_days    = sum(1 for r in forward_days if r > 0)
        neg_days    = sum(1 for r in forward_days if r < 0)

        # Determine phase
        if avg_forward > 0.0005 and pos_days >= 13:
            phase      = 'BULLISH'
            confidence = min(0.85, 0.60 + pos_days * 0.012)
        elif avg_forward < -0.0005 and neg_days >= 13:
            phase      = 'BEARISH'
            confidence = min(0.85, 0.60 + neg_days * 0.012)
        else:
            phase      = 'NEUTRAL'
            confidence = 0.55

        # Month labels
        month = datetime.now(IL).month
        labels = {
            1:'Jan neutral', 2:'Pre-spring dip', 3:'Spring rally starts',
            4:'Peak planting premium', 5:'Max weather premium',
            6:'Harvest pressure', 7:'Post-harvest low', 8:'Summer lull',
            9:'Fall recovery', 10:'Winter demand builds',
            11:'Pre-winter rally', 12:'Winter high'
        }

        self.phase      = phase
        self.bias       = avg_forward
        self.confidence = confidence

        return {
            'phase':       phase,
            'bias':        round(avg_forward, 5),
            'confidence':  round(confidence, 3),
            'pos_days':    pos_days,
            'neg_days':    neg_days,
            'explanation': labels.get(month, ''),
        }

    def blocks_direction(self, direction):
        """
        Hard seasonal override.
        If seasonal phase strongly disagrees with direction → block.
        This is the most important filter in the system.
        """
        if self.phase is None:
            return False, ""

        if direction == 'UP' and self.phase == 'BEARISH' and self.confidence >= 0.72:
            return True, f"Seasonal BEARISH phase blocks UP (confidence {self.confidence:.0%})"

        if direction == 'DOWN' and self.phase == 'BULLISH' and self.confidence >= 0.72:
            return True, f"Seasonal BULLISH phase blocks DOWN (confidence {self.confidence:.0%})"

        return False, ""


# ── TREND ENGINE ──────────────────────────────────────────────────────────────

class TrendEngine:
    """
    Determines the current trend from price action.
    Never fight a confirmed trend.
    """

    def get_trend(self, df):
        close = df['Close']
        price = float(close.iloc[-1])
        sma5  = float(close.rolling(5).mean().iloc[-1])
        sma10 = float(close.rolling(10).mean().iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])

        # Consecutive up/down days
        rets = close.pct_change()
        last5 = rets.iloc[-5:]
        up_days   = int((last5 > 0).sum())
        down_days = int((last5 < 0).sum())

        # Trend strength
        if price > sma5 > sma10 > sma20:
            trend     = 'UP'
            strength  = 'STRONG' if price > sma50 else 'MODERATE'
        elif price < sma5 < sma10 < sma20:
            trend     = 'DOWN'
            strength  = 'STRONG' if price < sma50 else 'MODERATE'
        else:
            trend     = 'NEUTRAL'
            strength  = 'WEAK'

        return {
            'trend':     trend,
            'strength':  strength,
            'price':     price,
            'sma5':      round(sma5, 2),
            'sma10':     round(sma10, 2),
            'sma20':     round(sma20, 2),
            'sma50':     round(sma50, 2),
            'up_days':   up_days,
            'down_days': down_days,
        }

    def blocks_direction(self, direction, trend_data):
        """Block signals that fight a strong confirmed trend."""
        if direction == 'DOWN' and trend_data['trend'] == 'UP' and trend_data['strength'] == 'STRONG':
            return True, f"Strong uptrend blocks DOWN (price {trend_data['price']:.1f} > all MAs)"
        if direction == 'UP' and trend_data['trend'] == 'DOWN' and trend_data['strength'] == 'STRONG':
            return True, f"Strong downtrend blocks UP (price {trend_data['price']:.1f} < all MAs)"
        return False, ""


# ── CONVICTION GATE ───────────────────────────────────────────────────────────

class ConvictionGate:
    """
    REBUILT 2026-07-09 using real train/holdout backtest validation
    (see backtest.py and backtest_results.json).
    UPDATED 2026-07-10: vol_low REMOVED after discovering a structural
    data problem, not just occasional bad data.

    The previous version of this class used combinations found by
    searching hundreds of condition combos against a single dataset,
    reporting the best result as "100% accuracy". That number was
    proven fake: when tested against a holdout period the combos
    were never fitted to, most either collapsed to ~55-62% (barely
    better than a coin flip) or never occurred at all in the last
    4 months of data.

    This version used ONLY single conditions that were individually
    validated on a real holdout set AND beat the baseline UP rate
    (67.46% — wheat trended up most of this 2-year window anyway,
    so anything below that adds zero real value, even if it "looks"
    high in isolation).

    2026-07-10 DISCOVERY — vol_low is unreliable, removed entirely:
    A diagnostic (volume_lag_check.py) showed ZW=F's Yahoo daily
    Volume field takes roughly 1-2 WEEKS to fully backfill for
    continuous futures contracts. Dates within the last ~10 days
    showed volume readings of single/low-double digits (e.g. 7, 48,
    136 contracts) on the most liquid wheat contract in the world —
    obviously incomplete, not real. Since vol_low (ratio < 0.80)
    compares TODAY's volume against a 20-day average that is ITSELF
    partly built from these same artificially-low recent values, it
    was almost certainly firing as effectively-always-true on recent
    dates — meaning its 84.8% holdout accuracy likely measured "is
    this date recent" rather than any real market behavior. This
    was NOT a rare data glitch — it is structural and will recur
    every single day this script runs. Removed rather than patched.

    CONFIRMED CONDITIONS (holdout period 2026-03-20 to 2026-07-09) —
    vol_low removed, the rest do not depend on the Volume field:
      momentum_up   : 84.0% UP (n=25 holdout) — strongest reliable signal
      macd_bullish  : 70.0% UP (n=30 holdout) — modest but real edge
      bearish_month : 68.0% UP (n=25 holdout) — barely above baseline, weak

    EXPLICITLY EXCLUDED (do not re-add without re-validating):
      vol_low        : REMOVED 2026-07-10 — structural data lag, not
                        a genuine volume signal (see above)
      rsi_oversold   : collapsed to 50.0% and flipped direction on holdout
      momentum_down  : collapsed to 57.7% and flipped direction on holdout
      near_bb_lower  : collapsed to 57.1% on tiny holdout sample (n=7)
      in_lower_half  : never occurred once in the entire holdout period
      wc_bullish, rsi_neutral, inside_bb, vol_good : held up on holdout
                        but scored BELOW the 67.46% baseline

    IMPORTANT: if volume data quality is ever fixed/verified reliable
    (e.g. switching to a direct CME/CBOT feed instead of Yahoo), re-run
    backtest.py fresh before re-adding any volume-based condition.
    Re-run backtest.py periodically and update the numbers below —
    do not let this drift stale.
    """

    # Holdout-validated accuracies — vol_low removed 2026-07-10 (see docstring)
    # FALLBACK values, used only if validated_conditions.json is missing
    # or invalid — see _load_holdout_accuracy() below for the real,
    # auto-updating source. Update these manually only as a last resort.
    FALLBACK_HOLDOUT_ACCURACY = {
        'momentum_up':   0.840,
        'macd_bullish':  0.700,
        'bearish_month': 0.680,
    }
    FALLBACK_BASELINE_UP = 0.6746

    # Never auto-trust these even if a loaded file somehow contains them —
    # defense in depth alongside backtest.py's own exclusion (see
    # backtest.py's STRUCTURAL_EXCLUSIONS for why: confirmed structural
    # Yahoo volume data lag for ZW=F, 2026-07-10).
    STRUCTURAL_EXCLUSIONS = {'vol_low', 'vol_good', 'vol_high'}

    def __init__(self):
        self.HOLDOUT_ACCURACY, self.BASELINE_UP, self._source = self._load_holdout_accuracy()

    def _load_holdout_accuracy(self):
        """
        Loads validated_conditions.json (auto-produced weekly by
        backtest.py) if present and valid. Falls back to the hardcoded
        FALLBACK_* values otherwise, with a clear console message so
        it's never a silent, invisible fallback.
        """
        path = Path("validated_conditions.json")
        if not path.exists():
            print("   ⚠️ validated_conditions.json not found — using hardcoded fallback accuracy values")
            return dict(self.FALLBACK_HOLDOUT_ACCURACY), self.FALLBACK_BASELINE_UP, "FALLBACK (no file)"

        try:
            data = json.loads(path.read_text())
            loaded = data.get('validated_conditions', {})
            baseline = data.get('baseline_up', self.FALLBACK_BASELINE_UP)

            # Defense in depth: strip any structurally-excluded condition
            # even if it somehow made it into the file
            cleaned = {k: v for k, v in loaded.items() if k not in self.STRUCTURAL_EXCLUSIONS}
            removed = set(loaded.keys()) & self.STRUCTURAL_EXCLUSIONS
            if removed:
                print(f"   ⚠️ Ignored structurally-excluded condition(s) found in file: {removed}")

            if not cleaned:
                print("   ⚠️ validated_conditions.json had no usable conditions — using hardcoded fallback")
                return dict(self.FALLBACK_HOLDOUT_ACCURACY), self.FALLBACK_BASELINE_UP, "FALLBACK (empty file)"

            generated_at = data.get('generated_at', 'unknown date')
            print(f"   Loaded {len(cleaned)} validated condition(s) from validated_conditions.json "
                  f"(generated {generated_at})")
            return cleaned, baseline, f"LIVE (generated {generated_at})"

        except Exception as e:
            print(f"   ⚠️ Failed to load validated_conditions.json ({e}) — using hardcoded fallback")
            return dict(self.FALLBACK_HOLDOUT_ACCURACY), self.FALLBACK_BASELINE_UP, "FALLBACK (load error)"

    def evaluate(self, df):
        close = df['Close']
        price = float(close.iloc[-1])

        # NOTE: vol_low removed 2026-07-10 — see class docstring.
        # ZW=F's Yahoo volume field is structurally unreliable for
        # dates within ~1-2 weeks (takes that long to backfill), so
        # this condition was almost always firing on incomplete data,
        # not a genuine low-volume signal.

        # Momentum (for momentum_up) — same-direction 1d and 3d returns
        ret_1d = float(close.pct_change(1).iloc[-1])
        ret_3d = float(close.pct_change(3).iloc[-1])
        momentum_up = (ret_1d > 0) and (ret_3d > 0)

        # MACD (for macd_bullish)
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        macd_bullish = float(macd.iloc[-1]) > float(macd_signal.iloc[-1])

        # Month (for bearish_month)
        month = datetime.now(IL).month
        bearish_month = month in [6, 7, 8]

        conditions = {
            'momentum_up':   momentum_up,
            'macd_bullish':  macd_bullish,
            'bearish_month': bearish_month,
            'ret_1d':        round(ret_1d, 4),
            'ret_3d':        round(ret_3d, 4),
            'month':         month,
            'price':         round(price, 2),
        }

        # Rank active conditions by their real holdout accuracy — highest wins
        active = [(name, acc) for name, acc in self.HOLDOUT_ACCURACY.items()
                  if conditions.get(name)]

        if not active:
            tier, accuracy = 0, self.BASELINE_UP
            reason = f"⚪ NO SIGNAL — baseline only ({self.BASELINE_UP:.1%}, no validated condition active)"
        else:
            active.sort(key=lambda x: x[1], reverse=True)
            best_name, best_acc = active[0]
            active_names = " + ".join(name for name, _ in active)

            if best_acc >= 0.80:
                tier = 2
                reason = f"🥇 TIER 2 (holdout-validated {best_acc:.1%}) — {active_names}"
            elif best_acc >= 0.68:
                tier = 1
                reason = f"🥈 TIER 1 (holdout-validated {best_acc:.1%}) — {active_names}"
            else:
                tier = 0
                reason = f"⚪ WEAK — {active_names} ({best_acc:.1%}, near baseline)"
            accuracy = best_acc

        return tier, accuracy, reason, conditions


MONTHLY_CACHE_FILE = Path("monthly_range_cache.json")
MONTHLY_BREAK_LOG_FILE = Path("monthly_break_log.json")
# Same 0.1% "even a small breach counts" philosophy as weekly's
# BREAK_THRESHOLD_PCT (see that constant's comment) — kept as its own
# named constant rather than reusing BREAK_THRESHOLD_PCT so the two
# timeframes can be tuned independently later if needed.
MONTHLY_BREAK_THRESHOLD_PCT = 0.001


def log_monthly_break(month_key, current_price, old_monthly, reason):
    """Records when/why a monthly outlook range got breached and
    regenerated — mirrors log_weekly_break() exactly, see that
    function. New file, monthly_break_log.json, parallel to
    weekly_break_log.json."""
    log = []
    if MONTHLY_BREAK_LOG_FILE.exists():
        try:
            log = json.loads(MONTHLY_BREAK_LOG_FILE.read_text())
        except Exception:
            log = []

    log.append({
        'month_key': month_key,
        'broken_at': datetime.now(IL).isoformat(),
        'price_at_break': round(current_price, 2),
        'old_range': f"{old_monthly['monthly_low']:.0f}-{old_monthly['monthly_high']:.0f}",
        'reason': reason,
    })

    try:
        MONTHLY_BREAK_LOG_FILE.write_text(json.dumps(log, indent=2))
    except Exception as e:
        print(f"   Failed to log monthly break: {e}")


def get_frozen_monthly_range(wre, df, current_price, cost_floor_cents):
    """
    UPDATED 2026-08-19: previously this ONLY regenerated when the
    calendar flipped to a new month — it never checked whether
    CURRENT PRICE had actually moved outside the frozen
    [monthly_low, monthly_high] range in between. Confirmed live:
    August's frozen range (594-669c) stayed displayed unchanged even
    after price reached 681c — a real, meaningful breach with no
    mechanism to catch it, unlike the weekly plan (get_frozen_weekly_
    plan) which already re-checks for a break on every single run.
    This was inconsistent with the project's own core design ("weekly
    frozen unless broken, then regenerate, since it's the most
    reliable") — monthly is meant to follow the same rule, just on
    its own timeframe/tolerance, not skip breach-checking entirely.

    Now mirrors get_frozen_weekly_plan()'s pattern: on each run,
    checks current_price against the frozen range (with
    MONTHLY_BREAK_THRESHOLD_PCT tolerance); if breached, logs it
    (log_monthly_break(), new monthly_break_log.json) and regenerates
    a fresh range centered on current price, still frozen at the
    calendar-month cache_key so it doesn't over-regenerate.
    """
    today = datetime.now(IL)
    cache_key = f"{today.year}-{today.month:02d}"

    cached = None
    if MONTHLY_CACHE_FILE.exists():
        try:
            cached = json.loads(MONTHLY_CACHE_FILE.read_text())
        except Exception:
            cached = None

    if cached and cached.get('month_key') == cache_key:
        old_monthly = cached['monthly']
        low, high = old_monthly.get('monthly_low'), old_monthly.get('monthly_high')

        breached = False
        if low is not None and high is not None:
            if current_price > high * (1 + MONTHLY_BREAK_THRESHOLD_PCT):
                breached, reason = True, f"price {current_price:.0f}c broke above monthly high {high:.0f}c"
            elif current_price < low * (1 - MONTHLY_BREAK_THRESHOLD_PCT):
                breached, reason = True, f"price {current_price:.0f}c broke below monthly low {low:.0f}c"

        if breached:
            print(f"   ⚠️ MONTHLY RANGE BREACHED: {reason} — regenerating")
            log_monthly_break(cache_key, current_price, old_monthly, reason)

            monthly = wre.predict_monthly_range(df, current_price, cost_floor_cents)
            if monthly:
                try:
                    MONTHLY_CACHE_FILE.write_text(json.dumps({
                        'month_key': cache_key,
                        'frozen_at': today.isoformat(),
                        'regenerated_after_breach': True,
                        'monthly': monthly,
                    }, indent=2))
                    print(f"   Re-froze monthly range after breach: "
                          f"{monthly['monthly_low']:.0f}-{monthly['monthly_high']:.0f}c")
                except Exception as e:
                    print(f"   Failed to cache regenerated monthly range: {e}")
            return monthly

        print(f"   Using FROZEN monthly range (locked earlier this month, {cache_key})")
        return old_monthly

    monthly = wre.predict_monthly_range(df, current_price, cost_floor_cents)
    if monthly:
        try:
            MONTHLY_CACHE_FILE.write_text(json.dumps({
                'month_key': cache_key,
                'frozen_at': today.isoformat(),
                'monthly': monthly,
            }, indent=2))
            print(f"   Froze NEW monthly range for {cache_key}: "
                  f"{monthly['monthly_low']:.0f}-{monthly['monthly_high']:.0f}c")
        except Exception as e:
            print(f"   Failed to cache monthly range: {e}")

    return monthly


BREACH_TOLERANCE_PCT = 0.02  # 2% beyond stop/target triggers a re-freeze — adjust here


def _check_breach(weekly, current_price, direction):
    """
    Returns (is_breached, reason). A forecast is "broken" when price
    has moved BREACH_TOLERANCE_PCT beyond either the frozen target
    (forecast already achieved/exceeded) or the frozen stop (forecast
    invalidated) — not just touched, to avoid re-freezing on noise.
    """
    target = weekly.get('target')
    stop = weekly.get('stop')
    if target is None or stop is None:
        return False, ""

    if direction == 'UP':
        target_breach = current_price >= target * (1 + BREACH_TOLERANCE_PCT)
        stop_breach   = current_price <= stop * (1 - BREACH_TOLERANCE_PCT)
    elif direction == 'DOWN':
        target_breach = current_price <= target * (1 - BREACH_TOLERANCE_PCT)
        stop_breach   = current_price >= stop * (1 + BREACH_TOLERANCE_PCT)
    else:
        return False, ""

    if target_breach:
        return True, f"price {current_price:.0f}c is {BREACH_TOLERANCE_PCT:.0%}+ past frozen target {target:.0f}c"
    if stop_breach:
        return True, f"price {current_price:.0f}c is {BREACH_TOLERANCE_PCT:.0%}+ past frozen stop {stop:.0f}c"
    return False, ""


NEWS_SIGNAL_MAX_AGE_HOURS = 12  # only trust a signal this fresh


def get_news_signal():
    """
    Reads the most recent LLM news interpretation from news_log.json
    (produced by news_scanner.py's macro/commodity Gemini analysis)
    if it's fresh enough. Returns None if missing, stale, or NEUTRAL —
    caller should treat None as "no news nudge this run", not guess.

    UPDATED 2026-07-28: news_scanner.py was rewritten to a broader
    macro/commodity scanner writing news_log.json (not the original
    news_signal_log.json), with entries newest-first and wheat signal
    nested under llm_analysis.wheat_impact — which appears as either
    a string ("BULLISH"/"BEARISH"/"NEUTRAL") or a dict
    ({"direction": ..., "reason": ...}) depending on how the model
    formatted its JSON that run. This function normalizes both shapes.
    The new scanner does not emit a numeric confidence, so a fixed
    moderate confidence (60) is used — kept deliberately unremarkable
    since, same as before, this signal is unvalidated and gets only a
    small nudge weight in predict_next_week() (see NEWS_SIGNAL_NUDGE_SCALE
    there). Do not treat this fixed value as a real confidence score.

    WEIGHT NOTE (2026-07-19, still applies): this is a brand new,
    UNVALIDATED signal. Do not increase its nudge weight based on a
    good week or two; check score_news_signals.py's real win rate
    over many scored signals first.
    """
    path = Path("news_log.json")
    if not path.exists():
        return None
    try:
        log = json.loads(path.read_text())
        if not log:
            return None

        latest = log[0]  # newest-first (news_scanner.py inserts at index 0)
        ts = datetime.fromisoformat(latest['timestamp'])
        age_hours = (datetime.now(IL) - ts).total_seconds() / 3600
        if age_hours > NEWS_SIGNAL_MAX_AGE_HOURS:
            return None

        analysis = latest.get('llm_analysis')
        if not analysis:
            return None

        wheat_impact = analysis.get('wheat_impact')
        if isinstance(wheat_impact, dict):
            signal = wheat_impact.get('direction', 'NEUTRAL')
        else:
            signal = wheat_impact or 'NEUTRAL'
        signal = str(signal).upper()

        if signal not in ('BULLISH', 'BEARISH'):
            return None  # NEUTRAL or unrecognized — no nudge this run

        confidence = 60  # fixed — see docstring; new scanner has no numeric confidence
        return signal, confidence

    except Exception as e:
        print(f"   Failed to read news signal: {e}")
        return None


WEEKLY_CACHE_FILE = Path("weekly_range_cache.json")
WEEKLY_BREAK_LOG_FILE = Path("weekly_break_log.json")
WEEKLY_PERFORMANCE_LOG_FILE = Path("weekly_performance_log.json")

# How far past the frozen stop/target price must move before the
# weekly forecast is considered "broken" and gets regenerated.
# UPDATED 2026-07-14: changed from 2% to 0.1% per explicit design
# decision — even a small breach now triggers regeneration, rather
# than waiting for a larger, more clearly-confirmed break.
BREAK_THRESHOLD_PCT = 0.001


def log_daily_performance(iso_year, iso_week, current_price, weekly):
    """
    Appends today's price vs. this week's frozen range to a running
    log, so a Friday report can show a real day-by-day breakdown (not
    a memory-based impression) of how the week's forecast actually
    held up.

    UPDATED 2026-09-05, real bug found and confirmed — this used to
    also log on Saturday (see weekly_report.py's original docstring:
    "so a Friday/Saturday report can show..."), but Saturday isn't a
    real CBOT trading day. Whatever "current_price" is available on a
    Saturday is just Friday's stale closing price re-logged under a
    new day label — not a real new data point. This produced a fake
    "6th day" in weekly_report.py's day-by-day breakdown and inflated
    its day-count math. Guarded here (inside the function itself,
    rather than at each of this function's 3 call sites) so it can
    never happen regardless of which code path calls this.
    """
    from trading_calendar import is_trading_day
    if not is_trading_day(datetime.now(IL)):
        return

    log = []
    if WEEKLY_PERFORMANCE_LOG_FILE.exists():
        try:
            log = json.loads(WEEKLY_PERFORMANCE_LOG_FILE.read_text())
        except Exception:
            log = []

    iso_key = f"{iso_year}-W{iso_week}"
    today = datetime.now(IL)
    today_str = today.strftime('%Y-%m-%d')

    # Don't double-log if this script runs more than once on the same day
    if any(e['iso_key'] == iso_key and e['date'] == today_str for e in log):
        return

    range_low  = weekly['range_low']
    range_high = weekly['range_high']
    range_width = range_high - range_low
    if range_width > 0:
        position_pct = round(((current_price - range_low) / range_width) * 100, 1)
    else:
        position_pct = None

    within_range = range_low <= current_price <= range_high

    log.append({
        'iso_key': iso_key,
        'date': today_str,
        'day_name': today.strftime('%A'),
        'price': round(current_price, 2),
        'range_low': range_low,
        'range_high': range_high,
        'position_in_range_pct': position_pct,
        'within_range': within_range,
        'bias': weekly.get('bias'),
    })

    try:
        WEEKLY_PERFORMANCE_LOG_FILE.write_text(json.dumps(log, indent=2))
    except Exception as e:
        print(f"   Failed to log daily performance: {e}")


def log_weekly_break(iso_year, iso_week, current_price, old_weekly, reason):
    """Records when/why a weekly forecast got broken and regenerated."""
    log = []
    if WEEKLY_BREAK_LOG_FILE.exists():
        try:
            log = json.loads(WEEKLY_BREAK_LOG_FILE.read_text())
        except Exception:
            log = []

    log.append({
        'iso_key': f"{iso_year}-W{iso_week}",
        'broken_at': datetime.now(IL).isoformat(),
        'price_at_break': round(current_price, 2),
        'old_range': f"{old_weekly['range_low']:.0f}-{old_weekly['range_high']:.0f}",
        'old_stop': old_weekly.get('stop'),
        'old_target': old_weekly.get('target'),
        'reason': reason,
    })

    try:
        WEEKLY_BREAK_LOG_FILE.write_text(json.dumps(log, indent=2))
    except Exception as e:
        print(f"   Failed to log weekly break: {e}")


def get_frozen_weekly_plan(wre, df, current_price, cost_floor_cents, daily_direction,
                            backtest_tier=None, backtest_accuracy=None, news_signal=None):
    """
    REBUILT 2026-07-14, corrected same day: entry/stop/target now come
    from predict_next_week()'s OWN real historical/ATR-based forecast
    (clamped to a -15%/+25% outer safety boundary), NOT a hardcoded
    fixed-percentage formula. An earlier version of this fix mistakenly
    always used exactly -15%/+25% as the forecast itself; corrected to
    treat those as outer limits only — the model's real forecast can
    (and usually should) be narrower.

    - weekly['final_call'] is the FROZEN weekly direction shown as the
      top-level header — only changes when the current setup breaks.
    - On break: WIN (broke past target) → same direction, new real
      forecast around current price. LOSS (broke past stop) → flip
      direction, new real forecast for the flipped direction.
    - forced_direction is passed to predict_next_week() so the
      regenerated forecast still uses real seasonal/ATR data, just
      pinned to the win/loss-determined direction rather than
      whatever the data would have picked on its own.
    - On a fresh week, NO forced_direction — the model's own bias
      score (now also nudged by today's daily_direction_hint) decides
      final_call, rather than blindly copying the daily ensemble read.
    """
    iso_year, iso_week, _ = datetime.now(IL).isocalendar()

    cached = None
    if WEEKLY_CACHE_FILE.exists():
        try:
            cached = json.loads(WEEKLY_CACHE_FILE.read_text())
        except Exception:
            cached = None

    if cached and cached.get('iso_year') == iso_year and cached.get('iso_week') == iso_week:
        old_weekly = cached['weekly']
        stop   = old_weekly.get('stop')
        target = old_weekly.get('target')
        final_call = old_weekly.get('final_call', daily_direction)
        frozen_at = cached.get('frozen_at')

        broken = False
        break_type = None  # 'target' (win) or 'stop' (loss)
        breach_price = current_price  # may be replaced below by a more extreme real daily High/Low
        if stop is not None and target is not None:
            if final_call == 'UP':
                if current_price > target * (1 + BREAK_THRESHOLD_PCT):
                    broken, break_type = True, 'target'
                elif current_price < stop * (1 - BREAK_THRESHOLD_PCT):
                    broken, break_type = True, 'stop'
            else:  # DOWN
                if current_price < target * (1 - BREAK_THRESHOLD_PCT):
                    broken, break_type = True, 'target'
                elif current_price > stop * (1 + BREAK_THRESHOLD_PCT):
                    broken, break_type = True, 'stop'

        # UPDATED 2026-09-05, real bug found and confirmed by bug_detector.py:
        # the check above only ever compared the LIVE snapshot price at
        # whatever moment a script happened to run — it could miss a real
        # break that happened between checks and reverted before the next
        # one. First fix attempt checked intraday High/Low, but that's
        # asymmetric in practice: a stop wick that reverts by end of day
        # was watched happening in real time and should NOT count as a
        # break, same as a target wick that reverts shouldn't count as a
        # win. Corrected to check each trading day's CLOSE since this
        # setup was frozen, not the intraday High/Low — a brief touch
        # that reverts by end of day doesn't count either way; a breach
        # still true at close is real, for either direction. Same rule
        # for WIN and LOSS, no favoritism.
        if not broken and stop is not None and target is not None and frozen_at:
            frozen_date = datetime.fromisoformat(frozen_at).date()
            bars_since_freeze = df[df.index.date >= frozen_date]
            for close_date, bar in bars_since_freeze.iterrows():
                close_price = float(bar['Close'])
                if final_call == 'UP':
                    if close_price > target * (1 + BREAK_THRESHOLD_PCT):
                        broken, break_type, breach_price = True, 'target', close_price
                    elif close_price < stop * (1 - BREAK_THRESHOLD_PCT):
                        broken, break_type, breach_price = True, 'stop', close_price
                else:  # DOWN
                    if close_price < target * (1 - BREAK_THRESHOLD_PCT):
                        broken, break_type, breach_price = True, 'target', close_price
                    elif close_price > stop * (1 + BREAK_THRESHOLD_PCT):
                        broken, break_type, breach_price = True, 'stop', close_price
                if broken:
                    print(f"   ⚠️ Breach found via daily CLOSE check ({close_date.date()} "
                          f"close={close_price:.2f}c), not caught by live snapshot checks")
                    break

        if broken:
            outcome = 'WIN' if break_type == 'target' else 'LOSS'
            reason = f"price {breach_price:.0f}c broke past {break_type} ({outcome})"
            print(f"   ⚠️ WEEKLY SETUP BROKEN: {reason} — regenerating")
            log_weekly_break(iso_year, iso_week, breach_price, old_weekly, reason)

            # Win → keep same direction, fresh real forecast.
            # Loss → the directional read was wrong, flip it.
            new_final_call = final_call if break_type == 'target' else (
                'DOWN' if final_call == 'UP' else 'UP')

            weekly = wre.predict_next_week(
                df, current_price, cost_floor_cents,
                forced_direction=new_final_call,
                daily_direction_hint=daily_direction,
                backtest_tier=backtest_tier,
                backtest_accuracy=backtest_accuracy,
                news_signal=news_signal,
            )
            weekly['final_call'] = new_final_call

            history = old_weekly.get('history', [])
            history.append({
                'closed_at': datetime.now(IL).isoformat(),
                'entry': old_weekly.get('entry'), 'stop': stop, 'target': target,
                'final_call': final_call, 'outcome': outcome,
                'price_at_close': round(current_price, 2),
            })
            weekly['history'] = history

            try:
                WEEKLY_CACHE_FILE.write_text(json.dumps({
                    'iso_year': iso_year, 'iso_week': iso_week,
                    'frozen_at': datetime.now(IL).isoformat(),
                    'broken_and_regenerated': True,
                    'weekly': weekly,
                }, indent=2))
                print(f"   Re-froze weekly plan after {outcome}: final_call={new_final_call}, "
                      f"entry={weekly['entry']:.0f} stop={weekly['stop']:.0f} target={weekly['target']:.0f} "
                      f"(range {weekly['range_low']:.0f}-{weekly['range_high']:.0f})")
            except Exception as e:
                print(f"   Failed to cache regenerated weekly plan: {e}")

            log_daily_performance(iso_year, iso_week, current_price, weekly)
            return weekly, True, outcome

        print(f"   Using FROZEN weekly plan (locked earlier this week, iso {iso_year}-W{iso_week})")
        weekly = dict(old_weekly)
        log_daily_performance(iso_year, iso_week, current_price, weekly)
        in_range = weekly['stop'] <= current_price <= weekly['target'] if weekly['final_call'] == 'UP' \
                   else weekly['target'] <= current_price <= weekly['stop']
        return weekly, not in_range, None

    # New week — let the model's OWN real forecast (nudged by today's
    # daily direction, not overridden by it) determine final_call
    weekly = wre.predict_next_week(
        df, current_price, cost_floor_cents,
        daily_direction_hint=daily_direction,
        backtest_tier=backtest_tier,
        backtest_accuracy=backtest_accuracy,
        news_signal=news_signal,
    )
    weekly['final_call'] = weekly['bias'] if weekly['bias'] in ('UP', 'DOWN') else daily_direction
    weekly['history'] = []

    try:
        WEEKLY_CACHE_FILE.write_text(json.dumps({
            'iso_year': iso_year, 'iso_week': iso_week,
            'frozen_at': datetime.now(IL).isoformat(),
            'weekly': weekly,
        }, indent=2))
        print(f"   Froze NEW weekly plan for iso {iso_year}-W{iso_week}: "
              f"final_call={weekly['final_call']}, entry={weekly['entry']:.0f} "
              f"stop={weekly['stop']:.0f} target={weekly['target']:.0f} "
              f"(range {weekly['range_low']:.0f}-{weekly['range_high']:.0f})")
    except Exception as e:
        print(f"   Failed to cache weekly plan: {e}")

    log_daily_performance(iso_year, iso_week, current_price, weekly)
    return weekly, False, None


# ── INDICATORS ────────────────────────────────────────────────────────────────

def add_indicators(df):
    df = df.copy()
    # Preserve corn column if present before any operations
    corn_close = df['Corn_Close'].copy() if 'Corn_Close' in df.columns else None

    df['Returns']    = df['Close'].pct_change()
    df['SMA_20']     = df['Close'].rolling(20).mean()
    df['SMA_50']     = df['Close'].rolling(50).mean()
    df['EMA_12']     = df['Close'].ewm(span=12).mean()
    df['EMA_26']     = df['Close'].ewm(span=26).mean()
    df['MACD']       = df['EMA_12'] - df['EMA_26']
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI']        = 100 - (100 / (1 + gain / loss))
    bb_mid           = df['Close'].rolling(20).mean()
    bb_std           = df['Close'].rolling(20).std()
    df['BB_Upper']   = bb_mid + 2 * bb_std
    df['BB_Lower']   = bb_mid - 2 * bb_std
    df['BB_Width']   = (bb_std * 2) / bb_mid
    df['Volatility'] = df['Returns'].rolling(20).std()
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    df['ATR']        = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df = df.dropna()

    # Re-attach corn after dropna (forward-fill any gaps)
    if corn_close is not None:
        df['Corn_Close'] = corn_close.reindex(df.index, method='ffill')

    return df


# ── ENSEMBLE MODELS ───────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Three models with daily-sensitive features.
    No frozen predictions — all three retrain fresh each run.
    """

    def __init__(self):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler_lstm = MinMaxScaler()
        self.scaler_ml   = MinMaxScaler()
        self.lstm_model  = None
        self.rf_model    = None
        self.xgb_model   = None
        self.seq_len     = 60
        self.features    = [
            'Close', 'Volume', 'Returns', 'SMA_20', 'SMA_50',
            'RSI', 'MACD', 'BB_Width', 'Volatility', 'ATR'
        ]

    def train(self, df):
        from keras.models import Sequential
        from keras.layers import LSTM as KerasLSTM, Dense, Dropout
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb

        print("   Training LSTM + RF + XGB...")

        # Labels: did price go up next day?
        y = np.array([
            1 if df['Close'].iloc[i] > df['Close'].iloc[i-1] else 0
            for i in range(self.seq_len, len(df))
        ])

        # LSTM data
        data_lstm   = df[self.features].values
        scaled_lstm = self.scaler_lstm.fit_transform(data_lstm)
        X_lstm      = np.array([scaled_lstm[i-self.seq_len:i] for i in range(self.seq_len, len(scaled_lstm))])

        # ML features — daily-sensitive, not frozen 60-day window
        ml_feat = self._build_ml_features(df)
        n       = len(y)
        ml_feat = ml_feat.iloc[-n:]
        X_ml    = self.scaler_ml.fit_transform(ml_feat.fillna(0))

        # Train LSTM
        self.lstm_model = Sequential([
            KerasLSTM(64, return_sequences=True, input_shape=(self.seq_len, len(self.features))),
            Dropout(0.2),
            KerasLSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1,  activation='sigmoid')
        ])
        self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
        self.lstm_model.fit(X_lstm, y, epochs=25, batch_size=32, validation_split=0.15, verbose=0)

        # Daily seed so RF/XGB vary each day
        seed = datetime.now(IL).timetuple().tm_yday

        self.rf_model = RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=5,
            random_state=seed, n_jobs=-1
        )
        self.rf_model.fit(X_ml, y)

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            random_state=seed, use_label_encoder=False, eval_metric='logloss'
        )
        self.xgb_model.fit(X_ml, y, verbose=False)

        print("   ✓ All models trained")

    def _build_ml_features(self, df):
        f = pd.DataFrame(index=df.index)
        f['ret_1d']      = df['Close'].pct_change(1)
        f['ret_3d']      = df['Close'].pct_change(3)
        f['ret_5d']      = df['Close'].pct_change(5)
        f['ret_10d']     = df['Close'].pct_change(10)
        f['ret_20d']     = df['Close'].pct_change(20)
        sma5             = df['Close'].rolling(5).mean()
        sma10            = df['Close'].rolling(10).mean()
        f['sma5_vs_20']  = sma5  / df['SMA_20'] - 1
        f['sma10_vs_50'] = sma10 / df['SMA_50'] - 1
        f['above_sma20'] = (df['Close'] > df['SMA_20']).astype(float)
        f['above_sma50'] = (df['Close'] > df['SMA_50']).astype(float)
        f['rsi']         = df['RSI']
        f['rsi_change']  = df['RSI'].diff(3)
        f['macd']        = df['MACD']
        f['macd_change'] = df['MACD'].diff(3)
        f['atr_pct']     = df['ATR'] / df['Close']
        f['bb_width']    = df['BB_Width']
        vol_avg          = df['Volume'].rolling(20).mean()
        f['vol_ratio']   = df['Volume'] / vol_avg
        high10           = df['High'].rolling(10).max()
        low10            = df['Low'].rolling(10).min()
        f['range_pos']   = (df['Close'] - low10) / (high10 - low10 + 1e-6)
        f['volatility']  = df['Volatility']

        # ── Corn inter-market features (if available) ──
        # When wheat/corn ratio is high → wheat expensive vs corn → bearish wheat
        # When corn is rising → acreage competition → bullish wheat
        if 'Corn_Close' in df.columns:
            corn_close         = df['Corn_Close']
            wc_ratio           = df['Close'] / corn_close.replace(0, np.nan)
            wc_ratio_mean      = wc_ratio.rolling(60).mean()
            wc_ratio_std       = wc_ratio.rolling(60).std().replace(0, np.nan)
            f['wc_ratio_z']    = (wc_ratio - wc_ratio_mean) / wc_ratio_std
            f['corn_mom_3d']   = corn_close.pct_change(3)
            f['corn_mom_5d']   = corn_close.pct_change(5)

        return f.dropna()

    def predict(self, df):
        """
        REBUILT 2026-07-09 — fixed a real bug in how model outputs
        were combined.

        OLD BEHAVIOR (removed): each model's "weight" was set to
        abs(prediction - 0.5) — meaning the MORE EXTREME a model's
        guess, the MORE it controlled the final answer. On a real
        alert (2026-07-09), this meant XGB's 0.001 (essentially "0%
        chance", more likely a miscalibrated/overconfident output
        than genuine certainty) got ~50% of the total decision
        weight, while LSTM's honest, moderate 0.481 (near a genuine
        coin flip) got under 2% influence. Combined with an "all
        models agree" bonus, this produced a fake 92% confidence
        built almost entirely on the single most extreme number.
        Two days earlier, the same mechanism had produced a 92%+
        confidence in the OPPOSITE direction (RF/XGB near 0.95-0.998
        UP) — proof the models are unstable day to day, and the old
        formula was amplifying that instability into false certainty
        instead of damping it.

        NEW BEHAVIOR: equal weighting (simple average) — no model's
        opinion counts more just because it's extreme. Confidence is
        reported honestly, and real disagreement between models is
        surfaced explicitly (reliability flag) instead of hidden
        behind an agreement bonus.
        """
        # LSTM
        data   = df[self.features].values
        scaled = self.scaler_lstm.transform(data)
        X_lstm = np.array([scaled[-self.seq_len:]])
        lstm_p = float(self.lstm_model.predict(X_lstm, verbose=0)[0][0])

        # RF + XGB
        feat  = self._build_ml_features(df).iloc[[-1]]
        X_ml  = self.scaler_ml.transform(feat.fillna(0))
        rf_p  = float(self.rf_model.predict_proba(X_ml)[0][1])
        xgb_p = float(self.xgb_model.predict_proba(X_ml)[0][1])

        preds = [lstm_p, rf_p, xgb_p]

        # Equal-weighted average — no model gets extra say for being extreme
        weighted = float(np.mean(preds))

        votes_up = sum(1 for p in preds if p >= 0.5)
        direction = 'UP' if weighted >= 0.5 else 'DOWN'
        confidence = weighted if weighted >= 0.5 else 1 - weighted

        # Real disagreement measure — how spread out are the 3 opinions?
        spread = float(np.std(preds))

        agreement = 'FULL' if votes_up in [0, 3] else 'MAJORITY' if votes_up in [1, 2] else 'SPLIT'

        # If models disagree substantially, that's real information —
        # cap confidence instead of letting one extreme model dominate.
        # A high spread means "the models don't actually know," which
        # should LOWER stated confidence, not get averaged away.
        reliability = 'LOW' if spread > 0.35 else 'MODERATE' if spread > 0.15 else 'HIGH'
        if reliability == 'LOW':
            confidence = min(confidence, 0.60)  # don't claim high confidence when models sharply disagree

        return {
            'direction':   direction,
            'confidence':  confidence,
            'lstm':        lstm_p,
            'rf':          rf_p,
            'xgb':         xgb_p,
            'weighted':    weighted,
            'votes_up':    votes_up,
            'agreement':   agreement,
            'spread':      round(spread, 3),
            'reliability': reliability,
        }


# ── WASDE MULTI-GRAIN ─────────────────────────────────────────────────────────

def get_wasde_signal():
    """Fetch wheat, corn, soy from USDA. Derive wheat signal from all three."""
    api_key  = os.getenv("USDA_API_KEY", "3338B84E-694D-3E6A-925C-F35064C59BAE")
    base_url = "https://quickstats.nass.usda.gov/api/api_GET/"

    ANNUAL_USE = {'WHEAT': 2e9, 'CORN': 14.5e9, 'SOYBEANS': 4.4e9}
    STU_TIGHT  = {'WHEAT': 0.30, 'CORN': 0.10, 'SOYBEANS': 0.07}
    STU_AMPLE  = {'WHEAT': 0.33, 'CORN': 0.13, 'SOYBEANS': 0.10}

    grain_stu = {}
    for grain in ['WHEAT', 'CORN', 'SOYBEANS']:
        try:
            r = requests.get(base_url, params={
                'key': api_key, 'source_desc': 'SURVEY',
                'commodity_desc': grain, 'class_desc': 'ALL CLASSES',
                'statisticcat_desc': 'STOCKS', 'unit_desc': 'BU',
                'agg_level_desc': 'NATIONAL', 'format': 'JSON', 'year__GE': 2021,
            }, timeout=15)
            if r.status_code == 200:
                records = r.json().get('data', [])
                if records:
                    records = sorted(records, key=lambda x: x.get('year', 0), reverse=True)
                    val = float(records[0]['Value'].replace(',', ''))
                    grain_stu[grain] = val / ANNUAL_USE[grain]
        except Exception:
            pass

    if not grain_stu.get('WHEAT'):
        # Fallback: use wheat/corn ratio from yfinance
        return _wasde_market_proxy()

    score   = 0.0
    factors = []
    w_stu   = grain_stu['WHEAT']

    if w_stu < STU_TIGHT['WHEAT']:
        score += 0.20; factors.append(f"Wheat tight ({w_stu:.1%} STU)")
    elif w_stu > STU_AMPLE['WHEAT']:
        score -= 0.15; factors.append(f"Wheat ample ({w_stu:.1%} STU)")
    else:
        factors.append(f"Wheat balanced ({w_stu:.1%} STU)")

    for grain in ['CORN', 'SOYBEANS']:
        if grain in grain_stu:
            stu = grain_stu[grain]
            if stu < STU_TIGHT[grain]:
                score += 0.06; factors.append(f"{grain.title()} tight → acre competition")
            elif stu > STU_AMPLE[grain]:
                score -= 0.03

    signal = 'BULLISH' if score > 0.10 else 'BEARISH' if score < -0.05 else 'NEUTRAL'
    return {'signal': signal, 'score': round(score, 4),
            'stu': w_stu, 'factors': factors[:2], 'source': 'USDA LIVE'}


def _wasde_market_proxy():
    """Fallback: wheat/corn + wheat/soy ratio z-scores."""
    try:
        end   = datetime.now(IL)
        start = end - timedelta(days=400)
        wdf   = yf.Ticker(TICKER).history(start=start, end=end, auto_adjust=False)
        cdf   = yf.Ticker(CORN_TICKER).history(start=start, end=end, auto_adjust=False)

        if wdf.empty or cdf.empty:
            return {'signal': 'NEUTRAL', 'score': 0.0, 'stu': 0.0, 'factors': ['No data'], 'source': 'Proxy'}

        wc    = (wdf['Close'] / cdf['Close'].reindex(wdf.index, method='ffill')).dropna()
        z     = float((wc.iloc[-1] - wc.mean()) / wc.std())
        score = 0.12 if z > 0.75 else -0.08 if z < -0.75 else 0.0
        sig   = 'BULLISH' if score > 0 else 'BEARISH' if score < 0 else 'NEUTRAL'
        return {'signal': sig, 'score': score, 'stu': 0.0,
                'factors': [f"W/C ratio z={z:+.2f}"], 'source': 'Market proxy'}
    except Exception:
        return {'signal': 'NEUTRAL', 'score': 0.0, 'stu': 0.0, 'factors': [], 'source': 'Error'}


# ── WEATHER ───────────────────────────────────────────────────────────────────

def get_weather_signal():
    """Fetch weather for key wheat regions. Cache for 8 hours."""
    cache_file = Path("weather_cache.json")
    api_key    = os.getenv("VISUAL_CROSSING_API_KEY", "W2FNC8VKT94JKH9ZRZYHUE63P")

    # Use cache if fresh
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            age   = (datetime.now(IL) - datetime.fromisoformat(cache['ts'])).total_seconds()
            if age < 28800:  # 8 hours
                return cache['data']
        except Exception:
            pass

    regions = {
        'Kansas': '38.5,-98.0', 'Oklahoma': '35.5,-98.0',
        'N.Dakota': '47.5,-100.5', 'Ukraine': '46.5,32.0',
        'Russia': '45.0,39.0', 'Canada': '52.0,-106.0',
    }

    scores = []
    for name, coords in regions.items():
        try:
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{coords}"
            end = datetime.now(IL)
            r   = requests.get(url, params={
                'key': api_key, 'unitGroup': 'metric', 'include': 'days',
                'elements': 'datetime,temp,tempmax,tempmin,precip',
                'contentType': 'json',
                'startDateTime': (end - timedelta(days=7)).strftime('%Y-%m-%d'),
                'endDateTime': end.strftime('%Y-%m-%d'),
            }, timeout=12)
            if r.status_code == 200:
                days   = r.json().get('days', [])
                precip = sum(d.get('precip', 0) for d in days)
                tmax   = max(d.get('tempmax', 20) for d in days)
                tmin   = min(d.get('tempmin', 0)  for d in days)
                month  = datetime.now(IL).month
                s      = 0.0
                if precip < 5:   s += 0.12
                if month in [5,6,7] and tmax > 35: s += 0.15
                if month in [12,1,2] and tmin < -10: s += 0.18
                scores.append(s)
        except Exception:
            pass

    if not scores:
        result = {'signal': 'NEUTRAL', 'score': 0.0, 'explanation': 'No data'}
    else:
        avg    = np.mean(scores)
        signal = 'BULLISH' if avg > 0.10 else 'BEARISH' if avg < -0.05 else 'NEUTRAL'
        result = {'signal': signal, 'score': round(avg, 4),
                  'explanation': f"{len(scores)}/6 regions checked"}

    try:
        cache_file.write_text(json.dumps({'ts': datetime.now(IL).isoformat(), 'data': result}))
    except Exception:
        pass

    return result


# ── VOLUME SIGNAL ─────────────────────────────────────────────────────────────

# ── FRONT-MONTH CONTRACT (for accurate volume only) ──────────────────────────
# CBOT wheat delivery months: Mar(H), May(K), Jul(N), Sep(U), Dec(Z)
# ZW=F's continuous-contract volume field has a confirmed ~1-2 week
# backfill lag (see volume_lag_check.py, 2026-07-10/11 diagnostics).
# The specific front-month contract does NOT have this lag — confirmed
# by direct comparison (ZWU26.CBT showed real volume 70k-113k on dates
# where ZW=F showed single/double digits). BUT a specific contract only
# has a few months of tradeable history, so it's used ONLY for the
# live volume display/diagnostic — price, seasonal, and backtest
# history all continue using ZW=F's long continuous series.
# UPDATED 2026-08-25: front-month contract resolution now lives in a
# single shared module, trading_calendar.py, imported by both this
# file and bug_detector.py — see that module's docstring for the
# full history/reasoning. Previously this logic was duplicated
# between the two files (to keep bug_detector.py free of this file's
# heavy TensorFlow/XGBoost/sklearn dependencies); trading_calendar.py
# resolves that tension since it only needs numpy/yfinance, which
# bug_detector.py already imports anyway. If this logic ever needs to
# change again, change it once in trading_calendar.py — both files
# pick it up automatically.
from trading_calendar import (
    is_trading_day,
    WHEAT_MONTH_CODES,
    WHEAT_ROLL_BUFFER_DAYS,
    VOLUME_CROSSOVER_MULTIPLIER,
    get_front_month_ticker,
)



def get_accurate_volume():
    """
    Fetches real, non-lagged volume from the front-month specific
    contract, for use in the display/diagnostic line only. Falls
    back to (None, False) if the fetch fails — caller should fall
    back to the ZW=F figure with a clear label, not silently trust
    a missing value.
    """
    ticker = get_front_month_ticker()
    try:
        df = yf.Ticker(ticker).history(period='5d', interval='1d', auto_adjust=False)
        if not df.empty:
            vol_avg  = float(df['Volume'].rolling(min(20, len(df))).mean().iloc[-1])
            vol_curr = float(df['Volume'].iloc[-1])
            return {
                'ticker': ticker,
                'raw_volume': vol_curr,
                'raw_avg_volume': round(vol_avg, 0),
                'ratio': round(vol_curr / vol_avg, 2) if vol_avg > 0 else None,
            }, True
    except Exception as e:
        print(f"   Front-month volume fetch failed ({ticker}): {e}")
    return None, False


def get_volume_signal(df):
    """
    UPDATED 2026-07-10/11: the Vol: Xx display now uses accurate,
    non-lagged volume from the current front-month specific contract
    (see get_accurate_volume() above) instead of ZW=F's continuous
    series, which has a confirmed multi-day backfill lag. This is
    DISPLAY ONLY — vol_low remains permanently excluded from
    ConvictionGate/HighConvictionGate (see those files' docstrings);
    re-enabling it as a real trading signal would require backtesting
    volume across many historical rolled contracts, a separate task.
    """
    accurate, is_accurate = get_accurate_volume()

    if is_accurate and accurate['ratio'] is not None:
        vol_avg  = accurate['raw_avg_volume']
        vol_curr = accurate['raw_volume']
        ratio    = accurate['ratio']
        source   = f"LIVE ({accurate['ticker']})"
    else:
        # Fallback to the old (known-lagged) ZW=F figure, clearly labeled
        vol_avg   = float(df['Volume'].rolling(20).mean().iloc[-1])
        vol_curr  = float(df['Volume'].iloc[-1])
        ratio     = vol_curr / vol_avg if vol_avg > 0 else 1.0
        source    = "⚠️ FALLBACK (ZW=F, known lag)"

    ret = float(df['Close'].pct_change(1).iloc[-1])

    if ratio > 1.5 and ret > 0:
        signal = 'BULLISH'
    elif ratio > 1.5 and ret < 0:
        signal = 'BEARISH'
    elif ratio < 0.7:
        signal = 'QUIET'
    else:
        signal = 'NEUTRAL'

    return {'signal': signal, 'ratio': round(ratio, 2),
            'raw_volume': vol_curr, 'raw_avg_volume': round(vol_avg, 0),
            'source': source,
            'explanation': f"{ratio:.1f}x average volume ({source})"}


# ── STATE ─────────────────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {'alerts_sent': 0, 'alerts_today': {}, 'last_alert_date': None}


def save_state(state):
    state['last_check'] = datetime.now(IL).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── ALERT GATE ────────────────────────────────────────────────────────────────

def should_send(state):
    """Only send in the target Israel-time window (see hour check
    below for the current target). Manual always sends."""
    force  = os.getenv('FORCE_ALERT', '').lower() in ('true', '1', 'yes')
    event  = os.getenv('GITHUB_EVENT_NAME', '')
    manual = force or 'workflow_dispatch' in event

    if manual:
        # UPDATED 2026-07-31: distinguish a genuine manual test run from
        # a price-move-triggered re-run (see check_price_trigger.py) —
        # both set FORCE_ALERT, but only the latter sets
        # PRICE_MOVE_REASON, so the log/reason accurately reflects why
        # this run actually happened instead of always saying "Manual".
        price_move_reason = os.getenv('PRICE_MOVE_REASON')
        if price_move_reason:
            return True, price_move_reason, True
        return True, "Manual trigger", True

    israel  = datetime.now(IL)
    il_hour = israel.hour
    il_date = israel.date().isoformat()

    # ADDED 2026-08-23: real incident — the weekly backtest cron
    # ('30 22 * * 6', UTC Saturday 22:30 = IL Sunday ~01:30) shares the
    # SAME job as the daily monitor and unconditionally runs the full
    # wheat_monitor_pro.py afterward, sending a real "scheduled morning
    # alert" — even though Sunday is not a trading day (trading is
    # Mon-Fri, see the trading_calendar_2026.csv-confirmed rule fixed
    # 2026-08-21). should_send() previously only checked the HOUR, never
    # the day of week, so it happily approved a Sunday morning send.
    # This is a defense-in-depth fix independent of workflow cron
    # correctness: whatever job or cron triggers this script, it will
    # now refuse a "scheduled" send outside Mon-Fri (weekday() 0-4),
    # regardless of hour. Manual/price-triggered runs are unaffected —
    # they return earlier above and always send, same as before.
    if not is_trading_day(israel):
        return False, f"Not a trading day ({israel.strftime('%A')})", False

    # UPDATED 2026-08-25: retimed from ~01:00 IL to ~03:00 IL after the
    # user confirmed Plus500's actual trading day starts at 03:00 IL
    # (not aligned with the underlying CME Globex daily-bar rollover,
    # which is ~01:00 IL — see wheat_monitor_github.yml's monitor cron
    # comment for the full reasoning on why 03:00 was chosen over a
    # second alert near Session 2's 16:30 open). The cron now targets
    # 02:53 IL (7min early, same GitHub Actions delay-buffer convention
    # as before).
    #
    # UPDATED 2026-09-02: real incident — GitHub Actions scheduling
    # delay pushed the run to 5:05 IL (target 02:53), landing on
    # il_hour=5, outside the old (2,3,4) window. should_send() returned
    # False, the job still exited 0 (no exception), so the run showed
    # green in Actions with zero visible signal that the alert never
    # sent — confirmed live 2026-09-01, commit e834b53: full pipeline
    # ran and logged a Tier 2 prediction, but alerts_sent/last_alert_date
    # never updated because send_telegram() was never reached. Widened
    # through hour 15 (Session 1 closes 15:45 IL — past that the day's
    # data is stale, so no point sending) so a late-but-still-useful
    # run still sends instead of being silently dropped. The
    # alerts_today slot-key check right below this is the real
    # duplicate-prevention guard, not this hour window — widening this
    # does not risk a second send for the same day.
    if il_hour not in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        return False, f"Not scheduled hour ({il_hour}:00 Israel)", False

    slot_key = f"{il_date}_morning"
    if state.get('alerts_today', {}).get(slot_key):
        return False, "Morning alert already sent today", False

    return True, "Scheduled morning alert (~03:00 Israel)", False


# ── TELEGRAM ──────────────────────────────────────────────────────────────────

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("Telegram not configured")
        return False
    try:
        # Send as plain text — no markdown parsing, no 400 errors
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT, "text": message},
            timeout=10
        )
        success = r.status_code == 200
        print(f"   Telegram: {'✓ sent' if success else '✗ failed'} ({r.status_code})")
        if not success:
            print(f"   Error: {r.text[:200]}")
        return success
    except Exception as e:
        print(f"   Telegram error: {e}")
        return False


# ── PERFORMANCE LOG ───────────────────────────────────────────────────────────

def log_prediction(direction, price, confidence, tier, seasonal_phase,
                    stop_price=None, target_price=None):
    """
    UPDATED 2026-09-03, real fix: previously only logged entry_price and
    left it to score_predictions.py to invent its own synthetic stop/
    target (1.5%/2.5% from entry_price) — a definition that matched
    neither the actual weekly setup shown in the Telegram alert nor
    anything the user was actually trading off of. Now also stores the
    REAL stop/target from the live weekly setup (weekly['stop'],
    weekly['target']) at the moment of logging, so score_predictions.py
    can score against what was actually communicated instead of a
    disconnected synthetic trade. stop_price/target_price are optional
    (default None) so this stays backward compatible with any code path
    that doesn't have a weekly setup handy; None is later treated by
    score_predictions.py as "use the legacy synthetic definition" so old
    entries logged before this change keep scoring exactly as they
    always did — nothing retroactive, going-forward only.
    """
    log_file = Path("prediction_log.json")
    try:
        log = json.loads(log_file.read_text()) if log_file.exists() else []
    except Exception:
        log = []

    log.append({
        'timestamp':      datetime.now(IL).isoformat(),
        'direction':      direction,
        'entry_price':    price,
        'confidence':     confidence,
        'tier':           tier,
        'seasonal_phase': seasonal_phase,
        'stop_price':     stop_price,
        'target_price':   target_price,
        'validated':      False,
        'outcome':        None,
        'exit_reason':    None,
        'pnl_cents':      None,
    })

    log_file.write_text(json.dumps(log, indent=2))
    print(f"   Prediction logged: {direction} at {price:.2f}¢ (Tier {tier})")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print(f"WHEAT MONITOR v4.0")
    print(f"Time: {datetime.now(IL).strftime('%Y-%m-%d %H:%M:%S')} Israel")
    print(f"{'='*70}\n")

    state = load_state()

    send, reason, is_manual = should_send(state)
    print(f"Alert gate: {reason}")

    # ── Fetch 5 years of data ──
    print(f"\nFetching {TICKER} (5 years)...")
    end    = datetime.now(IL)
    start  = end - timedelta(days=5 * 365)
    df_raw = yf.Ticker(TICKER).history(start=start, end=end, auto_adjust=False)

    if df_raw.empty:
        print("ERROR: No data"); return

    # ── Fetch corn for inter-market features ──
    print(f"Fetching {CORN_TICKER} (corn correlation)...")
    try:
        corn_raw = yf.Ticker(CORN_TICKER).history(start=start, end=end, auto_adjust=False)
        if not corn_raw.empty:
            # Align corn to wheat index and add as column
            corn_aligned = corn_raw['Close'].reindex(df_raw.index, method='ffill')
            df_raw['Corn_Close'] = corn_aligned
            print(f"  Corn data: {len(corn_raw)} candles merged")
        else:
            print("  Corn data unavailable — inter-market features disabled")
    except Exception as e:
        print(f"  Corn fetch skipped: {e}")

    if df_raw.index[-1].date() == datetime.now(IL).date():
        df_raw = df_raw.iloc[:-1]

    last_candle_date  = df_raw.index[-1].date()
    today_date        = datetime.now(IL).date()
    days_since_candle = (today_date - last_candle_date).days  # kept for the log line only, NOT used to decide closure anymore — see fix below

    # ── FIX: get the live quote BEFORE deciding whether the market is
    # closed, not after. A successful live fetch is direct, unambiguous
    # proof trading is happening right now — it should always win over
    # any date-based guess about the daily bar.
    live_price, is_live_price = get_live_price()

    # UPDATED 2026-08-17: the old check used raw CALENDAR days since
    # the last daily candle (>=3 => "closed"). That counts weekends,
    # so every Monday ~01:00 IL run — where the last candle is Friday's
    # close — saw a 3-CALENDAR-day gap and incorrectly skipped the
    # alert as "market closed", even though the market is open normally
    # every Monday. This was a real, recurring bug (confirmed missed
    # on 2026-08-17), not a one-off. Fixed two ways, applied together:
    #   1. missed_business_days uses np.busday_count on the days AFTER
    #      the last candle, which automatically excludes weekends —
    #      Fri->Mon correctly reads 0 missed business days, while a
    #      real holiday cluster (e.g. Thu holiday + weekend) still
    #      correctly reads >=1.
    #   2. Even if missed_business_days looks large, a successful live
    #      quote (is_live_price=True) overrides it — if get_live_price()
    #      is actually returning fresh data, the market is plainly not
    #      closed, whatever the daily-bar gap suggests.
    missed_business_days = np.busday_count(last_candle_date + timedelta(days=1), today_date)
    market_likely_closed = (missed_business_days >= 2) and not is_live_price

    if market_likely_closed and not is_manual:
        print(f"\nMarket closed — last candle {last_candle_date} "
              f"({missed_business_days} business day(s) missed, live fetch also failed). No alert.")
        save_state(state)
        return

    if is_live_price:
        current_price = live_price
        print(f"Price: {current_price:.2f}c  (LIVE — daily bar was {last_candle_date})")
    else:
        current_price = float(df_raw['Close'].iloc[-1])
        print(f"Price: {current_price:.2f}c  ⚠️ (STALE — daily bar {last_candle_date}, live fetch failed)")

    df = add_indicators(df_raw)

    # ── Engines ──
    print("\nRunning engines...")
    seasonal = SeasonalEngine()
    seasonal.fit(df)
    s_phase = seasonal.get_current_phase()
    print(f"  Seasonal: {s_phase['phase']} ({s_phase['confidence']:.0%}) — {s_phase['explanation']}")

    trend_engine = TrendEngine()
    trend_data   = trend_engine.get_trend(df)
    print(f"  Trend:    {trend_data['trend']} ({trend_data['strength']})")

    gate = ConvictionGate()
    tier, accuracy, gate_reason, gate_conds = gate.evaluate(df)
    print(f"  Gate:     {gate_reason}")

    # ── Signals ──
    print("\nFetching signals...")
    wasde   = get_wasde_signal()
    weather = get_weather_signal()
    volume  = get_volume_signal(df)
    print(f"  WASDE: {wasde['signal']} | Weather: {weather['signal']} | Vol: {volume['ratio']:.1f}x")

    # ── Ensemble ──
    print("\nTraining models...")
    ensemble = EnsemblePredictor()
    ensemble.train(df)
    pred      = ensemble.predict(df)
    direction = pred['direction']
    print(f"  Ensemble: {direction} | LSTM={pred['lstm']:.3f} RF={pred['rf']:.3f} XGB={pred['xgb']:.3f}")

    # ── Filters ──
    # UPDATED 2026-09-04: real bug found and confirmed — trend_blocked
    # was computed from the SAME pre-override `direction` as
    # seasonal_blocked, before either override ran. So if the seasonal
    # filter flipped direction (e.g. UP -> DOWN), trend_blocked had
    # already been evaluated against the OLD 'UP' value and couldn't
    # re-check the NEW 'DOWN' value against the trend — meaning
    # TrendEngine's whole purpose ("block signals that fight a strong
    # confirmed trend") silently failed to catch exactly the case it
    # exists for: a seasonal override flipping into a strong opposing
    # trend. Confirmed as the root cause of a real 9-day losing streak
    # (2026-08-13 to 2026-08-21, 9.1% win rate, DOWN calls forced by a
    # BEARISH seasonal override straight into a real STRONG uptrend).
    # Fix: apply the seasonal override FIRST, then evaluate
    # trend_blocked against the direction AS IT STANDS AFTER that
    # override — so the trend filter can actually catch a bad flip,
    # not just the original ensemble call.
    seasonal_blocked, _ = seasonal.blocks_direction(direction)

    if seasonal_blocked:
        direction          = 'DOWN' if direction == 'UP' else 'UP'
        pred['confidence'] = 0.60
        print(f"  Seasonal override → {direction}")

    trend_blocked, _ = trend_engine.blocks_direction(direction, trend_data)

    if trend_blocked:
        direction          = 'DOWN' if direction == 'UP' else 'UP'
        pred['confidence'] = 0.58
        print(f"  Trend filter → {direction}")

    # UPDATED 2026-08-23: real incident found — gate.evaluate() computes
    # tier/accuracy BEFORE the seasonal/trend override above can flip
    # direction, and every currently-validated condition (momentum_up
    # etc.) has ONLY ever been backtested for predicting UP moves (see
    # ConvictionGate's own docstring). When an override flips direction
    # to DOWN, the tier/accuracy badge stayed attached anyway, showing
    # e.g. "TIER 1 - 77.3% holdout-validated accuracy" on a DOWN call
    # that number has zero evidence for. Confirmed live: every Tier 1
    # entry in prediction_log.json since the 2026-08-04 stats cutoff was
    # direction=DOWN via seasonal override, with a 9.1% live win rate
    # against the 77.3% badge shown — not model failure, a mislabeled
    # confidence score. Downgrading to Tier 0 here means it also stops
    # being logged as a tracked prediction (existing "if tier > 0" gate
    # further down), so it no longer pollutes accuracy stats either.
    # Does NOT touch the weekly trade setup (entry/stop/target) — that
    # has its own separate freeze/break logic, unaffected either way.
    # UPDATED 2026-09-05, real gap found in the 2026-08-23 fix above:
    # that fix only downgraded Tier when seasonal_blocked or
    # trend_blocked caused the flip to DOWN — it didn't cover the case
    # where the ensemble's OWN raw vote is natively DOWN with no
    # override involved at all. momentum_up/macd_bullish are
    # independent technical checks (momentum, MACD crossover) that can
    # easily still read "active" on a day the ensemble concludes DOWN
    # on its own — in that case Tier could still show e.g. "Tier 2 —
    # momentum_up + macd_bullish" attached to a DOWN call, using
    # UP-only-validated accuracy, and the override-flag check above
    # wouldn't catch it since neither flag would be true. Checking
    # direction directly instead of the two override flags covers
    # both cases uniformly — simpler and more robust than the original.
    if direction == 'DOWN' and tier > 0:
        print(f"  Tier downgraded: {tier} -> 0 — validated condition only proven for UP, "
              f"but direction is {direction}")
        gate_reason = f"⚪ NO SIGNAL — validated condition is UP-only; direction is {direction}"
        tier = 0
        accuracy = gate.BASELINE_UP

    # ── Cost floor ──
    print("\nCalculating cost floor...")
    cost_signal = None
    try:
        from cost_floor_analyzer import CostFloorAnalyzer
        cost_signal = CostFloorAnalyzer().get_floor_signal(current_price)
    except Exception as e:
        print(f"  Cost floor skipped: {e}")

    cost_floor_cents = cost_signal['floor_cents'] if cost_signal else None

    # ── Weekly range prediction ──
    print("\nBuilding weekly range prediction...")
    news_signal = get_news_signal()
    if news_signal:
        print(f"   News signal (unvalidated, small nudge): {news_signal[0]} ({news_signal[1]}%)")
    break_outcome = None  # safe default — set for real inside the try block below
    try:
        from weekly_range_engine import WeeklyRangeEngine
        wre = WeeklyRangeEngine()
        wre.fit(df, exclude_years=[2022])
        weekly, out_of_range, break_outcome = get_frozen_weekly_plan(
            wre, df, current_price, cost_floor_cents, direction,
            backtest_tier=tier, backtest_accuracy=accuracy,
            news_signal=news_signal,
        )
        monthly = get_frozen_monthly_range(wre, df, current_price, cost_floor_cents)

        print(f"  Weekly range: {weekly['range_low']:.0f} - {weekly['range_high']:.0f}c")
        print(f"  Weekly FINAL CALL: {weekly['final_call']} (frozen — only changes on break)")
        print(f"  Daily direction (today): {direction}")
        print(f"  Monthly bias: {monthly['bias'] if monthly else 'N/A'}")

        # NOTE (2026-07-14): the old "weekly bias overrides daily ensemble
        # direction" block was removed here on purpose. weekly['final_call']
        # is now an intentionally FROZEN value (only changes when the
        # setup actually breaks) — letting it silently overwrite the
        # live daily `direction` every run would defeat that freeze and
        # recreate the exact "which number is real" confusion fixed
        # earlier. The two are now shown as separate, clearly labeled
        # lines (WEEKLY FINAL CALL vs daily direction) instead.

        status_word = "out range" if out_of_range else "in range"
        status_line = f"current price {current_price:.0f}c \"{status_word}\" direction {direction}"
        if break_outcome:
            status_line += f" — weekly setup just closed as {break_outcome}, new setup generated"

        # Build weekly message
        message = wre.format_alert(
            weekly      = weekly,
            monthly     = monthly,
            tier        = tier,
            gate_conds  = gate_conds,
            wasde       = wasde,
            weather     = weather,
            seasonal    = s_phase,
            cost_signal = cost_signal,
            gate_accuracy = accuracy,
            gate_reason   = gate_reason,
            final_direction = weekly['final_call'],
            daily_direction = direction,
            status_line     = status_line,
            current_price   = current_price,
        )

        # Add ensemble footnote
        message += (
            f"\nMODELS (supporting data):\n"
            f"LSTM: {pred['lstm']:.3f} | RF: {pred['rf']:.3f} | XGB: {pred['xgb']:.3f}\n"
            f"Agreement: {pred['agreement']} | Trend: {trend_data['trend']}\n"
        )

        use_weekly = True

    except Exception as e:
        print(f"  Weekly engine error: {e}")
        import traceback; traceback.print_exc()
        use_weekly = False

    # ── Fallback to daily message if weekly fails ──
    if not use_weekly:
        stop    = current_price * (1 - STOP_PCT) if direction == 'UP' else current_price * (1 + STOP_PCT)
        target  = current_price * (1 + TARGET_PCT) if direction == 'UP' else current_price * (1 - TARGET_PCT)
        message = (
            f"WHEAT MONITOR v4.0\n"
            f"------------------------------\n"
            f"{direction} ({pred['confidence']:.1%})\n"
            f"Price: {current_price:.2f}c\n\n"
            f"SEASONAL: {s_phase['phase']} ({s_phase['confidence']:.0%})\n"
            f"WASDE: {wasde['signal']} | Weather: {weather['signal']}\n"
            f"MODELS: LSTM={pred['lstm']:.3f} RF={pred['rf']:.3f} XGB={pred['xgb']:.3f}\n\n"
            f"Entry: {current_price:.2f}c | Stop: {stop:.2f}c | Target: {target:.2f}c\n"
        )

    # UPDATED 2026-07-31: make the trigger reason visible in the actual
    # alert, not just the GitHub Actions console log. Previously
    # PRICE_MOVE_REASON only affected should_send()'s internal reason
    # string, invisible to anyone just reading Telegram — this adds a
    # short header line so a price-triggered alert is distinguishable
    # from the routine scheduled 1am one at a glance.
    price_move_reason = os.getenv('PRICE_MOVE_REASON')
    if price_move_reason:
        message = f"⚡ {price_move_reason}\n\n" + message

    print(f"\nFINAL: {direction} | Tier {tier}")

    # ── Send ──
    # UPDATED 2026-08-07: full formatted alert now sends every day the
    # alert gate is open, regardless of tier — restoring the original
    # daily visibility that was lost when the 2026-07-29 tier-gating
    # fix (correctly) stopped Tier 0 from being logged as a tracked
    # prediction. Sending and logging are now fully decoupled: every
    # send still updates alerts_sent/alerts_today, but log_prediction()
    # (which feeds the accuracy stats bug_detector.py checks) still
    # only fires on tier > 0, so Tier 0 days stay excluded from
    # win-rate tracking without going silent/heartbeat-only.
    #
    # UPDATED 2026-08-22: real incident found in prediction_log.json —
    # 4 LOSS entries logged within ~35 minutes on 2026-08-19, 3 of them
    # from the user manually re-running the script during a volatile
    # morning to check in. Each run got its own logged prediction row,
    # scored independently by score_predictions.py against nearly the
    # same forward price path — one real event counted several times,
    # inflating both the sample size and the win/loss ratio behind
    # tier accuracy numbers. Fixed by skipping log_prediction() for
    # TRUE human button-presses only. Deliberately does NOT skip
    # price-move-triggered automatic re-runs (check_price_trigger.py,
    # via FORCE_ALERT + PRICE_MOVE_REASON) — those represent a real,
    # independent event (a genuine >=2% move) and should keep counting
    # as their own data point, same as a scheduled run would.
    is_human_manual = (
        os.getenv('GITHUB_EVENT_NAME', '') == 'workflow_dispatch'
        and not os.getenv('PRICE_MOVE_REASON')
    )

    if send:
        success = send_telegram(message)
        if success:
            state['alerts_sent'] = state.get('alerts_sent', 0) + 1
            state['last_alert_date'] = datetime.now(IL).date().isoformat()
            if not is_manual:
                # UPDATED 2026-08-23: was `= True`, now stores the actual
                # HH:MM send time. This slot only ever gets set for a
                # genuine SCHEDULED send (never a price-triggered
                # mid-day re-run, which legitimately varies in hour and
                # would make a blanket hour-check meaningless) — so it's
                # a clean, unambiguous signal for bug_detector.py to
                # verify the daily alert actually landed close to its
                # intended ~02:53 IL target (retimed 2026-08-25, see
                # should_send() and wheat_monitor_github.yml's monitor
                # cron comment), not just on the right day.
                slot_key = f"{datetime.now(IL).date().isoformat()}_morning"
                state.setdefault('alerts_today', {})[slot_key] = datetime.now(IL).strftime('%H:%M')
        if is_human_manual:
            print("   Manual (human-triggered) run — alert sent, NOT logged as a tracked prediction "
                  "(avoids inflating win/loss stats with clustered manual re-checks).")
        elif tier > 0:
            log_prediction(direction, current_price, pred['confidence'], tier, s_phase['phase'],
                           stop_price=weekly.get('stop'), target_price=weekly.get('target'))
        else:
            print("   Tier 0 — alert sent for visibility, NOT logged as a tracked prediction.")
    else:
        print(f"No alert: {reason}")

    state['last_direction'] = direction
    state['last_price']     = current_price
    save_state(state)

    print(f"\nTotal alerts sent: {state.get('alerts_sent', 0)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
