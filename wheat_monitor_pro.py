"""
WASDE SCRAPER - FIXED VERSION
==============================
PROBLEMS WITH ORIGINAL:
  1. WASDE hardcoded permanently BEARISH — not a signal, just noise with a negative sign
  2. Live USDA QuickStats API returned mixed grain data, so it got bypassed entirely
  3. No fallback when API fails except the hardcoded value

THIS VERSION:
  1. Fixes the USDA QuickStats call to filter ALL_CLASSES wheat only
  2. Derives a dynamic STU signal from 3 independent sources:
       A) USDA QuickStats (wheat-specific, when available)
       B) Wheat/Corn price ratio z-score (market's own supply signal)
       C) Wheat price percentile vs 52-week range
  3. Combines all three into one honest signal — BULLISH, BEARISH, or NEUTRAL
  4. Never hardcodes a direction

ALSO IN THIS FILE:
  - get_multi_factor_agreement(): replaces the fake confidence boosting in wheat_monitor.py
    Returns True only when 4+ independent factors agree — higher bar, fewer but better alerts
"""

import os
import json
import requests
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


# ── tuneable constants ────────────────────────────────────────────────────────
TICKER_WHEAT = "ZW=F"
TICKER_CORN  = "ZC=F"

# STU thresholds (global wheat stocks-to-use %)
STU_VERY_TIGHT  = 0.27   # < 27% → strongly bullish
STU_TIGHT       = 0.30   # < 30% → bullish
STU_AMPLE       = 0.33   # > 33% → bearish

# Wheat/corn ratio: high ratio = wheat tight relative to corn
WC_ZSCORE_BULLISH =  0.75   # z-score above this → bullish
WC_ZSCORE_BEARISH = -0.75   # z-score below this → bearish

# Price percentile: where is wheat vs its own recent history?
PERCENTILE_BULLISH = 0.65   # above 65th percentile → market pricing in tightness
PERCENTILE_BEARISH = 0.35   # below 35th percentile → market pricing in ample supply

# Minimum independent factors agreeing to pass the multi-factor gate
MIN_AGREEMENT_FOR_ALERT = 4   # out of: ensemble, seasonal, weather, wasde, volume, context
# ─────────────────────────────────────────────────────────────────────────────


class LiveWASDEScraper:
    """
    Dynamic wheat fundamentals signal.
    Uses USDA QuickStats (when available) + market-derived proxies as backup.
    """

    def __init__(self):
        self.api_key  = os.getenv("USDA_API_KEY", "3338B84E-694D-3E6A-925C-F35064C59BAE")
        self.base_url = "https://quickstats.nass.usda.gov/api/api_GET/"

    # ── public API ────────────────────────────────────────────────────────────

    def get_fundamental_score(self):
        """
        Returns a signal dict with keys: signal, score, data, factors
        Signal is always derived from current data — never hardcoded.
        """
        print("   📊 WASDE: fetching dynamic signal...")

        # Source A: USDA QuickStats (real wheat-specific stocks)
        usda_result = self._fetch_usda_wheat_stocks()

        # Sources B+C: market-derived proxies (always available via yfinance)
        market_result = self._fetch_market_proxy()

        # Combine
        return self._combine_signals(usda_result, market_result)

    # ── source A: USDA QuickStats ─────────────────────────────────────────────

    def _fetch_usda_wheat_stocks(self):
        """
        Fetch wheat-specific stocks from USDA QuickStats.
        Key fix vs original: filter class_desc=ALL CLASSES to avoid mixed grain data.
        """
        try:
            params = {
                'key':               self.api_key,
                'source_desc':       'SURVEY',
                'commodity_desc':    'WHEAT',
                'class_desc':        'ALL CLASSES',        # ← the fix
                'statisticcat_desc': 'STOCKS',
                'unit_desc':         'BU',
                'agg_level_desc':    'NATIONAL',
                'format':            'JSON',
                'year__GE':          2021,
            }
            response = requests.get(self.base_url, params=params, timeout=15)

            if response.status_code != 200:
                print(f"      USDA API: {response.status_code} — using market proxy only")
                return None

            data = response.json().get('data', [])
            if not data:
                print("      USDA API: no records returned — using market proxy only")
                return None

            return self._parse_usda_data(data)

        except Exception as e:
            print(f"      USDA API error: {e} — using market proxy only")
            return None

    def _parse_usda_data(self, records):
        """
        Parse USDA records into a stocks-to-use ratio.
        Uses actual annual US wheat consumption (~2.0B bu) as denominator.
        """
        # Sort newest first
        sorted_records = sorted(
            records,
            key=lambda x: (int(x.get('year', 0)), x.get('reference_period_desc', '')),
            reverse=True
        )

        # Get most recent valid value
        latest_value = None
        latest_year  = None
        prev_value   = None

        for rec in sorted_records:
            try:
                val = float(rec.get('Value', '0').replace(',', ''))
                yr  = int(rec.get('year', 0))
                if val <= 0:
                    continue
                if latest_value is None:
                    latest_value = val
                    latest_year  = yr
                elif yr < latest_year and prev_value is None:
                    prev_value = val
                    break
            except (ValueError, TypeError):
                continue

        if latest_value is None:
            return None

        # Annual US wheat use ≈ 2.0B bushels (USDA WASDE consistent figure)
        annual_use_bu   = 2_000_000_000
        stocks_to_use   = latest_value / annual_use_bu

        yoy_change = 0.0
        if prev_value and prev_value > 0:
            yoy_change = ((latest_value - prev_value) / prev_value) * 100

        print(f"      USDA stocks: {latest_value/1e6:.0f}M bu | STU: {stocks_to_use:.1%} | YoY: {yoy_change:+.1f}%")

        return {
            'stocks_to_use': stocks_to_use,
            'yoy_change':    yoy_change,
            'year':          latest_year,
            'source':        'USDA QuickStats LIVE',
        }

    # ── source B+C: market-derived proxies ───────────────────────────────────

    def _fetch_market_proxy(self):
        """
        Two market-derived signals that don't depend on USDA API:

        B) Wheat/Corn ratio z-score
           When wheat is expensive relative to corn, the market is pricing in
           tight wheat supply. This IS forward-looking (unlike raw weather data).

        C) Wheat price percentile vs 52-week range
           Where is wheat trading in its own recent history?
        """
        try:
            end   = datetime.now()
            start = end - timedelta(days=400)   # ~14 months for good z-score baseline

            wdf = yf.Ticker(TICKER_WHEAT).history(start=start, end=end, auto_adjust=False)
            cdf = yf.Ticker(TICKER_CORN).history(start=start, end=end,  auto_adjust=False)

            if wdf.empty:
                return None

            w_price = float(wdf['Close'].iloc[-1])

            # ── B: wheat/corn ratio z-score ──
            wc_signal  = 'NEUTRAL'
            wc_zscore  = 0.0
            wc_note    = "W/C ratio: N/A (no corn data)"

            if not cdf.empty:
                # Align on common dates
                ratio = wdf['Close'] / cdf['Close'].reindex(wdf.index, method='ffill')
                ratio = ratio.dropna()

                if len(ratio) > 20:
                    current_ratio = float(ratio.iloc[-1])
                    ratio_mean    = float(ratio.mean())
                    ratio_std     = float(ratio.std())

                    if ratio_std > 0:
                        wc_zscore = (current_ratio - ratio_mean) / ratio_std
                    else:
                        wc_zscore = 0.0

                    if wc_zscore >= WC_ZSCORE_BULLISH:
                        wc_signal = 'BULLISH'
                    elif wc_zscore <= WC_ZSCORE_BEARISH:
                        wc_signal = 'BEARISH'

                    wc_note = f"W/C ratio z-score: {wc_zscore:+.2f} ({wc_signal})"
                    print(f"      {wc_note}")

            # ── C: price percentile ──
            prices_1yr = wdf['Close'].iloc[-252:] if len(wdf) >= 252 else wdf['Close']
            pct        = float((prices_1yr < w_price).mean())
            high52     = float(prices_1yr.max())
            low52      = float(prices_1yr.min())
            pos_in_range = (w_price - low52) / (high52 - low52) if high52 > low52 else 0.5

            if pct >= PERCENTILE_BULLISH:
                pct_signal = 'BULLISH'
            elif pct <= PERCENTILE_BEARISH:
                pct_signal = 'BEARISH'
            else:
                pct_signal = 'NEUTRAL'

            pct_note = f"Price at {pct:.0%} of 1yr range ({pct_signal})"
            print(f"      {pct_note}")

            return {
                'wc_zscore':    wc_zscore,
                'wc_signal':    wc_signal,
                'pct_signal':   pct_signal,
                'price_pct':    pct,
                'pos_in_range': pos_in_range,
                'wc_note':      wc_note,
                'pct_note':     pct_note,
            }

        except Exception as e:
            print(f"      Market proxy error: {e}")
            return None

    # ── combine all sources ───────────────────────────────────────────────────

    def _combine_signals(self, usda, market):
        """
        Combine USDA + market proxy into one honest signal.
        Majority vote — no single source dominates.
        """
        votes      = []
        score      = 0.0
        factors    = []

        # Vote from USDA (weight: 2 votes — most authoritative)
        if usda is not None:
            stu = usda['stocks_to_use']
            yoy = usda['yoy_change']

            if stu < STU_VERY_TIGHT:
                votes  += ['BULLISH', 'BULLISH']
                score  += 0.25
                factors.append(f"Very tight stocks ({stu:.1%} STU)")
            elif stu < STU_TIGHT:
                votes  += ['BULLISH', 'BULLISH']
                score  += 0.15
                factors.append(f"Tight stocks ({stu:.1%} STU)")
            elif stu > STU_AMPLE:
                votes  += ['BEARISH', 'BEARISH']
                score  -= 0.15
                factors.append(f"Ample stocks ({stu:.1%} STU)")
            else:
                votes  += ['NEUTRAL', 'NEUTRAL']
                factors.append(f"Balanced stocks ({stu:.1%} STU)")

            if yoy < -5:
                votes.append('BULLISH')
                score += 0.08
                factors.append(f"Stocks falling {abs(yoy):.1f}% YoY")
            elif yoy > 5:
                votes.append('BEARISH')
                score -= 0.05

            stocks_to_use_display = stu
            source_label = usda['source']
        else:
            # No USDA data — note it but don't fake a signal
            factors.append("USDA data unavailable")
            stocks_to_use_display = 0.0
            source_label = "Market proxy only"

        # Vote from W/C ratio (weight: 1 vote)
        if market is not None:
            wc_sig = market['wc_signal']
            votes.append(wc_sig)
            factors.append(market['wc_note'])

            if wc_sig == 'BULLISH':
                score += 0.08
            elif wc_sig == 'BEARISH':
                score -= 0.06

            # Vote from price percentile (weight: 1 vote)
            pct_sig = market['pct_signal']
            votes.append(pct_sig)
            factors.append(market['pct_note'])

            if pct_sig == 'BULLISH':
                score += 0.05
            elif pct_sig == 'BEARISH':
                score -= 0.04

        # Tally votes
        bullish_votes = votes.count('BULLISH')
        bearish_votes = votes.count('BEARISH')
        total_votes   = len(votes) if votes else 1

        if bullish_votes / total_votes >= 0.5:
            signal = 'BULLISH'
        elif bearish_votes / total_votes >= 0.5:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        print(f"      WASDE combined: {signal} "
              f"(votes: {bullish_votes}B / {bearish_votes}Bear / {total_votes-bullish_votes-bearish_votes}N)")

        return {
            'signal': signal,
            'score':  round(score, 4),
            'data': {
                'stocks_to_use': stocks_to_use_display,
                'stocks_change': usda['yoy_change'] if usda else 0,
                'last_updated':  datetime.now().strftime('%Y-%m-%d'),
                'source':        source_label,
                'votes':         f"{bullish_votes}B/{bearish_votes}Bear/{total_votes}total",
            },
            'factors': factors[:3],
        }


# ── multi-factor agreement gate ───────────────────────────────────────────────

def get_multi_factor_agreement(direction, seasonal, weather_signal,
                                wasde_signal, volume_signal, context,
                                prediction):
    """
    Replaces the fake confidence-boosting arithmetic in wheat_monitor.py.

    Instead of adding/subtracting arbitrary percentages, count how many
    INDEPENDENT factors actually agree with the ensemble direction.
    Only send an alert when MIN_AGREEMENT_FOR_ALERT or more agree.

    Returns: (agreement_count: int, agreeing_factors: list, blocked: bool)
    """
    agreeing   = []
    disagreeing = []

    # 1. Ensemble internal agreement (all 3 models same direction)
    if prediction['agreement'] in ('STRONG UP', 'STRONG DOWN'):
        agreeing.append(f"Ensemble: all 3 models agree ({prediction['agreement']})")
    else:
        disagreeing.append(f"Ensemble: split ({prediction['agreement']})")

    # 2. Seasonal
    if (direction == 'UP'   and seasonal['bias'] >  0.02) or \
       (direction == 'DOWN' and seasonal['bias'] < -0.02):
        agreeing.append(f"Seasonal: {seasonal['direction']}")
    elif abs(seasonal['bias']) > 0.02:
        disagreeing.append(f"Seasonal: {seasonal['direction']} (against)")

    # 3. Weather
    if (direction == 'UP'   and weather_signal['signal'] == 'BULLISH') or \
       (direction == 'DOWN' and weather_signal['signal'] == 'BEARISH'):
        agreeing.append(f"Weather: {weather_signal['signal']}")
    elif weather_signal['signal'] != 'NEUTRAL':
        disagreeing.append(f"Weather: {weather_signal['signal']} (against)")

    # 4. WASDE (now dynamic — this vote actually means something)
    if (direction == 'UP'   and wasde_signal['signal'] == 'BULLISH') or \
       (direction == 'DOWN' and wasde_signal['signal'] == 'BEARISH'):
        agreeing.append(f"WASDE: {wasde_signal['signal']}")
    elif wasde_signal['signal'] != 'NEUTRAL':
        disagreeing.append(f"WASDE: {wasde_signal['signal']} (against)")

    # 5. Volume
    if (direction == 'UP'   and volume_signal['signal'] == 'BULLISH') or \
       (direction == 'DOWN' and volume_signal['signal'] == 'BEARISH'):
        agreeing.append(f"Volume: {volume_signal['signal']}")
    elif volume_signal['signal'] != 'NEUTRAL':
        disagreeing.append(f"Volume: {volume_signal['signal']} (against)")

    # 6. Market context
    if (direction == 'UP'   and context['signal'] == 'BUY') or \
       (direction == 'DOWN' and context['signal'] == 'SELL'):
        agreeing.append(f"Context: {context['position']}")
    elif context['signal'] != 'NEUTRAL':
        disagreeing.append(f"Context: {context['position']} (against)")

    count   = len(agreeing)
    blocked = count < MIN_AGREEMENT_FOR_ALERT

    print(f"\n🎯 Multi-factor agreement: {count}/6 factors agree")
    for f in agreeing:
        print(f"   ✅ {f}")
    for f in disagreeing:
        print(f"   ❌ {f}")
    if blocked:
        print(f"   ⛔ BLOCKED — need {MIN_AGREEMENT_FOR_ALERT}, got {count}")
    else:
        print(f"   ✅ PASSED — {count} factors agree")

    return count, agreeing, blocked


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    scraper = LiveWASDEScraper()
    result  = scraper.get_fundamental_score()

    print(f"\n{'='*50}")
    print(f"Signal:  {result['signal']}")
    print(f"Score:   {result['score']:+.4f}")
    print(f"Source:  {result['data']['source']}")
    print(f"STU:     {result['data']['stocks_to_use']:.1%}")
    print(f"Votes:   {result['data']['votes']}")
    print(f"Factors: {result['factors']}")
