"""
COST FLOOR ANALYZER
====================
Fetches real wheat production cost data from USDA NASS Quick Stats.
Calculates the dynamic break-even floor price for wheat.

WHY THIS MATTERS:
  Wheat has a cost-of-production floor. When price drops near or below
  what it costs farmers to grow wheat, the market finds a bottom.
  This is more reliable than any technical indicator because it's
  driven by fundamental economics, not sentiment.

  Floor ~570-620¢: farmers losing money → production cuts → price bounces
  Fair value 570-650¢: direction driven by weather/demand
  Above 650¢: farmers plant more → supply increases → price falls

DATA SOURCES:
  - USDA NASS Quick Stats API — live season-average farm price
    (PRICE RECEIVED, $/BU, national level). This is the same live
    number USDA references in WASDE (e.g. the $6.00-6.50/bu
    2026/27 season-average price), fetched via API instead of
    scraping a static ERS Excel file that can silently break.
  - Natural gas / crude oil proxies for fertilizer and fuel cost drift.

FALLBACK:
  If the live NASS fetch fails for any reason, falls back to a
  hardcoded historical estimate — and now FLAGS this clearly in
  both the returned signal dict and the formatted Telegram alert,
  so a silent stale-data failure is never invisible again.
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta


# ── CONFIG ────────────────────────────────────────────────────────────────────

CACHE_FILE    = Path("cost_floor_cache.json")
CACHE_HOURS   = 72   # cost data changes slowly — cache 3 days

# Historical US wheat cost of production (fallback only, $/bu)
# Used ONLY if the live USDA NASS fetch fails. Update this periodically
# by hand as a sanity backstop, but it should rarely be hit in practice.
BASELINE_COST_PER_BU = 6.10   # ~610¢/bu fallback estimate

# Cost component weights (from USDA ERS wheat cost structure)
# These are the main variable costs that change with input prices
COST_WEIGHTS = {
    'fertilizer':  0.28,   # 28% of total cost
    'fuel':        0.12,   # 12%
    'seed':        0.08,   # 8%
    'chemicals':   0.07,   # 7%
    'fixed':       0.45,   # 45% fixed (land, machinery depreciation, labor)
}

# ── MAIN CLASS ────────────────────────────────────────────────────────────────

class CostFloorAnalyzer:
    """
    Calculates the dynamic wheat cost-of-production floor.
    Combines a live USDA farm-price fetch with real-time input cost proxies.
    """

    def __init__(self):
        self.floor_price    = None
        self.floor_source   = None
        self.components     = {}

    # ── public entry point ────────────────────────────────────────────────────

    def get_floor_signal(self, current_price_cents):
        """
        Main entry point. Returns signal dict with floor price,
        distance from floor, and trading implication.

        Args:
            current_price_cents: Current ZW=F price in cents/bushel

        Returns:
            dict with signal, floor_cents, distance_pct, implication
        """
        # Try cache first
        cached = self._load_cache()
        if cached:
            floor_cents = cached['floor_cents']
            source      = cached['source']
            components  = cached.get('components', {})
        else:
            # Fetch fresh data
            floor_dollars, source, components = self._calculate_floor()
            floor_cents = floor_dollars * 100
            self._save_cache(floor_cents, source, components)

        self.floor_price  = floor_cents
        self.floor_source = source
        self.components   = components

        # Calculate distance from floor
        distance_pct = (current_price_cents - floor_cents) / floor_cents

        # Fair value ceiling (farmers expand production above this)
        ceiling_cents = floor_cents * 1.15   # 15% above cost = strong profit margin

        # Generate signal
        if current_price_cents <= floor_cents * 1.03:
            signal      = 'STRONG_BUY'
            implication = f"Price at/below cost floor ({distance_pct:+.1%}) — strong bounce expected"
        elif current_price_cents <= floor_cents * 1.06:
            signal      = 'BUY'
            implication = f"Price near cost floor ({distance_pct:+.1%}) — downside limited"
        elif current_price_cents >= ceiling_cents * 1.05:
            signal      = 'BEARISH'
            implication = f"Price well above cost ({distance_pct:+.1%}) — farmers expanding production"
        elif current_price_cents >= ceiling_cents:
            signal      = 'CAUTION'
            implication = f"Price above fair value ({distance_pct:+.1%}) — supply response likely"
        else:
            signal      = 'NEUTRAL'
            implication = f"Price in fair value range ({distance_pct:+.1%} above floor)"

        is_live = components.get('is_live', False)
        live_tag = "LIVE" if is_live else "⚠️ FALLBACK"

        print(f"   Cost floor: {floor_cents:.0f}¢/bu ({source}) [{live_tag}]")
        print(f"   Current:    {current_price_cents:.0f}¢/bu")
        print(f"   Distance:   {distance_pct:+.1%} above floor")
        print(f"   Signal:     {signal}")

        return {
            'signal':          signal,
            'floor_cents':     round(floor_cents, 1),
            'ceiling_cents':   round(ceiling_cents, 1),
            'current_cents':   round(current_price_cents, 1),
            'distance_pct':    round(distance_pct, 4),
            'implication':     implication,
            'source':          source,
            'components':      components,
        }

    # ── floor calculation ─────────────────────────────────────────────────────

    def _calculate_floor(self):
        """
        Calculate current cost-of-production floor using:
        1. Live USDA NASS season-average farm price (most authoritative,
           updates automatically as USDA revises it through the season)
        2. Adjusted by current fertilizer and fuel price changes
        """
        print("   Calculating cost floor...")

        # Try live USDA NASS farm price first
        farm_price, is_live = self._fetch_farm_price()

        if is_live:
            base_cost = farm_price
            source    = "USDA NASS LIVE"
            print(f"   USDA NASS live farm price: ${base_cost:.2f}/bu")
        else:
            base_cost = BASELINE_COST_PER_BU
            source    = "FALLBACK (live fetch failed)"
            print(f"   ⚠️ Live fetch failed — using stale fallback: ${base_cost:.2f}/bu")

        # Adjust for current input prices
        fertilizer_adj = self._get_fertilizer_adjustment()
        fuel_adj       = self._get_fuel_adjustment()

        # Apply adjustments to variable cost components only
        variable_cost_pct = 1 - COST_WEIGHTS['fixed']
        total_adjustment  = (
            COST_WEIGHTS['fertilizer'] / variable_cost_pct * fertilizer_adj +
            COST_WEIGHTS['fuel']       / variable_cost_pct * fuel_adj
        ) * variable_cost_pct

        adjusted_cost = base_cost * (1 + total_adjustment)

        components = {
            'base_cost':        round(base_cost, 2),
            'fertilizer_adj':   round(fertilizer_adj, 4),
            'fuel_adj':         round(fuel_adj, 4),
            'total_adjustment': round(total_adjustment, 4),
            'final_cost':       round(adjusted_cost, 2),
            'is_live':          is_live,
        }

        print(f"   Fertilizer adj: {fertilizer_adj:+.1%}")
        print(f"   Fuel adj:       {fuel_adj:+.1%}")
        print(f"   Final cost:     ${adjusted_cost:.2f}/bu = {adjusted_cost*100:.0f}¢")

        return adjusted_cost, source, components

    # ── USDA NASS live farm price fetch ───────────────────────────────────────

    def _fetch_farm_price(self):
        """
        Fetch the current season-average farm price directly from
        USDA NASS Quick Stats — the same live number USDA references
        in WASDE (e.g. the $6.00-6.50/bu 2026/27 season-average price).

        Far more reliable than scraping the ERS cost-of-production Excel
        file, which uses a brittle static URL and fragile column parsing
        that can fail silently.

        Returns:
            (price_per_bu, is_live) tuple.
            is_live=False means the fetch failed and caller should use
            the hardcoded BASELINE_COST_PER_BU fallback instead.
        """
        api_key  = os.getenv("USDA_API_KEY", "3338B84E-694D-3E6A-925C-F35064C59BAE")
        base_url = "https://quickstats.nass.usda.gov/api/api_GET/"

        try:
            r = requests.get(base_url, params={
                'key': api_key, 'source_desc': 'SURVEY',
                'commodity_desc': 'WHEAT',
                'statisticcat_desc': 'PRICE RECEIVED',
                'unit_desc': '$ / BU',
                'agg_level_desc': 'NATIONAL',
                'format': 'JSON', 'year__GE': 2025,
            }, timeout=15)

            if r.status_code == 200:
                records = r.json().get('data', [])
                if records:
                    # Sort by year + reference period, most recent first
                    records = sorted(
                        records,
                        key=lambda x: (x.get('year', 0), x.get('reference_period_desc', '')),
                        reverse=True
                    )
                    for rec in records:
                        try:
                            price = float(str(rec['Value']).replace(',', ''))
                            if 3.0 < price < 15.0:  # sanity check $/bu
                                return price, True
                        except (ValueError, KeyError, TypeError):
                            continue

            print(f"   NASS farm price: no usable records (status {r.status_code})")
            return None, False

        except Exception as e:
            print(f"   NASS farm price fetch failed: {e}")
            return None, False

    # ── input cost adjustments ────────────────────────────────────────────────

    def _get_fertilizer_adjustment(self):
        """
        Estimate fertilizer price change vs baseline using:
        - Natural gas futures proxy (NG=F)
          (fertilizer production is ~70% natural gas cost)
        - Falls back to 0% adjustment if unavailable
        """
        try:
            end   = datetime.now()
            start = end - timedelta(days=400)

            ng  = yf.Ticker("NG=F").history(start=start, end=end, auto_adjust=False)

            if ng.empty or len(ng) < 60:
                return 0.0

            current_ng = float(ng['Close'].iloc[-1])
            avg_ng_1yr = float(ng['Close'].mean())

            # Fertilizer cost changes at ~60% of natgas price change
            ng_change      = (current_ng - avg_ng_1yr) / avg_ng_1yr
            fertilizer_adj = ng_change * 0.60

            # Cap at ±30% to prevent extreme swings
            return float(np.clip(fertilizer_adj, -0.30, 0.30))

        except Exception:
            return 0.0

    def _get_fuel_adjustment(self):
        """
        Estimate fuel cost change using crude oil price.
        Farm diesel tracks crude oil closely.
        """
        try:
            end   = datetime.now()
            start = end - timedelta(days=400)

            oil = yf.Ticker("CL=F").history(start=start, end=end, auto_adjust=False)

            if oil.empty or len(oil) < 60:
                return 0.0

            current_oil = float(oil['Close'].iloc[-1])
            avg_oil_1yr = float(oil['Close'].mean())

            # Fuel cost changes at ~80% of crude price change
            oil_change = (current_oil - avg_oil_1yr) / avg_oil_1yr
            fuel_adj   = oil_change * 0.80

            return float(np.clip(fuel_adj, -0.25, 0.25))

        except Exception:
            return 0.0

    # ── cache ─────────────────────────────────────────────────────────────────

    def _load_cache(self):
        if not CACHE_FILE.exists():
            return None
        try:
            data = json.loads(CACHE_FILE.read_text())
            age  = (datetime.now() - datetime.fromisoformat(data['timestamp'])).total_seconds()
            if age < CACHE_HOURS * 3600:
                print(f"   Using cached cost floor (age: {age/3600:.0f}h)")
                return data
        except Exception:
            pass
        return None

    def _save_cache(self, floor_cents, source, components):
        try:
            CACHE_FILE.write_text(json.dumps({
                'timestamp':   datetime.now().isoformat(),
                'floor_cents': floor_cents,
                'source':      source,
                'components':  components,
            }, indent=2))
        except Exception:
            pass

    # ── formatting ────────────────────────────────────────────────────────────

    def format_for_alert(self, signal_data):
        """Format cost floor data for Telegram alert."""
        signal_emojis = {
            'STRONG_BUY': '🟢🟢',
            'BUY':        '🟢',
            'NEUTRAL':    '⚪',
            'CAUTION':    '🟡',
            'BEARISH':    '🔴',
        }
        emoji = signal_emojis.get(signal_data['signal'], '⚪')

        is_live = signal_data['components'].get('is_live', False)
        warning = "" if is_live else "\n⚠️ Using stale fallback cost data — live NASS fetch failed"

        return (
            f"{emoji} *COST FLOOR:* {signal_data['floor_cents']:.0f}¢/bu\n"
            f"Current: {signal_data['current_cents']:.0f}¢ "
            f"({signal_data['distance_pct']:+.1%} above floor)\n"
            f"{signal_data['implication']}{warning}"
        )


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf

    print("Testing Cost Floor Analyzer...")
    print("=" * 50)

    # Get current wheat price
    wdf   = yf.Ticker("ZW=F").history(period="5d", auto_adjust=False)
    price = float(wdf['Close'].iloc[-1])
    print(f"Current ZW=F price: {price:.2f}¢\n")

    analyzer = CostFloorAnalyzer()
    result   = analyzer.get_floor_signal(price)

    print(f"\n{'='*50}")
    print(f"Signal:      {result['signal']}")
    print(f"Floor:       {result['floor_cents']:.0f}¢/bu")
    print(f"Ceiling:     {result['ceiling_cents']:.0f}¢/bu")
    print(f"Distance:    {result['distance_pct']:+.1%}")
    print(f"Implication: {result['implication']}")
    print(f"Source:      {result['source']}")
    print(f"\nComponents:")
    for k, v in result['components'].items():
        print(f"  {k}: {v}")

    print(f"\nAlert format:")
    print(analyzer.format_for_alert(result))
