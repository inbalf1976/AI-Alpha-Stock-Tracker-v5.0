"""
COST FLOOR ANALYZER
====================
Fetches real wheat production cost data from USDA ERS and AMS.
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
  - USDA ERS Commodity Costs and Returns (wheat cost per bushel)
  - USDA AMS Illinois Fertilizer Report (input cost trends)
  - USDA NASS Agricultural Prices (fuel/input index)

FALLBACK:
  If APIs unavailable, uses historical cost estimates updated by
  fertilizer and fuel price proxies from commodity markets.
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

# Historical US wheat cost of production (USDA ERS baseline, $/bu)
# Updated from ERS Commodity Costs and Returns report
BASELINE_COST_PER_BU = 6.10   # ~610¢/bu (2024-2025 USDA estimate)

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
    Combines USDA data with real-time input cost proxies.
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

        print(f"   Cost floor: {floor_cents:.0f}¢/bu ({source})")
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
        1. USDA ERS base cost (most authoritative)
        2. Adjusted by current fertilizer and fuel price changes
        """
        print("   Calculating cost floor...")

        # Try USDA ERS first
        ers_cost = self._fetch_ers_cost()

        if ers_cost:
            base_cost = ers_cost
            source    = "USDA ERS"
            print(f"   USDA ERS cost: ${base_cost:.2f}/bu")
        else:
            base_cost = BASELINE_COST_PER_BU
            source    = "Baseline estimate"
            print(f"   Using baseline: ${base_cost:.2f}/bu")

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
        }

        print(f"   Fertilizer adj: {fertilizer_adj:+.1%}")
        print(f"   Fuel adj:       {fuel_adj:+.1%}")
        print(f"   Final cost:     ${adjusted_cost:.2f}/bu = {adjusted_cost*100:.0f}¢")

        return adjusted_cost, source, components

    # ── USDA ERS fetch ────────────────────────────────────────────────────────

    def _fetch_ers_cost(self):
        """
        Fetch wheat cost of production from USDA ERS.
        URL: ers.usda.gov/data-products/commodity-costs-and-returns
        Returns cost in $/bushel or None if unavailable.
        """
        try:
            # USDA ERS provides CSV download
            url = "https://www.ers.usda.gov/webdocs/DataFiles/50048/WheatCostsReturn.xlsx"
            r   = requests.get(url, timeout=20)

            if r.status_code == 200 and len(r.content) > 1000:
                # Parse Excel file
                from io import BytesIO
                df = pd.read_excel(BytesIO(r.content), sheet_name=0, header=None)

                # Find most recent total cost per bushel
                # ERS format: rows are years, columns include "Total, gross value of production"
                for col in df.columns:
                    col_data = df[col].astype(str)
                    if col_data.str.contains('Total operating', case=False, na=False).any():
                        # Found cost column — get most recent non-null value
                        cost_col = df[col + 1] if col + 1 in df.columns else None
                        if cost_col is not None:
                            numeric = pd.to_numeric(cost_col, errors='coerce').dropna()
                            if len(numeric) > 0:
                                cost = float(numeric.iloc[-1])
                                if 3.0 < cost < 15.0:  # sanity check $/bu
                                    return cost
            return None

        except Exception as e:
            print(f"   ERS fetch: {e}")
            return None

    # ── input cost adjustments ────────────────────────────────────────────────

    def _get_fertilizer_adjustment(self):
        """
        Estimate fertilizer price change vs baseline using:
        - UAN (urea ammonium nitrate) futures proxy: UNG natural gas
          (fertilizer production is ~70% natural gas cost)
        - Falls back to 0% adjustment if unavailable
        """
        try:
            # Natural gas is the primary input for nitrogen fertilizer
            # Price change in natgas → similar % change in fertilizer
            end   = datetime.now()
            start = end - timedelta(days=400)

            ng  = yf.Ticker("NG=F").history(start=start, end=end, auto_adjust=False)

            if ng.empty or len(ng) < 60:
                return 0.0

            # Compare current price to 1-year average
            current_ng = float(ng['Close'].iloc[-1])
            avg_ng_1yr = float(ng['Close'].mean())

            # Fertilizer cost changes at ~60% of natgas price change
            # (other inputs buffer the full impact)
            ng_change       = (current_ng - avg_ng_1yr) / avg_ng_1yr
            fertilizer_adj  = ng_change * 0.60

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

        return (
            f"{emoji} *COST FLOOR:* {signal_data['floor_cents']:.0f}¢/bu\n"
            f"Current: {signal_data['current_cents']:.0f}¢ "
            f"({signal_data['distance_pct']:+.1%} above floor)\n"
            f"{signal_data['implication']}"
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
