"""
LIVE WASDE Scraper - USDA QuickStats API
Real-time supply and demand data for wheat trading
"""

import requests
import os
from datetime import datetime

class LiveWASDEScraper:
    """Fetch and analyze live USDA WASDE data for wheat"""
    
    def __init__(self):
        self.api_key = os.getenv("USDA_API_KEY", "3338B84E-694D-3E6A-925C-F35064C59BAE")
        self.base_url = "https://quickstats.nass.usda.gov/api/api_GET/"
        
    def fetch_wheat_stocks(self):
        """
        Fetch latest wheat ending stocks data
        
        Returns:
            dict with stocks data
        """
        try:
            params = {
                'key': self.api_key,
                'source_desc': 'SURVEY',
                'commodity_desc': 'WHEAT',
                'statisticcat_desc': 'STOCKS',
                'agg_level_desc': 'NATIONAL',
                'format': 'JSON',
                'year__GE': 2020  # Last 5+ years
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    return self._parse_stocks_data(data['data'])
            
            print(f"WASDE API error: {response.status_code}")
            return None
            
        except Exception as e:
            print(f"WASDE fetch error: {e}")
            return None
    
    def fetch_wheat_production(self):
        """Fetch latest wheat production forecast"""
        try:
            params = {
                'key': self.api_key,
                'source_desc': 'SURVEY',
                'commodity_desc': 'WHEAT',
                'statisticcat_desc': 'PRODUCTION',
                'agg_level_desc': 'NATIONAL',
                'format': 'JSON',
                'year__GE': 2023
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    return self._parse_production_data(data['data'])
            
            return None
            
        except Exception as e:
            print(f"Production fetch error: {e}")
            return None
    
    def _parse_stocks_data(self, data):
        """Parse stocks data and calculate trends"""
        # Sort by year and period
        sorted_data = sorted(data, key=lambda x: (x.get('year', 0), x.get('reference_period_desc', '')), reverse=True)
        
        if not sorted_data:
            return None
        
        # Get most recent value
        latest = sorted_data[0]
        latest_value = float(latest.get('Value', 0).replace(',', ''))
        latest_year = latest.get('year')
        
        # Get previous year for comparison
        previous_value = None
        for record in sorted_data[1:]:
            if record.get('year') != latest_year:
                try:
                    previous_value = float(record.get('Value', 0).replace(',', ''))
                    break
                except:
                    continue
        
        # Calculate year-over-year change
        yoy_change = 0
        if previous_value and previous_value > 0:
            yoy_change = ((latest_value - previous_value) / previous_value) * 100
        
        return {
            'current_stocks': latest_value,
            'previous_stocks': previous_value,
            'yoy_change_pct': yoy_change,
            'year': latest_year,
            'period': latest.get('reference_period_desc', 'Unknown'),
            'unit': latest.get('unit_desc', 'BU')
        }
    
    def _parse_production_data(self, data):
        """Parse production data"""
        sorted_data = sorted(data, key=lambda x: x.get('year', 0), reverse=True)
        
        if not sorted_data:
            return None
        
        latest = sorted_data[0]
        latest_value = float(latest.get('Value', 0).replace(',', ''))
        
        previous_value = None
        if len(sorted_data) > 1:
            try:
                previous_value = float(sorted_data[1].get('Value', 0).replace(',', ''))
            except:
                pass
        
        yoy_change = 0
        if previous_value and previous_value > 0:
            yoy_change = ((latest_value - previous_value) / previous_value) * 100
        
        return {
            'current_production': latest_value,
            'previous_production': previous_value,
            'yoy_change_pct': yoy_change,
            'year': latest.get('year')
        }
    
    def get_fundamental_score(self):
        """
        Calculate fundamental score from live WASDE data
        
        Returns:
            dict with signal and score
        """
        print("      Fetching USDA data...", end=" ")
        
        stocks_data = self.fetch_wheat_stocks()
        production_data = self.fetch_wheat_production()
        
        if not stocks_data and not production_data:
            print("Failed")
            return self._get_default_estimates()
        
        print("Success")
        
        score = 0.0
        factors = []
        
        # Analyze stocks
        if stocks_data:
            stocks_value = stocks_data['current_stocks']
            yoy_change = stocks_data['yoy_change_pct']
            
            # Estimate stocks-to-use ratio (simplified)
            # Typical US wheat use is ~2,000 million bushels/year
            estimated_use = 2000  # Million bushels
            stocks_to_use = stocks_value / estimated_use if estimated_use > 0 else 0.20
            
            # Score based on stocks-to-use
            if stocks_to_use < 0.15:  # Very tight
                score += 0.30
                factors.append(f"Very tight stocks ({stocks_to_use:.1%})")
            elif stocks_to_use < 0.18:  # Tight
                score += 0.20
                factors.append(f"Tight stocks ({stocks_to_use:.1%})")
            elif stocks_to_use > 0.25:  # Ample
                score -= 0.15
                factors.append(f"Ample stocks ({stocks_to_use:.1%})")
            
            # Score based on trend
            if yoy_change < -5:  # Stocks declining
                score += 0.15
                factors.append(f"Stocks down {abs(yoy_change):.1f}% YoY")
            elif yoy_change > 5:  # Stocks rising
                score -= 0.10
                factors.append(f"Stocks up {yoy_change:.1f}% YoY")
        else:
            stocks_to_use = 0.18  # Default estimate
        
        # Analyze production
        if production_data:
            prod_change = production_data['yoy_change_pct']
            
            if prod_change < -3:  # Production down
                score += 0.12
                factors.append(f"Production down {abs(prod_change):.1f}%")
            elif prod_change > 3:  # Production up
                score -= 0.08
                factors.append(f"Production up {prod_change:.1f}%")
        
        # Determine signal
        if score > 0.20:
            signal = 'BULLISH'
            confidence = 0.80
        elif score > 0.10:
            signal = 'BULLISH'
            confidence = 0.70
        elif score < -0.10:
            signal = 'BEARISH'
            confidence = 0.65
        else:
            signal = 'NEUTRAL'
            confidence = 0.60
        
        return {
            'signal': signal,
            'score': score,
            'confidence': confidence,
            'data': {
                'stocks_to_use': stocks_to_use,
                'production_change': production_data['yoy_change_pct'] if production_data else 0,
                'stocks_change': stocks_data['yoy_change_pct'] if stocks_data else 0,
                'last_updated': datetime.now().strftime('%Y-%m-%d'),
                'source': 'USDA QuickStats LIVE'
            },
            'factors': factors[:2],  # Top 2 factors
            'explanation': self._generate_explanation(signal, factors)
        }
    
    def _generate_explanation(self, signal, factors):
        """Generate human-readable explanation"""
        if not factors:
            return "USDA fundamentals analysis"
        
        if signal == 'BULLISH':
            return f"Supply concerns: {factors[0]}"
        elif signal == 'BEARISH':
            return f"Ample supply: {factors[0]}"
        else:
            return f"Balanced: {factors[0]}"
    
    def _get_default_estimates(self):
        """Fallback when API fails"""
        month = datetime.now().month
        
        if month in [1, 2, 3]:
            stocks_ratio = 0.18
            signal = 'BULLISH'
            score = 0.20
        elif month in [7, 8, 9]:  # Post-harvest
            stocks_ratio = 0.22
            signal = 'BEARISH'
            score = -0.10
        else:
            stocks_ratio = 0.20
            signal = 'NEUTRAL'
            score = 0.10
        
        return {
            'signal': signal,
            'score': score,
            'confidence': 0.55,
            'data': {
                'stocks_to_use': stocks_ratio,
                'production_change': -1.0,
                'stocks_change': 0,
                'last_updated': datetime.now().strftime('%Y-%m-%d'),
                'source': 'ESTIMATED (API unavailable)'
            },
            'factors': ['Using seasonal estimates'],
            'explanation': 'USDA data unavailable - using estimates'
        }

# Quick test
if __name__ == "__main__":
    scraper = LiveWASDEScraper()
    result = scraper.get_fundamental_score()
    print(f"\nWASDE Score: {result}")
