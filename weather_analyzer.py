"""
LIVE Weather Analyzer - Visual Crossing API
Agricultural data for wheat trading: soil moisture, drought, temperature stress
"""

import requests
import os
from datetime import datetime, timedelta

class LiveWeatherAnalyzer:
    """Analyze weather impact on wheat using Visual Crossing agricultural data"""
    
    def __init__(self):
        # API configuration
        self.api_key = os.getenv("VISUAL_CROSSING_API_KEY", "W2FNC8VKT94JKH9ZRZYHUE63P")
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        
        # Key wheat-growing regions
        self.wheat_regions = {
            'Kansas': '38.5,-98.0',  # Center of Kansas wheat belt
            'Oklahoma': '35.5,-98.0',  # Oklahoma panhandle
            'North Dakota': '47.5,-100.5',  # Spring wheat region
            'Montana': '47.0,-110.0'  # Montana wheat
        }
    
    def fetch_weather_data(self, location, days=7):
        """
        Fetch weather data from Visual Crossing
        
        Args:
            location: lat,lon coordinates
            days: number of days to fetch
        
        Returns:
            dict with weather data
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Build API request
            url = f"{self.base_url}/{location}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            
            params = {
                'key': self.api_key,
                'unitGroup': 'metric',
                'include': 'days',
                'elements': 'datetime,temp,tempmax,tempmin,precip,precipcover,humidity,windspeed,conditions',
                'contentType': 'json'
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Weather API error: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"Weather fetch error: {e}")
            return None
    
    def analyze_agricultural_impact(self, weather_data):
        """
        Analyze weather data for wheat impact
        
        Returns:
            dict with impact analysis
        """
        if not weather_data or 'days' not in weather_data:
            return self._get_neutral_signal()
        
        days = weather_data['days']
        recent_days = days[-7:]  # Last week
        
        # Initialize scores
        drought_score = 0
        temperature_score = 0
        precipitation_score = 0
        
        # Analyze recent weather
        total_precip = sum(day.get('precip', 0) for day in recent_days)
        avg_temp = sum(day.get('temp', 0) for day in recent_days) / len(recent_days)
        max_temp = max(day.get('tempmax', 0) for day in recent_days)
        min_temp = min(day.get('tempmin', 0) for day in recent_days)
        
        # PRECIPITATION ANALYSIS
        if total_precip < 5:  # Less than 5mm in week = dry
            drought_score = 0.15  # Bullish (supply concern)
            precip_impact = "BULLISH"
            precip_note = f"Dry conditions ({total_precip:.1f}mm/week) - yield concern"
        elif total_precip > 50:  # More than 50mm = very wet
            drought_score = -0.10  # Bearish (good for crops)
            precip_impact = "BEARISH"
            precip_note = f"Heavy rain ({total_precip:.1f}mm/week) - good moisture"
        else:
            drought_score = 0.05  # Neutral to slightly bullish
            precip_impact = "NEUTRAL"
            precip_note = f"Adequate moisture ({total_precip:.1f}mm/week)"
        
        # TEMPERATURE ANALYSIS
        # Critical temps: <-10°C (winter kill), >35°C (heat stress)
        month = datetime.now().month
        
        if month in [12, 1, 2]:  # Winter
            if min_temp < -10:
                temperature_score = 0.20  # Winter kill risk = very bullish
                temp_impact = "BULLISH"
                temp_note = f"Freeze risk! Min {min_temp:.1f}°C"
            elif min_temp < 0:
                temperature_score = 0.10
                temp_impact = "BULLISH"
                temp_note = f"Cold temps {min_temp:.1f}°C - some stress"
            else:
                temperature_score = 0.0
                temp_impact = "NEUTRAL"
                temp_note = f"Mild winter {avg_temp:.1f}°C"
        
        elif month in [5, 6, 7]:  # Growing season
            if max_temp > 35:
                temperature_score = 0.18  # Heat stress = bullish
                temp_impact = "BULLISH"
                temp_note = f"Heat stress! Max {max_temp:.1f}°C"
            elif max_temp > 30:
                temperature_score = 0.08
                temp_impact = "BULLISH"
                temp_note = f"Warm temps {max_temp:.1f}°C - slight stress"
            else:
                temperature_score = 0.0
                temp_impact = "NEUTRAL"
                temp_note = f"Good growing temps {avg_temp:.1f}°C"
        
        else:  # Spring/Fall
            temperature_score = 0.0
            temp_impact = "NEUTRAL"
            temp_note = f"Normal seasonal temps {avg_temp:.1f}°C"
        
        # COMBINED SCORE
        total_score = drought_score + temperature_score
        
        # Determine overall signal
        if total_score > 0.15:
            signal = "BULLISH"
            confidence = 0.75
        elif total_score > 0.08:
            signal = "BULLISH"
            confidence = 0.65
        elif total_score < -0.05:
            signal = "BEARISH"
            confidence = 0.60
        else:
            signal = "NEUTRAL"
            confidence = 0.50
        
        return {
            'signal': signal,
            'score': total_score,
            'confidence': confidence,
            'factors': [precip_note, temp_note],
            'precipitation': {
                'total_mm': total_precip,
                'impact': precip_impact,
                'score': drought_score
            },
            'temperature': {
                'avg_c': avg_temp,
                'max_c': max_temp,
                'min_c': min_temp,
                'impact': temp_impact,
                'score': temperature_score
            },
            'explanation': self._generate_explanation(signal, precip_note, temp_note)
        }
    
    def _generate_explanation(self, signal, precip_note, temp_note):
        """Generate human-readable explanation"""
        if signal == "BULLISH":
            return f"Weather concerns support prices - {precip_note}, {temp_note}"
        elif signal == "BEARISH":
            return f"Favorable weather pressures prices - {precip_note}"
        else:
            return f"Neutral weather impact - {precip_note}"
    
    def _get_neutral_signal(self):
        """Return neutral signal when data unavailable"""
        return {
            'signal': 'NEUTRAL',
            'score': 0.0,
            'confidence': 0.50,
            'factors': ['Weather data unavailable'],
            'explanation': 'Using seasonal average (no live data)'
        }
    
    def get_multi_region_signal(self):
        """
        Fetch weather from all key wheat regions and combine
        
        Returns:
            Combined weather signal
        """
        regional_signals = []
        
        print("   🌾 Fetching live weather for wheat regions...")
        
        for region_name, coords in self.wheat_regions.items():
            print(f"      {region_name}...", end=" ")
            
            weather_data = self.fetch_weather_data(coords, days=7)
            
            if weather_data:
                analysis = self.analyze_agricultural_impact(weather_data)
                analysis['region'] = region_name
                regional_signals.append(analysis)
                print(f"{analysis['signal']}")
            else:
                print("Failed")
        
        if not regional_signals:
            print("   ⚠️ No weather data available")
            return self._get_neutral_signal()
        
        # Combine regional signals
        avg_score = sum(s['score'] for s in regional_signals) / len(regional_signals)
        bullish_count = sum(1 for s in regional_signals if s['signal'] == 'BULLISH')
        
        # Determine combined signal
        if bullish_count >= 3:  # Majority bullish
            signal = 'BULLISH'
            confidence = 0.70
        elif bullish_count >= 2:
            signal = 'BULLISH'
            confidence = 0.60
        elif avg_score < -0.05:
            signal = 'BEARISH'
            confidence = 0.60
        else:
            signal = 'NEUTRAL'
            confidence = 0.55
        
        # Collect key factors
        all_factors = []
        for s in regional_signals:
            if s['signal'] != 'NEUTRAL':
                all_factors.append(f"{s['region']}: {s['factors'][0]}")
        
        return {
            'signal': signal,
            'score': avg_score,
            'confidence': confidence,
            'regional_count': len(regional_signals),
            'bullish_regions': bullish_count,
            'factors': all_factors[:3],  # Top 3 factors
            'explanation': self._generate_combined_explanation(signal, bullish_count, len(regional_signals))
        }
    
    def _generate_combined_explanation(self, signal, bullish_count, total_count):
        """Generate explanation for combined regional signal"""
        if signal == 'BULLISH':
            return f"Weather concerns in {bullish_count}/{total_count} wheat regions"
        elif signal == 'BEARISH':
            return f"Favorable weather across wheat belt"
        else:
            return f"Mixed weather conditions ({bullish_count}/{total_count} regions show concern)"

# Quick test
if __name__ == "__main__":
    analyzer = LiveWeatherAnalyzer()
    signal = analyzer.get_multi_region_signal()
    print(f"\n🌾 Combined Weather Signal: {signal}")
