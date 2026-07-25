import json
import os
from google import genai
from google.genai.errors import APIError

def interpret_with_gemini(headline, summary="", api_key=None):
    """
    Interprets a news item for wheat market sentiment using Gemini API.
    Uses the modern google-genai SDK and gemini-2.0-flash model.
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Returning NEUTRAL.")
        return {"signal": "NEUTRAL", "confidence": 0, "key_phrase": "Missing API key"}

    # Construct prompt with wheat market context
    prompt = f"""
Analyze the following wheat market news item and classify its price impact.

Headline: {headline}
Summary: {summary}

Context/Heuristics:
- Bearish indicators: Large harvests, export bans lifted, bumper crops, weakening freight/tender prices.
- Bullish indicators: Droughts, frost damage, export taxes/restrictions, high tender prices, geopolitical disruptions.

Return ONLY a valid raw JSON object with no markdown formatting:
{{
  "signal": "BULLISH", "BEARISH", or "NEUTRAL",
  "confidence": integer between 0 and 100,
  "key_phrase": "brief reason summary"
}}
"""

    try:
        # Initialize the new SDK client
        client = genai.Client(api_key=api_key)
        
        # Call Gemini 2.0 Flash
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Extract text response
        raw_text = response.text.strip()
        
        # Strip markdown formatting if returned
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
        return json.loads(raw_text)

    except APIError as e:
        print(f"Gemini API Error: {e}")
        return {"signal": "NEUTRAL", "confidence": 0, "key_phrase": f"API Error: {str(e)}"}
    except Exception as e:
        print(f"Gemini interpretation failed: {e}")
        return {"signal": "NEUTRAL", "confidence": 0, "key_phrase": f"Error: {str(e)}"}
