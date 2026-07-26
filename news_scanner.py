"""
news_scanner.py — Background Automated News Fetcher & LLM Interpreter
========================================================================

Runs as a GitHub Action on a 6-hour cron (or manually via workflow_dispatch).
Fetches financial / agricultural / macro news from multiple free RSS feeds,
filters for high-impact keywords, sends flagged headlines to Gemini Flash,
and appends structured signals to `news_log.json`.

Outputs:
  - news_log.json (committed back to repo or saved as an artifact)
  - stdout logs (visible in GitHub Action runner)

Dependencies:
  - google-genai
  - feedparser

Environment Variables:
  - GEMINI_API_KEY (optional, required only for LLM interpretation)

Note on Google GenAI SDK:
  Using `from google import genai`. The old `google.generativeai` is
  deprecated — switched to the new google-genai package. Model updated to
  gemini-2.5-flash for maximum cost efficiency and low-latency
  background news scans.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
import feedparser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("news_scanner")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

LOG_FILE = "news_log.json"
MAX_LOG_ENTRIES = 200  # Keep file manageable

# Custom User-Agent to prevent RSS feeds (like USDA) from closing connections
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Free RSS Feeds (No API keys required)
RSS_FEEDS = {
    # Agricultural & Commodities
    "USDA News": "https://www.usda.gov/rss/home.xml",
    "AgWeb Markets": "https://www.agweb.com/rss/markets",
    
    # Macro / Forex / Financial
    "Investing.com Forex": "https://www.investing.com/rss/news_1.rss",
    "Investing.com Commodities": "https://www.investing.com/rss/news_11.rss",
    "MarketWatch Top Stories": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "CNBC Economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    
    # Central Banks
    "Fed Reserve Press Releases": "https://www.federalreserve.gov/feeds/press_all.xml",
}

# Google News Query Feeds (Dynamic Keyword RSS)
GOOGLE_NEWS_BASE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

KEYWORD_QUERIES = [
    # Commodities / Wheat
    "wheat prices market",
    "grain export embargo",
    "USDA crop report wheat",
    "drought wheat harvest",
    "Black Sea grain corridor",
    
    # Currencies / ILS
    "Bank of Israel interest rate",
    "USD ILS shekel exchange rate",
    "Israel economy inflation",
    
    # Global Macro
    "Federal Reserve rate decision",
    "US dollar index DXY",
    "US inflation CPI PPI",
    "Middle East conflict oil supply",
    "Red Sea shipping disruption",
    "global supply chain crisis",
    "crude oil prices OPEC",
    "S&P 500 market crash rally",
]

# High-Impact Trigger Words (for initial filtering before sending to Gemini)
HIGH_IMPACT_KEYWORDS = [
    "wheat", "grain", "usda", "wasde", "crop", "drought", "harvest",
    "shekel", "ils", "bank of israel", "fed", "fomc", "rate hike", "rate cut",
    "cpi", "ppi", "inflation", "tariff", "embargo", "sanction", "opec", "crude",
    "oil", "geopolitical", "missile", "war", "red sea", "suez", "shipping",
    "black sea", "export ban", "yield", "treasury", "dxy", "recession"
]

# ---------------------------------------------------------------------------
# RSS FETCHING & PARSING
# ---------------------------------------------------------------------------

def fetch_rss_feed(source_name, url):
    """Fetch and parse a single RSS feed with custom headers."""
    headlines = []
    try:
        # Request feed with custom HTTP headers to avoid remote disconnection
        feed = feedparser.parse(url, request_headers=HTTP_HEADERS)
        for entry in feed.entries[:10]:  # Take top 10 per feed
            title = entry.get("title", "").strip()
            link = entry.get("link", "")
            published = entry.get("published", entry.get("updated", ""))
            
            if title:
                headlines.append({
                    "title": title,
                    "link": link,
                    "source": source_name,
                    "published": published
                })
    except Exception as e:
        logger.warning(f"   Direct feed fetch failed ({url}): {e}")
    return headlines


def fetch_all_headlines():
    """Fetch headlines from direct RSS feeds + Google News keyword queries."""
    all_headlines = []
    seen_titles = set()

    # 1. Direct Feeds
    for source, url in RSS_FEEDS.items():
        feed_items = fetch_rss_feed(source, url)
        for item in feed_items:
            # Deduplicate by lowercase title
            norm_title = item["title"].lower()
            if norm_title not in seen_titles:
                seen_titles.add(norm_title)
                all_headlines.append(item)

    # 2. Google News Keyword Feeds
    for query in KEYWORD_QUERIES:
        url = GOOGLE_NEWS_BASE.format(query=query.replace(" ", "+"))
        feed_items = fetch_rss_feed(f"Google News ({query})", url)
        for item in feed_items:
            norm_title = item["title"].lower()
            if norm_title not in seen_titles:
                seen_titles.add(norm_title)
                all_headlines.append(item)

    return all_headlines

# ---------------------------------------------------------------------------
# FILTERING & INTERPRETATION
# ---------------------------------------------------------------------------

def filter_high_impact(headlines):
    """Filter headlines that contain at least one high-impact keyword."""
    flagged = []
    for h in headlines:
        title_lower = h["title"].lower()
        matched = [kw for kw in HIGH_IMPACT_KEYWORDS if kw in title_lower]
        if matched:
            h["matched_keywords"] = matched
            flagged.append(h)
    return flagged


def interpret_with_gemini(flagged_headlines):
    """
    Send flagged headlines to Gemini 2.5 Flash for high-level macro impact interpretation.
    Returns a dict with overall sentiment, commodity impact, and forex impact.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.info("   GEMINI_API_KEY not found in environment. Skipping LLM interpretation.")
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        # Build prompt
        titles_text = "\n".join([f"- [{h['source']}] {h['title']}" for h in flagged_headlines[:15]])
        
        prompt = f"""
        You are a senior macro and commodities analyst. Analyze these recent financial & agricultural news headlines:

        {titles_text}

        Provide a concise analysis in JSON format with the following keys:
        - "summary": A 2-3 sentence overview of main market drivers.
        - "wheat_impact": "BULLISH", "BEARISH", or "NEUTRAL" with a 1-sentence reason.
        - "usd_ils_impact": "BULLISH", "BEARISH", or "NEUTRAL" with a 1-sentence reason.
        - "key_risk": Single main threat to watch today.

        Respond ONLY with raw valid JSON (no markdown ticks or wrapper text).
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()
        # Clean potential markdown formatting
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        return json.loads(text)

    except Exception as e:
        logger.error(f"   Gemini interpretation failed: {e}")
        return None

# ---------------------------------------------------------------------------
# LOG MANAGEMENT
# ---------------------------------------------------------------------------

def update_news_log(scan_data):
    """Load existing log file, prepend new scan data, trim history, and save."""
    log_data = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read existing {LOG_FILE}: {e}")

    # Prepend new scan
    log_data.insert(0, scan_data)

    # Trim to max length
    log_data = log_data[:MAX_LOG_ENTRIES]

    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Logged scan to {LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to write {LOG_FILE}: {e}")

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

def main():
    now_iso = datetime.now(timezone.utc).isoformat()
    logger.info(f"News scan started at {now_iso}")

    # 1. Fetch
    logger.info(f"Scanning {len(KEYWORD_QUERIES)} keyword queries + {len(RSS_FEEDS)} direct feeds...")
    headlines = fetch_all_headlines()
    logger.info(f"Found {len(headlines)} unique headlines in this scan batch")

    # 2. Filter
    flagged = filter_high_impact(headlines)
    logger.info(f"High-impact flagged headlines: {len(flagged)}")

    # 3. LLM Interpretation
    llm_analysis = None
    if flagged:
        logger.info("Interpreting headlines with Gemini Flash...")
        llm_analysis = interpret_with_gemini(flagged)
        if not llm_analysis:
            logger.info("   No LLM signal this scan (no key configured, or call failed)")

    # 4. Save Record
    scan_record = {
        "timestamp": now_iso,
        "total_scanned": len(headlines),
        "flagged_count": len(flagged),
        "flagged_headlines": flagged[:15],  # Save top 15 flagged headlines
        "llm_analysis": llm_analysis
    }

    update_news_log(scan_record)

if __name__ == "__main__":
    main()
