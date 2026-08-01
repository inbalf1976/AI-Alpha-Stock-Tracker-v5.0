"""
news_scanner.py — Background Automated News Fetcher & LLM Interpreter
========================================================================

Runs as a GitHub Action on a scheduled cron (or manually via workflow_dispatch).
Fetches financial / agricultural / macro news from multiple free RSS feeds,
filters for high-impact keywords, sends flagged headlines to Gemini Flash,
and appends structured signals to `news_log.json`.

Outputs:
  - news_log.json (committed back to repo or saved as an artifact)
  - stdout logs (visible in GitHub Action runner)

Dependencies:
  - google-genai
  - feedparser
  - beautifulsoup4

Environment Variables:
  - GEMINI_API_KEY (optional, required only for LLM interpretation)

CHANGELOG (2026-07-31):
  Added best-effort FULL ARTICLE TEXT fetching for the top flagged
  headlines (see FULL_TEXT_FETCH_COUNT), instead of sending only
  headline titles to Gemini. Real-world fetch success will vary by
  publisher — some sites block non-browser requests regardless of
  headers used. Any fetch that fails or returns too little text
  (likely paywalled/blocked) falls back to headline-only for that
  article; the pipeline never breaks or blocks on a failed fetch.
  This is a best-effort enrichment, not a guaranteed capability.
"""

import os
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import feedparser
from bs4 import BeautifulSoup

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

# How many of the top flagged headlines to attempt full-article fetch
# for, per scan. Kept small deliberately: each fetch is a real network
# request with its own timeout, and the workflow job has a fixed
# overall timeout. This does NOT increase Gemini call count (still one
# call per scan) — only token count for that one call, which has
# plenty of headroom on the current model's TPM limit.
FULL_TEXT_FETCH_COUNT = 8
ARTICLE_FETCH_TIMEOUT = 8       # seconds per article
ARTICLE_MAX_CHARS = 3000        # per-article cap, keeps prompt size sane
ARTICLE_MIN_CHARS = 200         # below this, treat as blocked/paywalled/empty

# Custom User-Agent header for HTTP requests
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Direct RSS Feeds (Reliable, open endpoints)
RSS_FEEDS = {
    # Agricultural & Commodities (Google News routed to bypass 403 / Timeout blocks)
    "USDA News": "https://news.google.com/rss/search?q=site:usda.gov+when:1d&hl=en-US&gl=US&ceid=US:en",
    "AgWeb Markets": "https://news.google.com/rss/search?q=site:agweb.com+markets+when:1d&hl=en-US&gl=US&ceid=US:en",

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
    """Fetch and parse a single RSS feed with explicit HTTP headers and timeout."""
    headlines = []
    try:
        req = urllib.request.Request(url, headers=HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()

        feed = feedparser.parse(xml_data)
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
    """Fetch headlines concurrently across direct feeds + keyword search feeds."""
    all_headlines = []
    seen_titles = set()

    # Prepare target workload list
    targets = []
    for source, url in RSS_FEEDS.items():
        targets.append((source, url))

    for query in KEYWORD_QUERIES:
        url = GOOGLE_NEWS_BASE.format(query=query.replace(" ", "+"))
        targets.append((f"Google News ({query})", url))

    # Execute requests concurrently using 10 worker threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_target = {
            executor.submit(fetch_rss_feed, source, url): (source, url)
            for source, url in targets
        }
        for future in as_completed(future_to_target):
            feed_items = future.result()
            for item in feed_items:
                norm_title = item["title"].lower()
                if norm_title not in seen_titles:
                    seen_titles.add(norm_title)
                    all_headlines.append(item)

    return all_headlines

# ---------------------------------------------------------------------------
# FULL ARTICLE TEXT FETCHING (best-effort, 2026-07-31)
# ---------------------------------------------------------------------------

def fetch_article_text(url, timeout=ARTICLE_FETCH_TIMEOUT,
                        max_chars=ARTICLE_MAX_CHARS, min_chars=ARTICLE_MIN_CHARS):
    """
    Fetches and extracts the main article text from a URL, best-effort.
    Returns None (never raises) if the fetch fails, times out, or the
    page appears blocked/paywalled (too little extractable text) —
    callers must treat None as "fall back to headline-only", not as
    an error to surface. Real-world success rate will vary by
    publisher; some sites block non-browser requests regardless of
    headers used here.
    """
    try:
        req = urllib.request.Request(url, headers=HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            html = response.read()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        text = " ".join(text.split())  # normalize whitespace
        if len(text) < min_chars:
            return None
        return text[:max_chars]
    except Exception as e:
        logger.info(f"   Article fetch failed ({url[:60]}...): {type(e).__name__}")
        return None


def enrich_with_full_text(flagged_headlines, count=FULL_TEXT_FETCH_COUNT):
    """
    Attempts to fetch full article text for the top `count` flagged
    headlines, concurrently. Headlines beyond `count`, and any within
    it that fail to fetch, are left with full_text=None — the prompt
    builder falls back to title-only for those. Never raises; a
    total failure here just means the scan proceeds headline-only,
    same as before this feature existed.
    """
    targets = flagged_headlines[:count]
    if not targets:
        return flagged_headlines

    with ThreadPoolExecutor(max_workers=min(8, len(targets))) as executor:
        future_to_headline = {
            executor.submit(fetch_article_text, h["link"]): h
            for h in targets if h.get("link")
        }
        fetched = 0
        for future in as_completed(future_to_headline):
            headline = future_to_headline[future]
            text = future.result()
            headline["full_text"] = text
            if text:
                fetched += 1

    logger.info(f"   Full article text fetched: {fetched}/{len(targets)} attempted "
                f"(rest fell back to headline-only)")
    return flagged_headlines

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
    Send flagged headlines (with full article text where available) to
    Gemini for macro & commodity interpretation. Returns a structured
    dictionary with market impacts.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.info("   GEMINI_API_KEY not found in environment. Skipping LLM interpretation.")
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        # Build prompt — use full article text where we have it,
        # headline-only otherwise (fetch failures, or beyond the
        # top-N fetch cutoff, both leave full_text unset/None).
        entries_text = []
        for h in flagged_headlines[:15]:
            full_text = h.get("full_text")
            if full_text:
                entries_text.append(
                    f"- [{h['source']}] HEADLINE: {h['title']}\n  ARTICLE TEXT: {full_text}"
                )
            else:
                entries_text.append(f"- [{h['source']}] {h['title']}")
        titles_text = "\n".join(entries_text)

        prompt = f"""
        You are a senior macro and commodities analyst. Analyze these recent financial & agricultural news items.
        Some include the full article text (marked ARTICLE TEXT) for deeper context — use it when present;
        otherwise rely on the headline alone.

        {titles_text}

        Provide a concise analysis in JSON format with the following keys:
        - "summary": A 2-3 sentence overview of main market drivers.
        - "wheat_impact": "BULLISH", "BEARISH", or "NEUTRAL" with a 1-sentence reason.
        - "usd_ils_impact": "BULLISH", "BEARISH", or "NEUTRAL" with a 1-sentence reason.
        - "key_risk": Single main threat to watch today.

        Respond ONLY with raw valid JSON (no markdown ticks or wrapper text).
        """

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        text = response.text.strip()
        # Clean potential markdown formatting wrappers
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

    # 3. Best-effort full-text enrichment for the top flagged headlines
    if flagged:
        logger.info(f"Attempting full-article fetch for top {min(FULL_TEXT_FETCH_COUNT, len(flagged))} headlines...")
        flagged = enrich_with_full_text(flagged)

    # 4. LLM Interpretation
    llm_analysis = None
    if flagged:
        logger.info("Interpreting headlines with Gemini Flash...")
        llm_analysis = interpret_with_gemini(flagged)
        if not llm_analysis:
            logger.info("   No LLM signal this scan (no key configured, or call failed)")

    # 5. Save Record
    # NOTE: full_text is intentionally NOT persisted to news_log.json —
    # it's only used transiently to build the Gemini prompt this run.
    # Saving full article bodies to the repo long-term isn't needed
    # (the LLM's structured analysis already captures what mattered)
    # and would bloat the log file considerably.
    flagged_for_log = [
        {k: v for k, v in h.items() if k != "full_text"}
        for h in flagged[:15]
    ]
    scan_record = {
        "timestamp": now_iso,
        "total_scanned": len(headlines),
        "flagged_count": len(flagged),
        "flagged_headlines": flagged_for_log,
        "llm_analysis": llm_analysis
    }

    update_news_log(scan_record)

if __name__ == "__main__":
    main()
