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

CHANGELOG (2026-08-08):
  1. Expanded HIGH_IMPACT_KEYWORDS after a loss_forensics.py review
     showed real losses whose driving news wasn't reliably being
     flagged. Added two groups:
       - Missing topic words: hormuz, escalation, heatwave/heat wave.
         (Other suggested words like "crop damage", "grain corridor",
         "yield cut" were NOT added as separate entries since "crop",
         "grain", and "yield" already catch them via substring match.)
       - Anticipatory/developing-event language: words that hint a
         disruption is forming or an event is about to resolve,
         BEFORE it's fully priced in — "warns of", "threatens to"
         (early warning), "halts exports"/"suspends shipping"/
         "blockade" (concrete early-stage disruption), "declares
         unsafe" (the original Black Sea navigation example),
         "ceasefire"/"peace deal"/"peace agreement" (de-escalation is
         just as much a "change coming" signal as escalation),
         "secretly" (low-noise, catches under-the-radar developments).
       Deliberately did NOT add broad generic verbs like "fire",
       "announce", or "signing" — these match almost anything and
       would flood flagged headlines with noise, the same problem
       "fed" already causes with routine Fed enforcement-action press
       releases.
       (Also considered and explicitly REJECTED: adding "Kpler" and
       "Energy Aspects" as named-source keywords — both require paid
       registration to access any underlying data, so a headline
       merely citing them gives no verifiable signal here. Not added.)
  2. Added fetch_windward_context() — a direct fetch of Windward AI's
     free, no-registration maritime chokepoint dashboard (Hormuz /
     Red Sea / Black Sea shipping status: blockade status, transit
     volumes vs. baseline, attack incidents, dark-shipping activity).
     This is NOT wired in as an RSS feed: the page is a live,
     JS-rendered status snapshot that gets overwritten in place, not
     a stream of discrete new items, so RSS's "new item" model doesn't
     fit it. Instead it's fetched fresh each scan (same direct-fetch
     pattern as fetch_article_text()) and its "BLUF — Bottom Line Up
     Front" summary section is extracted as fixed context appended to
     the Gemini prompt, independent of the keyword-flagged headline
     pipeline. Given how often Hormuz/Black Sea/Red Sea disruption has
     shown up as the actual driver behind real trading losses (see
     loss_forensics.py session, 2026-08-08), this gives a live,
     dated status ahead of when mainstream headlines catch up — the
     ~20h institutional-media lag already observed in this project.
     Like full_text, the raw snapshot is NOT persisted to
     news_log.json (transient prompt context only, would bloat the
     log) — only a boolean flag (maritime_context_included) is logged
     so it's visible in history whether this context was available
     for a given scan.
"""

import os
import json
import logging
import urllib.request
import urllib.error
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

# Sources whose headlines are routine/low-value boilerplate (e.g. Fed
# press releases approving individual bank mergers) rather than
# market-moving news. enrich_with_full_text() deprioritizes these so
# the limited fetch budget goes to sources actually worth fetching —
# see LOW_PRIORITY_FETCH_SOURCES below and its docstring note.
LOW_PRIORITY_FETCH_SOURCES = {"Fed Reserve Press Releases"}

# Windward AI maritime chokepoint dashboard — free, no registration,
# no API. See CHANGELOG (2026-08-08) above for why this is fetched
# directly rather than treated as an RSS source.
WINDWARD_URL = "https://insights.windward.ai/"
WINDWARD_FETCH_TIMEOUT = 10
WINDWARD_CONTEXT_MAX_CHARS = 2500   # BLUF summary only, not the full vessel-by-vessel tables
WINDWARD_CONTEXT_MIN_CHARS = 200    # below this, treat as blocked/empty/structure-changed

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
    "black sea", "export ban", "yield", "treasury", "dxy", "recession",
    # Added 2026-08-08 — missing topic words (see CHANGELOG above)
    "hormuz", "escalation", "heatwave", "heat wave",
    # Added 2026-08-08 — anticipatory/developing-event language, kept
    # as specific multi-word phrases so they don't fire independent of
    # any real topic (see CHANGELOG above for why "fire"/"announce"/
    # "signing" were deliberately left out)
    "declares unsafe", "warns of", "threatens to", "halts exports",
    "suspends shipping", "blockade", "secretly", "ceasefire",
    "peace deal", "peace agreement"
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
    except urllib.error.HTTPError as e:
        # 2026-08-01: was previously logged as a generic "HTTPError"
        # with no code/reason, making it impossible to tell a real
        # block (403) from a redirect issue, rate limit, or something
        # else. Now logs the actual status code and reason.
        logger.info(f"   Article fetch failed ({url[:60]}...): "
                    f"HTTP {e.code} {e.reason}")
        return None
    except urllib.error.URLError as e:
        logger.info(f"   Article fetch failed ({url[:60]}...): "
                    f"URLError {e.reason}")
        return None
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

    UPDATED 2026-08-08: previously took flagged_headlines[:count] in
    whatever order filter_high_impact() happened to produce. On scans
    where several routine Fed Reserve press releases (bank merger
    approvals, enforcement actions — see LOW_PRIORITY_FETCH_SOURCES)
    sorted first, they silently consumed the entire fetch budget,
    leaving genuinely market-moving Investing.com/MarketWatch articles
    further down the list never attempted at all. Now sorts
    low-priority sources to the back before taking the top `count`,
    so the limited fetch budget goes to the sources most likely to
    carry real market-moving detail worth the full-text fetch.
    """
    if not flagged_headlines:
        return flagged_headlines

    prioritized = sorted(
        flagged_headlines,
        key=lambda h: h.get("source") in LOW_PRIORITY_FETCH_SOURCES,
    )
    targets = prioritized[:count]
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
# WINDWARD MARITIME CHOKEPOINT CONTEXT (best-effort, 2026-08-08)
# ---------------------------------------------------------------------------

def fetch_windward_context(timeout=WINDWARD_FETCH_TIMEOUT,
                            max_chars=WINDWARD_CONTEXT_MAX_CHARS,
                            min_chars=WINDWARD_CONTEXT_MIN_CHARS):
    """
    Best-effort fetch of Windward AI's free, no-registration maritime
    chokepoint dashboard (Hormuz / Red Sea / Black Sea shipping
    status). This is a live status snapshot that gets overwritten in
    place each day, NOT a stream of discrete new items — see
    CHANGELOG (2026-08-08) for why this is fetched directly here
    rather than added as an RSS feed.

    Extracts just the "BLUF — Bottom Line Up Front" summary section
    (the daily top-line bullets) rather than the full page, which also
    contains large vessel-by-vessel tables too granular/noisy for a
    daily macro prompt.

    Returns None (never raises) on any fetch/parse failure, or if the
    page's structure has changed enough that the BLUF marker can't be
    found and the fallback slice is too short to be useful — callers
    must treat None as "no maritime context this scan", same pattern
    as fetch_article_text().
    """
    try:
        req = urllib.request.Request(WINDWARD_URL, headers=HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            html = response.read()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        text = " ".join(text.split())  # normalize whitespace

        # Anchor on "BLUF" — the page's own label for its daily
        # top-line summary bullets. If the site's structure changes
        # and this marker disappears, fall back to the start of the
        # page rather than silently returning nothing.
        marker = "BLUF"
        idx = text.find(marker)
        if idx == -1:
            idx = 0

        snippet = text[idx:idx + max_chars]
        if len(snippet) < min_chars:
            return None
        return snippet

    except urllib.error.HTTPError as e:
        logger.info(f"   Windward fetch failed: HTTP {e.code} {e.reason}")
        return None
    except urllib.error.URLError as e:
        logger.info(f"   Windward fetch failed: URLError {e.reason}")
        return None
    except Exception as e:
        logger.info(f"   Windward fetch failed: {type(e).__name__}")
        return None

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


def interpret_with_gemini(flagged_headlines, maritime_context=None):
    """
    Send flagged headlines (with full article text where available) to
    Gemini for macro & commodity interpretation. Returns a structured
    dictionary with market impacts.

    maritime_context: optional string from fetch_windward_context() —
    if present, appended to the prompt as fixed live shipping-status
    context (Hormuz/Red Sea/Black Sea), independent of the keyword-
    flagged headlines. See CHANGELOG (2026-08-08).
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

        maritime_block = ""
        if maritime_context:
            maritime_block = f"""

        LIVE MARITIME CHOKEPOINT STATUS (Windward AI, Hormuz/Red Sea/Black Sea shipping):
        {maritime_context}
        """

        prompt = f"""
        You are a senior macro and commodities analyst. Analyze these recent financial & agricultural news items.
        Some include the full article text (marked ARTICLE TEXT) for deeper context — use it when present;
        otherwise rely on the headline alone.

        {titles_text}
        {maritime_block}
        Provide a concise analysis in JSON format with the following keys:
        - "summary": A 2-3 sentence overview of main market drivers.
        - "wheat_impact": "BULLISH", "BEARISH", or "NEUTRAL" with a 1-sentence reason.
        - "usd_ils_impact": "BULLISH", "BEARISH", or "NEUTRAL" with a 1-sentence reason.
        - "key_risk": Single main threat to watch today.
        - "headline_overstated": true or false — for items where you have ARTICLE TEXT,
          does the headline's language (intensity, certainty, scale) go noticeably beyond
          what the article body actually describes? If no items have ARTICLE TEXT, set this
          to false. Base this ONLY on comparing headline framing to the article's own content
          — not on whether you think the underlying event itself is true or important.
        - "headline_vs_article_note": if headline_overstated is true, a 1-sentence note on
          which headline/article pair shows the gap and what the article actually said instead.
          If headline_overstated is false, use an empty string.

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

    # 3b. Best-effort maritime chokepoint context (Windward AI) — fetched
    # independent of whether any headline was keyword-flagged, since it's
    # cheap (one request) and gives live Hormuz/Red Sea/Black Sea status
    # even on a quiet news day.
    logger.info("Fetching maritime chokepoint context (Windward AI)...")
    maritime_context = fetch_windward_context()
    if maritime_context:
        logger.info(f"   Maritime context fetched ({len(maritime_context)} chars)")
    else:
        logger.info("   No maritime context this scan (fetch failed or page structure changed)")

    # 4. LLM Interpretation
    llm_analysis = None
    if flagged:
        logger.info("Interpreting headlines with Gemini Flash...")
        llm_analysis = interpret_with_gemini(flagged, maritime_context=maritime_context)
        if not llm_analysis:
            logger.info("   No LLM signal this scan (no key configured, or call failed)")

    # 5. Save Record
    # NOTE: full_text and the raw maritime_context snapshot are
    # intentionally NOT persisted to news_log.json — both are only
    # used transiently to build the Gemini prompt this run. Saving
    # full article bodies / dashboard snapshots to the repo long-term
    # isn't needed (the LLM's structured analysis already captures
    # what mattered) and would bloat the log file considerably. Only
    # a boolean flag is kept so scan history shows whether maritime
    # context was available that day.
    flagged_for_log = [
        {k: v for k, v in h.items() if k != "full_text"}
        for h in flagged[:15]
    ]
    scan_record = {
        "timestamp": now_iso,
        "total_scanned": len(headlines),
        "flagged_count": len(flagged),
        "flagged_headlines": flagged_for_log,
        "maritime_context_included": bool(maritime_context),
        "llm_analysis": llm_analysis
    }

    update_news_log(scan_record)

if __name__ == "__main__":
    main()
