"""
NEWS SCANNER — FREE, NO-REGISTRATION HEADLINE SURFACING + LLM SIGNAL
========================================================================
Fetches recent wheat-relevant headlines from Google News RSS (free, no
API key, no registration) and a couple of direct agricultural news RSS
feeds, then uses Google's free-tier Gemini API to interpret them into
a structured bullish/bearish/neutral signal.

RUNS SILENTLY (2026-07-19): no Telegram alert is sent for scans. This
runs purely in the background to feed the forecast model via
news_signal_log.json — the user does not want to see raw headline
alerts, only the effect on the forecast itself.

HONEST SCOPE:
  The LLM interpretation is NEW and UNVALIDATED. It is wired into the
  weekly forecast (see wheat_monitor_pro.py's get_news_signal()) with
  a deliberately SMALL nudge weight — much smaller than the real,
  holdout-validated backtest nudge — precisely because it hasn't
  earned trust yet. Every signal call gets logged to
  news_signal_log.json in a scoreable format; score_news_signals.py
  checks these against what price actually did afterward.

KEYWORDS/HEURISTICS UPDATE (2026-07-20): expanded after reviewing and
individually fact-checking a series of real market-mechanics
explanations (tender terminology, leading indicators, verified real
events like the Aug 2026 Morocco duty suspension and the Aug 2026
Algeria/Saudi tenders). Two categories from that review — stocks-to-use
trend tracking and maritime charter-vessel tracking — were explicitly
NOT added, since they require structured trend data or vessel-tracking
data that free news RSS + a headline-reading LLM cannot realistically
deliver. Only the three categories confirmed buildable from real news
coverage were added: weather/drought language, food-inflation/bread-
unrest language, and IMF/World Bank food-security financing language.

SDK MIGRATION (2026-07-26): the old google-generativeai package is
deprecated — switched to the new google-genai package. Model stays
gemini-2.5-flash: this project's free-tier quota dashboard confirmed
gemini-2.0-flash has ZERO allocated quota on this account, while
gemini-2.5-flash has real, working free-tier quota (5 RPM/20 RPD) and
prior usage — so the original 404 was an old-SDK/endpoint issue, not
the model being genuinely unavailable to this project.

SETUP:
  1. pip install feedparser google-genai
  2. Get a free Gemini API key at https://aistudio.google.com (no
     credit card, email/Google account only)
  3. Set GEMINI_API_KEY as a GitHub secret
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

import feedparser

IL = ZoneInfo("Asia/Jerusalem")
NEWS_LOG_FILE = Path("news_log.json")
NEWS_SIGNAL_LOG_FILE = Path("news_signal_log.json")
LOOKBACK_HOURS = 10

# ── Search queries — Google News RSS, free, no key ────────────────────────────
SEARCH_QUERIES = [
    # Core wheat/supply coverage
    "wheat price",
    "wheat export",
    "Russia wheat export",
    "Ukraine grain export",
    "Black Sea grain",
    "USDA WASDE wheat",
    "wheat drought harvest",

    # Tender-trigger coverage (2026-07-20) — verified real search terms,
    # confirmed to surface actual dated events (Algeria OAIC Aug 2026,
    # Saudi GFSA Sep-Oct 2026) during today's fact-checking pass.
    "GASC wheat tender",
    "OAIC wheat tender",
    "wheat tender issued",
    "wheat prompt delivery",
    "CFR wheat price",

    # Freight/shipping index — the real, verified "marrab" mechanism:
    # importers time flash tenders to freight rate dips.
    "Baltic Dry Index",

    # Three confirmed-buildable leading indicators (2026-07-20) — see
    # module docstring for why only these three, not all five, made
    # the cut.
    "wheat soil moisture deficit",        # weather anomaly language
    "bread price protest flour shortage", # food inflation / unrest language
    "IMF food security loan wheat",       # FX inflow / financing language
]

# Direct agricultural RSS feeds as a supplementary source (verify these
# resolve correctly once run somewhere with real internet access — this
# sandbox can't test-fetch arbitrary URLs, so treat as candidates, not
# guaranteed-working, until confirmed on a real run).
DIRECT_FEEDS = [
    "https://www.usda.gov/rss/home.xml",
]

# Simple keyword flagging — NOT sentiment analysis, just "this headline
# probably matters more, look at it first."
HIGH_IMPACT_KEYWORDS = [
    "export ban", "export tax", "export restrict", "canal", "strait",
    "attack", "strike", "sanction", "war", "conflict", "blockade",
    "drought", "shortage", "crop failure", "frost", "flood",
    "import duty", "import ban", "tender reject", "tender cancel",
    "ending stocks", "baltic dry", "bread price", "flour shortage",
    "imf loan", "world bank",
]


def fetch_google_news(query, hours_back=LOOKBACK_HOURS):
    """Free Google News RSS search — no API key, no registration."""
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        cutoff = datetime.now(IL) - timedelta(hours=hours_back)
        results = []
        for entry in feed.entries:
            pub = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub = datetime(*entry.published_parsed[:6], tzinfo=ZoneInfo("UTC")).astimezone(IL)
            if pub is None or pub >= cutoff:
                results.append({
                    'title': entry.get('title', '(no title)'),
                    'link': entry.get('link', ''),
                    'published': pub.isoformat() if pub else None,
                    'source_query': query,
                })
        return results
    except Exception as e:
        print(f"   Google News fetch failed for '{query}': {e}")
        return []


def fetch_direct_feed(url):
    """Fetch a direct RSS feed URL — free, no key."""
    try:
        feed = feedparser.parse(url)
        return [{'title': e.get('title', '(no title)'), 'link': e.get('link', ''),
                 'published': None, 'source_query': f"direct:{url}"}
                for e in feed.entries[:10]]
    except Exception as e:
        print(f"   Direct feed fetch failed ({url}): {e}")
        return []


def flag_high_impact(headline):
    """Simple keyword match — highlighting heuristic only, not analysis."""
    lower = headline.lower()
    return [kw for kw in HIGH_IMPACT_KEYWORDS if kw in lower]


def deduplicate(items):
    """Remove near-duplicate headlines (same title appearing across queries)."""
    seen = set()
    unique = []
    for item in items:
        key = re.sub(r'\W+', '', item['title'].lower())[:60]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def interpret_with_gemini(headlines):
    """
    Sends collected headlines to Gemini's free tier, asking for a
    structured signal, enriched with real, fact-checked interpretive
    heuristics (2026-07-20) — not a generic bullish/bearish ask.
    Returns None if no API key configured or the call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("   No GEMINI_API_KEY set — skipping LLM interpretation")
        return None
    if not headlines:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        headline_list = "\n".join(f"- {h['title']}" for h in headlines[:20])
        prompt = (
            "You are analyzing news headlines for their likely impact on "
            "CBOT wheat futures prices. Use these interpretive patterns "
            "when relevant (real market dynamics, not rigid rules — use "
            "judgment about which apply to the actual headlines below):\n\n"
            "- Shrinking ending stocks / falling stocks-to-use ratio for "
            "major exporters (US, Russia, EU) -> bullish (tighter supply)\n"
            "- Growing ending stocks / large surplus reports -> bearish\n"
            "- Sharp Baltic Dry Index / freight rate spike -> bullish for "
            "delivered price pressure on importers (may trigger urgent "
            "buying); a freight rate crash -> bearish (buyers can delay)\n"
            "- A country REJECTING tender offers as too expensive, or few "
            "trading houses bidding -> bullish (sellers holding firm)\n"
            "- A country easily buying at LOWER prices than prior tenders, "
            "or many competing sellers -> bearish (oversupply pressure)\n"
            "- A major producer/importer removing an import duty or import "
            "ban (e.g. after a large domestic harvest) -> bearish ceiling\n"
            "- A country's domestic harvest missing government collection "
            "targets, forcing emergency/deficit buying -> bullish\n"
            "- Attacks, blockades, canal/strait closures affecting major "
            "export routes (Black Sea, etc.) -> bullish\n"
            "- Drought / soil moisture deficit reports during a country's "
            "critical growing window -> bullish (early harvest-failure "
            "signal, weeks-to-months before it becomes a tender)\n"
            "- Bread price spikes, flour shortages, or bread-related "
            "protests -> bullish (signals emergency government buying is "
            "likely coming soon)\n"
            "- A country receiving a new IMF loan or World Bank food-"
            "security grant -> bullish (fresh USD reserves specifically "
            "earmarked for food imports, a leading indicator of buying)\n\n"
            "Based on the headlines below, answer with a single JSON "
            "object and nothing else:\n"
            '{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", '
            '"confidence": <integer 0-100>, '
            '"key_phrase": "<the single most important phrase driving this, '
            'under 15 words>"}\n\n'
            "If the headlines contain nothing clearly relevant to wheat "
            "supply, demand, or trade, respond NEUTRAL with low confidence. "
            "Confidence should reflect how directly the headlines match "
            "one of the patterns above, not general certainty.\n\n"
            f"Headlines:\n{headline_list}"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text.strip()
        text = re.sub(r'^```json\s*|\s*```$', '', text.strip())

        parsed = json.loads(text)
        signal = parsed.get('signal', 'NEUTRAL').upper()
        confidence = int(parsed.get('confidence', 0))
        key_phrase = parsed.get('key_phrase', '')

        if signal not in ('BULLISH', 'BEARISH', 'NEUTRAL'):
            signal = 'NEUTRAL'
        confidence = max(0, min(100, confidence))

        return {'signal': signal, 'confidence': confidence, 'key_phrase': key_phrase}

    except Exception as e:
        print(f"   Gemini interpretation failed: {e}")
        return None


def log_news_signal(signal_result, current_price=None):
    """Logs the LLM's interpretation in a scoreable shape for score_news_signals.py."""
    log = []
    if NEWS_SIGNAL_LOG_FILE.exists():
        try:
            log = json.loads(NEWS_SIGNAL_LOG_FILE.read_text())
        except Exception:
            log = []

    log.append({
        'timestamp': datetime.now(IL).isoformat(),
        'signal': signal_result['signal'],
        'confidence': signal_result['confidence'],
        'key_phrase': signal_result['key_phrase'],
        'entry_price': current_price,
        'validated': False,
        'outcome': None,
        'pnl_cents': None,
    })

    NEWS_SIGNAL_LOG_FILE.write_text(json.dumps(log[-300:], indent=2))
    print(f"   Logged news signal: {signal_result['signal']} "
          f"({signal_result['confidence']}%) — {signal_result['key_phrase']}")


def get_current_price_for_logging():
    """Lightweight price fetch just for tagging the news signal log entry."""
    try:
        import yfinance as yf
        fast = yf.Ticker("ZW=F").fast_info
        return float(fast.get('last_price') or fast.get('lastPrice'))
    except Exception:
        return None


def main():
    print(f"News scan at {datetime.now(IL).isoformat()}")
    print(f"Scanning {len(SEARCH_QUERIES)} keyword queries + {len(DIRECT_FEEDS)} direct feeds...")

    all_items = []
    for q in SEARCH_QUERIES:
        all_items.extend(fetch_google_news(q))
    for f in DIRECT_FEEDS:
        all_items.extend(fetch_direct_feed(f))

    all_items = deduplicate(all_items)
    print(f"Found {len(all_items)} unique headlines in the last ~{LOOKBACK_HOURS}h")

    high_impact = [item for item in all_items if flag_high_impact(item['title'])]
    for item in high_impact:
        item['flagged_keywords'] = flag_high_impact(item['title'])

    print(f"High-impact flagged: {len(high_impact)}")

    print("Interpreting headlines with Gemini (free tier)...")
    llm_result = interpret_with_gemini(all_items)
    if llm_result:
        current_price = get_current_price_for_logging()
        log_news_signal(llm_result, current_price)
    else:
        print("   No LLM signal this scan (no key configured, or call failed)")

    # NOTE: no Telegram message sent — runs silently, feeds the model only.
    log = []
    if NEWS_LOG_FILE.exists():
        try:
            log = json.loads(NEWS_LOG_FILE.read_text())
        except Exception:
            log = []
    log.append({
        'scan_time': datetime.now(IL).isoformat(),
        'total_headlines': len(all_items),
        'high_impact_count': len(high_impact),
        'headlines': all_items,
    })
    NEWS_LOG_FILE.write_text(json.dumps(log[-200:], indent=2))
    print(f"Logged scan to {NEWS_LOG_FILE}")


if __name__ == "__main__":
    main()
