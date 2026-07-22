"""
NEWS SCANNER — FREE, NO-REGISTRATION HEADLINE SURFACING + LLM SIGNAL
========================================================================
Fetches recent wheat-relevant headlines from Google News RSS (free, no
API key, no registration) and a couple of direct agricultural news RSS
feeds, then uses Google's free-tier Gemini API to interpret them into
a structured bullish/bearish/neutral signal — the genuine "read the
news like a human would" capability discussed but not built earlier
today, now possible with a free API key.

HONEST SCOPE — READ BEFORE TRUSTING THIS:
  The LLM interpretation is NEW and UNVALIDATED. It is wired into the
  weekly forecast (see wheat_monitor_pro.py's get_news_signal()) with
  a deliberately SMALL nudge weight — much smaller than the real,
  holdout-validated backtest nudge — precisely because it hasn't
  earned trust yet. Every signal call gets logged to
  news_signal_log.json in a scoreable format; score_news_signals.py
  (companion script) checks these against what price actually did
  afterward, the same reward/punishment pattern already used to
  validate ConvictionGate's conditions. Only after real weeks of
  evidence accumulate should this nudge's weight increase — do not
  manually raise it based on a good week or two; that's the same
  small-sample trap the fake 100% tier was built on.

SETUP: 
  1. pip install feedparser google-generativeai
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
NEWS_SIGNAL_LOG_FILE = Path("news_signal_log.json")  # scoreable LLM signal history
LOOKBACK_HOURS = 10

# Keyword searches run against Google News RSS — free, no key, keyword-driven
SEARCH_QUERIES = [
    "wheat price",
    "wheat export",
    "Russia wheat export",
    "Ukraine grain export",
    "Black Sea grain",
    "USDA WASDE wheat",
    "wheat drought harvest",
]

# Direct agricultural RSS feeds as a supplementary source (verify these
# resolve correctly once run somewhere with real internet access — this
# sandbox can't test-fetch arbitrary URLs, so treat as candidates, not
# guaranteed-working, until confirmed on a real run).
DIRECT_FEEDS = [
    "https://www.usda.gov/rss/home.xml",
]

# Simple keyword flagging — NOT sentiment analysis, just "this headline
# probably matters more, look at it first." Calibrated from real events
# this week (Black Sea/canal closure, export restrictions).
HIGH_IMPACT_KEYWORDS = [
    "export ban", "export tax", "export restrict", "canal", "strait",
    "attack", "strike", "sanction", "war", "conflict", "blockade",
    "drought", "shortage", "crop failure", "frost", "flood",
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
    matched = [kw for kw in HIGH_IMPACT_KEYWORDS if kw in lower]
    return matched


def interpret_with_gemini(headlines):
    """
    Sends collected headlines to Gemini's free tier, asking for a
    structured signal. Returns None if no API key configured or the
    call fails — caller must treat that as "no signal available",
    never silently guess.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("   No GEMINI_API_KEY set — skipping LLM interpretation")
        return None
    if not headlines:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        headline_list = "\n".join(f"- {h['title']}" for h in headlines[:20])
        prompt = (
            "You are analyzing news headlines for their likely impact on "
            "CBOT wheat futures prices. Based ONLY on the headlines below, "
            "answer with a single JSON object and nothing else:\n"
            '{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", '
            '"confidence": <integer 0-100>, '
            '"key_phrase": "<the single most important phrase driving this, '
            'under 15 words>"}\n\n'
            "If the headlines contain nothing clearly relevant to wheat "
            "supply, demand, or trade, respond NEUTRAL with low confidence.\n\n"
            f"Headlines:\n{headline_list}"
        )

        response = model.generate_content(prompt)
        text = response.text.strip()
        # Strip markdown code fences if the model added them anyway
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
    """
    Logs the LLM's interpretation in the SAME scoreable shape as
    prediction_log.json, so a companion scorer (score_news_signals.py)
    can check it against real price outcomes later — the actual
    reward/punishment mechanism for this new, unvalidated signal.
    """
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


def load_log():
    if NEWS_LOG_FILE.exists():
        try:
            return json.loads(NEWS_LOG_FILE.read_text())
        except Exception:
            pass
    return []


def save_log(log):
    # keep the log from growing unbounded — retain last 200 scan runs
    NEWS_LOG_FILE.write_text(json.dumps(log[-200:], indent=2))


def send_telegram(message):
    import os
    import requests
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram not configured — printing instead:\n")
        print(message)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=10
        )
        success = r.status_code == 200
        print(f"Telegram: {'sent' if success else 'failed'} ({r.status_code})")
        return success
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


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

    high_impact = []
    regular = []
    for item in all_items:
        matched = flag_high_impact(item['title'])
        if matched:
            item['flagged_keywords'] = matched
            high_impact.append(item)
        else:
            regular.append(item)

    # ── LLM interpretation (2026-07-19) ──────────────────────────────────
    print("Interpreting headlines with Gemini (free tier)...")
    llm_result = interpret_with_gemini(all_items)
    signal_line = ""
    if llm_result:
        current_price = get_current_price_for_logging()
        log_news_signal(llm_result, current_price)
        signal_line = (
            f"\n🤖 LLM SIGNAL (UNVALIDATED — see score_news_signals.py): "
            f"{llm_result['signal']} ({llm_result['confidence']}%)\n"
            f"   \"{llm_result['key_phrase']}\"\n"
        )
    else:
        signal_line = "\n🤖 LLM signal unavailable this scan (no key or call failed)\n"

    # Build message — high-impact first, capped so it stays readable
    lines = [f"📰 WHEAT NEWS SCAN — {datetime.now(IL).strftime('%Y-%m-%d %H:%M')} Israel\n"]
    lines.append(signal_line)

    if high_impact:
        lines.append("⚠️ HIGH-IMPACT (keyword-flagged, not sentiment-scored):")
        for item in high_impact[:8]:
            lines.append(f"  • {item['title']}")
            lines.append(f"    [{', '.join(item['flagged_keywords'])}] {item['link']}")
    else:
        lines.append("⚠️ No high-impact keyword matches this scan.")

    if regular:
        lines.append(f"\n📋 Other headlines ({len(regular)} total, showing up to 5):")
        for item in regular[:5]:
            lines.append(f"  • {item['title']}")

    if not all_items:
        lines.append("No headlines found this scan — check feed URLs are resolving correctly.")

    lines.append(f"\nHeadlines are raw surfacing. The LLM signal above is NEW and "
                 f"UNVALIDATED — weighted very lightly in the forecast until it "
                 f"proves itself over real weeks (see score_news_signals.py).")

    message = "\n".join(lines)
    # NOTE (2026-07-19): no longer sent to Telegram — this scan runs
    # silently in the background purely to feed the forecast model
    # (via news_signal_log.json). Printed to the Action log only, for
    # your own debugging, not sent as a user-facing alert.
    print("\n" + message)

    # Log this scan for future reward/punishment use once an LLM interpreter exists
    log = load_log()
    log.append({
        'scan_time': datetime.now(IL).isoformat(),
        'total_headlines': len(all_items),
        'high_impact_count': len(high_impact),
        'headlines': all_items,
    })
    save_log(log)
    print(f"Logged scan to {NEWS_LOG_FILE}")


if __name__ == "__main__":
    main()
