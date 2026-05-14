"""
MINIMAL TELEGRAM TEST - Proves new code is running
Version: TEST-2026-05-13-v1
"""

import os
import requests
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print("=" * 80)
print("🧪 TELEGRAM TEST SCRIPT - VERSION TEST-2026-05-13-v1")
print(f"Time: {datetime.now()}")
print("=" * 80)

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("❌ Telegram credentials missing")
    exit(1)

# Simple plain text message
message = """
🧪 TEST MESSAGE - VERSION TEST-2026-05-13-v1

This is a plain text test.
No markdown formatting.
No special characters.

If you receive this, the new code is running!

Time: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

print(f"\n📤 Sending test message...")
print(f"   Message length: {len(message)} chars")
print(f"   Plain text mode: YES (no parse_mode)")

url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}

response = requests.post(url, data=data, timeout=10)

print(f"\n📥 Response:")
print(f"   Status: {response.status_code}")
print(f"   Body: {response.text}")

if response.status_code == 200:
    print("\n✅ SUCCESS! New code is working!")
else:
    print("\n❌ FAILED! Error in response")

print("=" * 80)
