"""
SIMPLE TELEGRAM TEST - Send a test message NOW
"""

import os
import requests
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print("=" * 80)
print("🧪 TELEGRAM TEST - IMMEDIATE SEND")
print(f"Time: {datetime.now()}")
print("=" * 80)

if not TELEGRAM_BOT_TOKEN:
    print("❌ TELEGRAM_BOT_TOKEN is missing!")
    exit(1)
    
if not TELEGRAM_CHAT_ID:
    print("❌ TELEGRAM_CHAT_ID is missing!")
    exit(1)

print(f"\n✅ Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...{TELEGRAM_BOT_TOKEN[-5:]}")
print(f"✅ Chat ID: {TELEGRAM_CHAT_ID}")

# Simple test message
message = f"""
🧪 TELEGRAM TEST MESSAGE

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

If you receive this, Telegram is working!

This is a plain text message with no formatting.
"""

print(f"\n📤 Sending test message to Telegram...")

url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
data = {
    "chat_id": TELEGRAM_CHAT_ID,
    "text": message
}

try:
    response = requests.post(url, data=data, timeout=10)
    
    print(f"\n📥 RESPONSE:")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.text}")
    
    if response.status_code == 200:
        print("\n✅ SUCCESS! Message sent to Telegram!")
        print("\n👉 CHECK YOUR TELEGRAM APP NOW!")
    else:
        print("\n❌ FAILED! Telegram rejected the message")
        print(f"   Error: {response.json().get('description', 'Unknown error')}")
        
except Exception as e:
    print(f"\n❌ EXCEPTION: {e}")

print("=" * 80)
