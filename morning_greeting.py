"""
Morning Greeting - Daily 9 AM Alert (Sun-Fri)
Sends motivational greeting with symbol image
"""

import requests
import os
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_photo_telegram(image_path, caption):
    """Send photo with caption via Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram not configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        
        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'caption': caption,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, files=files, data=data, timeout=10)
            return response.status_code == 200
    
    except Exception as e:
        print(f"Telegram photo error: {e}")
        return False

def send_text_telegram(message):
    """Send text message via Telegram (fallback if image fails)"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except:
        return False

def get_daily_quote():
    """Get motivational quote for the day"""
    quotes = [
        "The trend is your friend until the end when it bends.",
        "Risk comes from not knowing what you're doing. - Warren Buffett",
        "In trading, the impossible happens about twice a year.",
        "The market can stay irrational longer than you can stay solvent.",
        "Plan your trade, trade your plan.",
        "The goal of a successful trader is to make the best trades. Money is secondary.",
        "Every battle is won before it is fought. - Sun Tzu",
        "Discipline is the bridge between goals and accomplishment."
    ]
    
    # Rotate quote based on day of year
    day_of_year = datetime.now().timetuple().tm_yday
    return quotes[day_of_year % len(quotes)]

def main():
    print(f"\n{'='*60}")
    print(f"☀️ MORNING GREETING - {datetime.now().strftime('%A, %B %d, %Y')}")
    print(f"{'='*60}\n")
    
    # Get day info
    now = datetime.now()
    day_name = now.strftime('%A')
    date_str = now.strftime('%B %d, %Y')
    
    # Check if it's a weekday (Mon-Fri) or Sunday
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    if weekday == 5:  # Saturday
        print("📅 Saturday - Market closed, skipping greeting")
        return
    
    # Get daily quote
    quote = get_daily_quote()
    
    # Create greeting message
    caption = f"""
☀️ *Good Morning Sir* ☀️

📅 {day_name}, {date_str}

💭 _{quote}_

🌾 *Wheat Monitor Status:*
✅ Professional System Active
✅ Ensemble AI Running
✅ 24/7 Cloud Monitoring

_Ready for another day of trading!_ 📈
"""
    
    # Try to send with image
    print("📤 Sending morning greeting...")
    
    image_path = "morning_symbol.png"
    
    if os.path.exists(image_path):
        success = send_photo_telegram(image_path, caption)
        if success:
            print("✅ Morning greeting with image sent!")
        else:
            print("⚠️ Image send failed, trying text only...")
            if send_text_telegram(caption):
                print("✅ Text greeting sent!")
            else:
                print("❌ Greeting failed")
    else:
        print("⚠️ Image not found, sending text only...")
        if send_text_telegram(caption):
            print("✅ Text greeting sent!")
        else:
            print("❌ Greeting failed")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
