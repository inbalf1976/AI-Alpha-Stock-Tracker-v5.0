Run python wheat_monitor_pro.py
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1778639543.296564    2334 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1778639545.116616    2334 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.

================================================================================
🌾 PROFESSIONAL WHEAT MONITOR - ULTIMATE EDITION v3.1 + FIXES
Time: 2026-05-13 02:32:25 UTC
Features: Ensemble AI + Weather + WASDE + Volume + Seasonal
================================================================================


📂 Loading state...
   ✓ State file exists
   ✓ Loaded - last_alert_date: 2026-05-10, daily_sent: True
📊 Fetching ZW=F...
   ℹ️  Using complete data through 2026-05-12
✓ Price: 674.00¢

🔬 Initializing advanced analyzers...
💾 Using CACHED weather & WASDE data (saves API calls)
  ✓ Using cached weather (age: 9h 20m)
  ✓ Weather: BULLISH (4/8 regions)
  ✓ Using cached WASDE (age: 9h 20m)
  ✓ WASDE: NEUTRAL
📡 Gathering signals...
  ✓ Seasonal: BULLISH
  ✓ Volume: BEARISH
  ✓ Context: ABOVE_NORMAL

🤖 Training ensemble AI (LSTM + RF + XGB)...
🤖 Training ensemble models...
 - Training LSTM...
 - Training Random Forest...
 - Training XGBoost...
✓ All models trained
🎯 Making ensemble prediction...

📊 BASE ENSEMBLE PREDICTION:
   Direction: UP
   Confidence: 50.0%
   Agreement: MAJORITY (2/3 UP)
   Models: LSTM=0.658, RF=0.382, XGB=0.717

⚡ Enhancing with fundamental factors...

🎯 FINAL ENHANCED PREDICTION:
   Direction: UP
   Base: 50.0%
   Enhanced: 53.8%
   Boost: +3.8%
   Details: Seasonal: +5.00%, Weather: +9.75%, Volume: -3.00%, Context: -8.00%

📢 Alert Check:
   Last alert: 2026-05-10T17:11:45.574549
   Last alert date: 2026-05-10 vs today: 2026-05-13
   Last direction: UP → Current: UP
   Last price: 619.0 → Current: 674.0
   → New trading day - SENDING daily alert for 2026-05-13

📢 Alert: Daily update for 2026-05-13

🔍 TELEGRAM DEBUG:
   Bot token: SET (length: 46)
   Chat ID: ***
   Message length: 721 chars
   Sending to Telegram API...
   Response status: 400
   Response body: {"ok":false,"error_code":400,"description":"Bad Request: can't parse entities: Can't find end of the entity starting at byte offset 782"}
   ❌ Telegram rejected the message!

❌ Alert failed - state NOT updated, will retry next run

💾 State saved:
   last_direction: UP
   last_price: 619.0
   last_alert_date: 2026-05-10
   last_alert_time: 2026-05-10T17:11:45.574549
   alerts_sent: 95

📊 Session Stats:
   Total alerts sent: 95
   Last check: 2026-05-13 02:32:32
================================================================================

